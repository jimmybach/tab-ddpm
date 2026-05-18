import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert train/test CSV files into TabDDPM dataset format.'
    )
    parser.add_argument('--train-csv', required=True, type=Path)
    parser.add_argument('--test-csv', required=True, type=Path)
    parser.add_argument('--target', required=True, help='Target column name')
    parser.add_argument('--dataset-name', required=True, help='Output dataset name')
    parser.add_argument(
        '--task-type',
        required=True,
        choices=['regression', 'binclass', 'multiclass'],
        help='TabDDPM task type',
    )
    parser.add_argument(
        '--categorical-cols',
        nargs='*',
        default=None,
        help='Explicit categorical feature columns. If omitted, object/category/bool columns are used.',
    )
    parser.add_argument(
        '--drop-cols',
        nargs='*',
        default=[],
        help='Columns to drop before conversion',
    )
    parser.add_argument('--val-size', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--data-root',
        type=Path,
        default=Path('data'),
        help='Root directory where the converted dataset folder will be written',
    )
    parser.add_argument(
        '--exp-root',
        type=Path,
        default=Path('exp'),
        help='Root directory where a starter config will be written',
    )
    parser.add_argument(
        '--device',
        default='cpu',
        help='Device to place into the generated config, e.g. cpu, cuda:0, mps',
    )
    parser.add_argument(
        '--normalization',
        default='quantile',
        choices=['quantile', 'standard', 'minmax', '__none__'],
        help='Training normalization to place into the generated config',
    )
    return parser.parse_args()


def infer_categorical_columns(df, target, explicit_cats):
    feature_cols = [col for col in df.columns if col != target]
    if explicit_cats is not None:
        missing = sorted(set(explicit_cats) - set(feature_cols))
        if missing:
            raise ValueError(f'Unknown categorical columns: {missing}')
        return explicit_cats

    categorical_cols = []
    for col in feature_cols:
        dtype = df[col].dtype
        if (
            pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_categorical_dtype(dtype)
            or pd.api.types.is_bool_dtype(dtype)
        ):
            categorical_cols.append(col)
    return categorical_cols


def encode_target(task_type, train_y, val_y, test_y):
    if task_type == 'regression':
        return (
            train_y.astype(float).to_numpy(),
            val_y.astype(float).to_numpy(),
            test_y.astype(float).to_numpy(),
            None,
        )

    combined = pd.concat([train_y, val_y, test_y], axis=0).astype(str)
    categories = pd.Index(pd.unique(combined))
    mapping = {value: idx for idx, value in enumerate(categories)}

    def encode(series):
        encoded = series.astype(str).map(mapping)
        if encoded.isna().any():
            raise ValueError('Failed to encode some target values')
        return encoded.astype(int).to_numpy()

    n_classes = len(categories)
    if task_type == 'binclass' and n_classes != 2:
        raise ValueError(f'Expected 2 classes for binclass, found {n_classes}')
    if task_type == 'multiclass' and n_classes < 3:
        raise ValueError(f'Expected at least 3 classes for multiclass, found {n_classes}')

    return encode(train_y), encode(val_y), encode(test_y), n_classes


def dataframe_to_arrays(df, numeric_cols, categorical_cols):
    x_num = None
    x_cat = None

    if numeric_cols:
        x_num = df[numeric_cols].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)

    if categorical_cols:
        x_cat = (
            df[categorical_cols]
            .fillna('__nan__')
            .astype(str)
            .replace({'nan': '__nan__'})
            .to_numpy(dtype=str)
        )

    return x_num, x_cat


def save_array_if_present(path, array):
    if array is not None:
        np.save(path, array)


def write_info_json(path, dataset_name, task_type, n_classes, counts, n_num_features, n_cat_features):
    info = {
        'task_type': task_type,
        'name': dataset_name,
        'id': dataset_name,
        'train_size': counts['train'],
        'val_size': counts['val'],
        'test_size': counts['test'],
        'n_num_features': n_num_features,
        'n_cat_features': n_cat_features,
    }
    if n_classes is not None:
        info['n_classes'] = n_classes

    path.write_text(json.dumps(info, indent=4) + '\n')


def build_config_text(dataset_name, task_type, n_num_features, n_classes, device, normalization, train_size):
    is_y_cond = 'true' if task_type != 'regression' else 'false'
    num_classes = n_classes if n_classes is not None else 0
    sample_num = max(train_size, 1000)

    return f"""seed = 0
parent_dir = "exp/{dataset_name}/check"
real_data_path = "data/{dataset_name}/"
model_type = "mlp"
num_numerical_features = {n_num_features}
device = "{device}"

[model_params]
is_y_cond = {is_y_cond}
num_classes = {num_classes}

[model_params.rtdl_params]
d_layers = [
    256,
    256,
]
dropout = 0.0

[diffusion_params]
num_timesteps = 1000
gaussian_loss_type = "mse"
scheduler = "cosine"

[train.main]
steps = 1000
lr = 0.001
weight_decay = 1e-05
batch_size = 4096

[train.T]
seed = 0
normalization = "{normalization}"
num_nan_policy = "__none__"
cat_nan_policy = "__none__"
cat_min_frequency = "__none__"
cat_encoding = "__none__"
y_policy = "default"

[sample]
num_samples = {sample_num}
batch_size = 10000
seed = 0

[eval.type]
eval_model = "catboost"
eval_type = "synthetic"

[eval.T]
seed = 0
normalization = "__none__"
num_nan_policy = "__none__"
cat_nan_policy = "__none__"
cat_min_frequency = "__none__"
cat_encoding = "__none__"
y_policy = "default"
"""


def main():
    args = parse_args()

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    missing_columns = sorted(set(train_df.columns) ^ set(test_df.columns))
    if missing_columns:
        raise ValueError(f'Train/test columns differ: {missing_columns}')
    if args.target not in train_df.columns:
        raise ValueError(f'Target column {args.target!r} not found')

    train_df = train_df.drop(columns=args.drop_cols, errors='ignore')
    test_df = test_df.drop(columns=args.drop_cols, errors='ignore')

    if args.task_type == 'regression':
        stratify = None
    else:
        stratify = train_df[args.target]

    train_split, val_split = train_test_split(
        train_df,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=stratify,
    )

    combined_features = pd.concat(
        [
            train_split.drop(columns=[args.target]),
            val_split.drop(columns=[args.target]),
            test_df.drop(columns=[args.target]),
        ],
        axis=0,
        ignore_index=True,
    )
    categorical_cols = infer_categorical_columns(combined_features, args.target, args.categorical_cols)
    numeric_cols = [col for col in combined_features.columns if col not in categorical_cols]

    train_y, val_y, test_y, n_classes = encode_target(
        args.task_type,
        train_split[args.target],
        val_split[args.target],
        test_df[args.target],
    )

    train_x_num, train_x_cat = dataframe_to_arrays(train_split, numeric_cols, categorical_cols)
    val_x_num, val_x_cat = dataframe_to_arrays(val_split, numeric_cols, categorical_cols)
    test_x_num, test_x_cat = dataframe_to_arrays(test_df, numeric_cols, categorical_cols)

    dataset_dir = args.data_root / args.dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    save_array_if_present(dataset_dir / 'X_num_train.npy', train_x_num)
    save_array_if_present(dataset_dir / 'X_num_val.npy', val_x_num)
    save_array_if_present(dataset_dir / 'X_num_test.npy', test_x_num)
    save_array_if_present(dataset_dir / 'X_cat_train.npy', train_x_cat)
    save_array_if_present(dataset_dir / 'X_cat_val.npy', val_x_cat)
    save_array_if_present(dataset_dir / 'X_cat_test.npy', test_x_cat)
    np.save(dataset_dir / 'y_train.npy', train_y)
    np.save(dataset_dir / 'y_val.npy', val_y)
    np.save(dataset_dir / 'y_test.npy', test_y)

    write_info_json(
        dataset_dir / 'info.json',
        args.dataset_name,
        args.task_type,
        n_classes,
        {'train': len(train_split), 'val': len(val_split), 'test': len(test_df)},
        len(numeric_cols),
        len(categorical_cols),
    )

    exp_dir = args.exp_root / args.dataset_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    config_path = exp_dir / 'config.toml'
    config_path.write_text(
        build_config_text(
            args.dataset_name,
            args.task_type,
            len(numeric_cols),
            n_classes,
            args.device,
            args.normalization,
            len(train_split),
        )
    )

    print(f'Wrote dataset to: {dataset_dir}')
    print(f'Wrote starter config to: {config_path}')
    print(f'Numeric columns ({len(numeric_cols)}): {numeric_cols}')
    print(f'Categorical columns ({len(categorical_cols)}): {categorical_cols}')


if __name__ == '__main__':
    main()
