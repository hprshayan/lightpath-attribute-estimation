import pathlib
from typing import Callable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from src.compressor import compressor_performance_calculator
from src.constants import DPI, EMBEDDING_DEMO_DIM, SEED, TEST_DEPICTION_COUNT
from src.preprocessing import MyArrayLike


def print_announcement(message: str) -> None:
    padded_message = " " + message + " "
    MAX_LEN = 102
    msg_len = len(padded_message)
    half_extra_char_count = (MAX_LEN - msg_len) // 2
    if msg_len > MAX_LEN:
        raise ValueError(f'Speak less but act more. Message must be less than {MAX_LEN - 2} characters.')
    print()
    print('#' * MAX_LEN)
    print(
        '#' * half_extra_char_count,
        padded_message,
        '#' * (MAX_LEN - half_extra_char_count - msg_len),
        sep='',
    )
    print('#' * MAX_LEN)


def create_regression_report(
    scenario_title: str,
    method: str,
    unit: str,
    r2_score: float,
    feature_test: pd.DataFrame,
    target_test: pd.Series,
    mode_test: pd.Series,
    pipeline: Callable,
    label_scaler: Callable,
    logger,
) -> None:
    predictions = label_scaler.inv_transform(pipeline(feature_test[:TEST_DEPICTION_COUNT])).to_numpy().flatten()
    targets = target_test[:TEST_DEPICTION_COUNT]
    modes = mode_test[:TEST_DEPICTION_COUNT]
    indices = targets.index
    result_demo_list = [
        ' ' * 2
        + 'index'
        + ' ' * 10
        + 'Mode'
        + ' ' * 13
        + 'prediction'
        + ' ' * 10
        + 'target'
        + ' ' * 10
        + '|target-prediction|'
    ]
    result_demo_list += [
        f"  {i:<14} {m:<16} {p:<19.1f} {t:<15.1f} {abs(t-p):.1f}"
        for i, m, p, t in zip(indices, modes, predictions, targets)
    ]
    logger.info(f'{scenario_title} regression with {method} approach is done.')
    logger.info(f'test score (coefficeint of determination): {r2_score:.5f}')
    logger.info(f'here are some predictions (with {unit} as unit):')
    logger.info('\n'.join(result_demo_list))


def create_compression_report(
    method: str,
    original_dim: int,
    compressed_dim: int,
    features: MyArrayLike,
    compressor_fwd_path: Callable,
    logger,
) -> None:
    compression_ratio, reconstruction_mae = compressor_performance_calculator(
        original_dim, compressed_dim, compressor_fwd_path(features), features
    )
    logger.info(f'compression with {method} method is done.')
    logger.info('\t'.join([f'compression ratio: {compression_ratio}', f'reconstruction MAE: {reconstruction_mae}']))
    logger.info(f'compressed data dimension: {compressed_dim}')


def create_classification_report(
    scenario_title: str,
    method: str,
    unit: str,
    confusion_mat: MyArrayLike,
    predictions: MyArrayLike,
    targets: MyArrayLike,
    logger,
) -> None:
    indices = targets.index[:TEST_DEPICTION_COUNT]
    predication_labels = predictions.astype(int) + 1
    target_labels = targets.astype(int) + 1
    result_demo_list = [' ' * 2 + 'index' + ' ' * 10 + 'prediction' + ' ' * 10 + 'target']
    result_demo_list += [f"  {i:<14} {p:<19} {t:<15}" for i, p, t in zip(indices, predication_labels, target_labels)]
    c_mat_df = pd.DataFrame(confusion_mat)
    c_mat_df.columns = ['1 dBm (prediction)', '2 dBm (prediction)']
    c_mat_df.index = ['1 dBm (target)', '2 dBm (target)']
    accuracy = (
        c_mat_df['1 dBm (prediction)']['1 dBm (target)'] + c_mat_df['2 dBm (prediction)']['2 dBm (target)']
    ) / c_mat_df.sum().sum()
    logger.info(f'{scenario_title} classification with {method} approach is done.')
    logger.info(f'confusion matrix:')
    logger.info(c_mat_df)
    logger.info(f'accuracy: {accuracy}')
    logger.info(f'here are some predictions (with {unit} as unit):')
    logger.info('\n'.join(result_demo_list))


def grouping_helper_function(df: pd.Series):
    reset_index_df = df.reset_index(drop=True)
    return list(reset_index_df.groupby(reset_index_df))


def calculate_normalized_embeddings(train_data: pd.DataFrame, test_data: pd.DataFrame) -> np.ndarray:
    pca = PCA(n_components=EMBEDDING_DEMO_DIM, random_state=SEED)
    pca.fit(train_data)
    embeddings = pca.transform(test_data)
    return (embeddings - embeddings.mean()) / embeddings.std()


def low_dim_embedding_plot(
    train_features: MyArrayLike,
    test_features: MyArrayLike,
    test_targets: MyArrayLike,
    fig_path: pathlib.Path,
    title: str,
    limit_axis=False,
) -> None:
    embeddings = calculate_normalized_embeddings(train_features, test_features)
    targets_group = grouping_helper_function(test_targets)
    fig, ax = plt.subplots()
    colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(targets_group))))
    for val, grp in targets_group:
        indices = grp.index
        grp_embeddings = embeddings[indices]
        cluster_center = grp_embeddings.mean(axis=0)
        color = next(colors)
        ax.scatter(*grp_embeddings.T, s=1, c=color)
        ax.text(*cluster_center, str(val), color=color)
    ax.set_xlabel('1st Scaled Principle Component')
    ax.set_ylabel('2nd Scaled Principle Component')
    ax.set_title(title)
    if limit_axis:
        ax.set_xlim((-1.5, 1.5))
        ax.set_ylim((-1.5, 1.5))
    ax.set_xticklabels(np.round(ax.get_xticks(), decimals=4), rotation=45)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=DPI)
