import pathlib
import warnings
from pprint import pprint
from typing import Callable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes._axes import _log as matplotlib_axes_logger
from sklearn.decomposition import PCA

from src.compressor import CompressorExplainer, compressor_performance_calculator
from src.constants import DPI, EMBEDDING_DEMO_DIM, EXP_DIR, TEST_DEPICTION_COUNT
from src.preprocessing import MyArrayLike

matplotlib_axes_logger.setLevel('ERROR')
warnings.filterwarnings('ignore')


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
    label_test: pd.DataFrame,
    pipeline: Callable,
    label_scaler: Callable,
) -> None:
    predictions = label_scaler.inv_transform(pipeline(feature_test[:TEST_DEPICTION_COUNT])).to_numpy().flatten()
    targets = label_test[:TEST_DEPICTION_COUNT]
    indices = targets.index
    result_demo_list = [
        ' ' * 2 + 'index' + ' ' * 10 + 'prediction' + ' ' * 10 + 'target' + ' ' * 10 + '|target-prediction|'
    ]
    result_demo_list += [
        f"  {i:<14} {p:<19.1f} {t:<15.1f} {abs(t-p):.1f}" for i, p, t in zip(indices, predictions, targets)
    ]
    print(f'{scenario_title} regression with {method} approach is done.')
    print(f'test score (coefficeint of determination): {r2_score:.5f}')
    print(f'here are some predictions (with {unit} as unit):')
    print('\n'.join(result_demo_list))


def create_compression_report(
    method: str,
    original_dim: int,
    compressed_dim: int,
    features: MyArrayLike,
    compressor_fwd_path: Callable,
    compressor_explainer: CompressorExplainer,
) -> None:
    compression_ratio, reconstruction_mae = compressor_performance_calculator(
        original_dim, compressed_dim, compressor_fwd_path(features), features
    )
    print(f'compression with {method} method is done.')
    print(
        f'compression ratio: {compression_ratio}',
        f'reconstruction MAE: {reconstruction_mae}',
        sep="\t",
    )
    print('compression model parameters:')
    pprint(compressor_explainer.get_params())


def create_classification_report(
    scenario_title: str,
    method: str,
    unit: str,
    confusion_mat: MyArrayLike,
    predictions: MyArrayLike,
    targets: MyArrayLike,
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
    print(f'{scenario_title} classification with {method} approach is done.')
    print(f'confusion matrix:')
    print(c_mat_df)
    print(f'accuracy: {accuracy}')
    print(f'here are some predictions (with {unit} as unit):')
    print('\n'.join(result_demo_list))


def grouping_helper_function(df: pd.Series):
    reset_index_df = df.reset_index(drop=True)
    return list(reset_index_df.groupby(reset_index_df))


def calculate_normalized_embeddings(train_data: pd.DataFrame, test_data: pd.DataFrame) -> np.ndarray:
    pca = PCA(n_components=EMBEDDING_DEMO_DIM)
    pca.fit(train_data)
    embeddings = pca.transform(test_data)
    return (embeddings - embeddings.mean()) / embeddings.std()


def low_dim_embedding_plot(
    train_features: MyArrayLike,
    test_features: MyArrayLike,
    test_targets: MyArrayLike,
    fig_path: pathlib.Path,
    title: str,
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
    ax.set_xticklabels(np.round(ax.get_xticks(), decimals=4), rotation=45)
    ax.set_xlabel('1st Scaled Principle Component')
    ax.set_ylabel('2nd Scaled Principle Component')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=DPI)


def single_scenario_low_dim_plot(
    train_features: MyArrayLike, test_features: MyArrayLike, test_targets: MyArrayLike, modes: MyArrayLike
) -> None:
    export_path = pathlib.Path(EXP_DIR)
    embeddings = calculate_normalized_embeddings(train_features, test_features)
    modes_group = grouping_helper_function(modes)
    fig, ax = plt.subplots()
    colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(modes_group))))
    for val, grp in modes_group:
        indices = grp.index
        low_dim_embedding_plot(
            train_features.iloc[indices],
            test_features.iloc[indices],
            test_targets.iloc[indices],
            export_path / f'single_scenario({val}).png',
            f'Single Link Scenario Embeddings ({val} Mode)',
        )
        grp_embeddings = embeddings[indices]
        cluster_center = grp_embeddings.mean(axis=0)
        color = next(colors)
        ax.scatter(*grp_embeddings.T, s=1, c=color)
        ax.text(*cluster_center, str(val), color=color)
    ax.set_xlabel('1st Scaled Principle Component')
    ax.set_ylabel('2nd Scaled Principle Component')
    ax.set_title('Single Link Scenario Embeddings')
    fig.savefig(export_path / 'single_scenario.png', dpi=DPI)
