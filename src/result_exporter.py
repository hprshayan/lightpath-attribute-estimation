from pprint import pprint
from typing import Callable

import pandas as pd

from src.compressor import CompressorExplainer, compressor_performance_calculator
from src.constants import DISTANCE_FEATURE, TEST_DEPICTION_COUNT
from src.preprocessing import ArrayLike, remove_column_features


def print_announcement(message: str) -> None:
    padded_message = ' ' + message + ' '
    MAX_LEN = 102
    msg_len = len(padded_message)
    half_extra_char_count = (MAX_LEN - msg_len) // 2
    if msg_len > MAX_LEN:
        raise ValueError(f'Speak less but act more. Message must be less than {MAX_LEN - 2} characters.')
    print()
    print('#' * MAX_LEN)
    print('#' * half_extra_char_count, padded_message, '#' * (MAX_LEN - half_extra_char_count - msg_len), sep='')
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
    predictions = label_scaler.inv_transform(pipeline(feature_test[:TEST_DEPICTION_COUNT])).flatten()
    targets = label_test[DISTANCE_FEATURE][:TEST_DEPICTION_COUNT]
    indices = targets.index
    result_demo_list = [
        ' ' * 2 + 'index' + ' ' * 10 + 'prediction' + ' ' * 10 + 'target' + ' ' * 10 + '|target-prediction|'
    ]
    result_demo_list += [
        f'  {i:<14} {p:<19.1f} {t:<15.1f} {abs(t-p):.1f}' for i, p, t in zip(indices, predictions, targets)
    ]
    print(f'{scenario_title} regression with {method} approach is done.')
    print(f'test score (coefficeint of determination): {r2_score:.5f}')
    print(f'here are some predictions (with {unit} as unit):')
    print('\n'.join(result_demo_list))


def create_compression_report(
    method: str,
    original_dim: int,
    compressed_dim: int,
    features: ArrayLike,
    compressor_fwd_path: Callable,
    compressor_explainer: CompressorExplainer,
) -> None:
    compression_ratio, reconstruction_mae = compressor_performance_calculator(
        original_dim, compressed_dim, compressor_fwd_path(features), remove_column_features(features)
    )
    print(f'compression with {method} method is done.')
    print(f'compression ratio: {compression_ratio}', f'reconstruction MAE: {reconstruction_mae}', sep='\t')
    print('compression model parameters:')
    pprint(compressor_explainer.get_params())
