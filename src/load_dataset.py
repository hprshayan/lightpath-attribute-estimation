import pathlib
import re
from enum import Enum, auto

import pandas as pd

from src.constants import (
    ACCESSIBLE_DATA_DIR,
    LINK_LENGTH_FEATURE,
    LOCATION_FEATURE,
    MULTIPLE_LINK_DATA_DIR,
    MULTIPLE_LINK_RE_PATTERN,
    POWER_FEATURE,
    SINGLE_LINK_DATA_DIR,
    SINGLE_LINK_MODE_FEATURE,
    SINGLE_LINK_RE_PATTERN,
    SINGLE_LINK_SPAN_COUNT_FEATURE,
)


class Scenario(Enum):
    SINGLE_LINK = auto()
    MULTIPLE_LINK = auto()


def concat_helper(*dfs: pd.DataFrame, ignore_index: bool = False) -> pd.DataFrame:
    '''horizontally concatenates sequence of pd.DataFrames'''
    return pd.concat(dfs, axis=1, ignore_index=ignore_index)


def label_extractor(file_name: str, compiler: re.compile) -> list[str]:
    '''helper function that extracts the labels from file names'''
    labels = compiler.findall(file_name)[0]
    if isinstance(labels, str):
        return (labels,)
    return labels


def load_csv_dataset(path: pathlib, label_pattern: str, labels: list[str]) -> pd.DataFrame:
    '''loads the dataset and concatenates targetst to the dataset'''
    dataset = pd.DataFrame()
    re_compiler = re.compile(label_pattern)
    for file_path in path.iterdir():
        data = pd.read_csv(file_path, header=None).transpose()
        extracted_labels = label_extractor(file_path.stem, re_compiler)
        for extracted_lable, label in zip(extracted_labels, labels):
            data[label] = extracted_lable
        dataset = pd.concat([dataset, data], axis=0, ignore_index=True)
    return dataset


def decompose_complex_data(df: pd.DataFrame, ignore_columns: list[str]) -> pd.DataFrame:
    '''decomposes the string values to float in phase and quadrature columns'''
    decomposed_dataset = pd.DataFrame()
    decompose_columns = list(set(df.columns) - set(ignore_columns))
    complex_data = df[decompose_columns].applymap(lambda x: complex(x.replace('i', 'j')))
    i_df = complex_data.applymap(lambda x: x.real)
    q_df = complex_data.applymap(lambda x: x.imag)
    decomposed_dataset = concat_helper(i_df, q_df, ignore_index=True)
    decomposed_dataset[ignore_columns] = df[ignore_columns]
    decomposed_dataset.columns = (
        [f'{c}_i' for c in decompose_columns] + [f'{c}_q' for c in decompose_columns] + ignore_columns
    )
    return decomposed_dataset


def load_csv_decompose(path: pathlib.Path, label_pattern: str, labels: list[str]) -> pd.DataFrame:
    '''loads the csv file and decomposes the complex data'''
    raw_dataset = load_csv_dataset(path, label_pattern, labels)
    return decompose_complex_data(raw_dataset, labels)


def load_dataset(scenario: Scenario, seed: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''loads the intended dataset with the extracted labels from file paths'''
    accessible_data_dir = pathlib.Path(ACCESSIBLE_DATA_DIR)
    if scenario == Scenario.SINGLE_LINK:
        single_link_data_dir = accessible_data_dir / SINGLE_LINK_DATA_DIR
        labels = [SINGLE_LINK_SPAN_COUNT_FEATURE, SINGLE_LINK_MODE_FEATURE]
        dataset = pd.DataFrame()
        for scenario_mode_path in single_link_data_dir.iterdir():
            sub_dataset = load_csv_decompose(
                scenario_mode_path, SINGLE_LINK_RE_PATTERN, [SINGLE_LINK_SPAN_COUNT_FEATURE]
            )
            sub_dataset[SINGLE_LINK_MODE_FEATURE] = scenario_mode_path.stem
            dataset = pd.concat([dataset, sub_dataset], axis=0, ignore_index=True)
    elif scenario == Scenario.MULTIPLE_LINK:
        multiple_link_data_dir = accessible_data_dir / MULTIPLE_LINK_DATA_DIR
        labels = [LOCATION_FEATURE, LINK_LENGTH_FEATURE, POWER_FEATURE]
        dataset = load_csv_decompose(multiple_link_data_dir, MULTIPLE_LINK_RE_PATTERN, labels)
    if seed is not None:
        dataset = dataset.sample(frac=1, random_state=seed)
    return dataset.drop(labels, axis=1), dataset[labels]
