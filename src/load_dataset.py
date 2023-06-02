import pathlib
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from src.const import ACCESSIBLE_DATA_DIR, SINGLE_LINK_DATA_DIR
from const import (
    ACCESSIBLE_DATA_DIR,
    MULTIPLE_LINK_DATA_DIR,
    MULTIPLE_LINK_LABELS_LIST,
    MULTIPLE_LINK_RE_PATTERN,
    SINGLE_LINK_DATA_DIR,
    SINGLE_LINK_DATA_OPTIMAL_DIR,
    SINGLE_LINK_LABELS_LIST,
    SINGLE_LINK_RE_PATTERN,
)


def label_extractor(file_name: str, compiler: re.compile) -> list[str]:
    """helper function that extracts the labels from file names"""
    labels = compiler.findall(file_name)[0]
    if isinstance(labels, str):
        return (labels,)
    return labels


def load_dataset(path: pathlib, label_pattern: str, labels: list[str]) -> pd.DataFrame:
    """loads the dataset and concatenates targetst to the dataset"""
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
    """decomposes the string values to float in phase and quadrature columns"""
    decomposed_dataset = pd.DataFrame()
    decompose_columns = list(set(df.columns) - set(ignore_columns))
    complex_data = df[decompose_columns].applymap(lambda x: complex(x.replace('i', 'j')))
    i_df = complex_data.applymap(lambda x: x.real)
    q_df = complex_data.applymap(lambda x: x.imag)
    decomposed_dataset = pd.concat([i_df, q_df], axis=1, ignore_index=True)
    decomposed_dataset[ignore_columns] = df[ignore_columns]
    decomposed_dataset.columns = (
        [f'{c}_i' for c in decompose_columns] + [f'{c}_q' for c in decompose_columns] + ignore_columns
    )
    return decomposed_dataset


# accessible_data_dir = pathlib.Path(ACCESSIBLE_DATA_DIR)
# single_link_data_dir = accessible_data_dir / SINGLE_LINK_DATA_DIR
# single_link_data_optimal_dir = single_link_data_dir / SINGLE_LINK_DATA_OPTIMAL_DIR
# multiple_link_data_dir = accessible_data_dir / MULTIPLE_LINK_DATA_DIR

# single_df = load_dataset(single_link_data_optimal_dir, SINGLE_LINK_RE_PATTERN, SINGLE_LINK_LABELS_LIST)
# multiple_df = load_dataset(multiple_link_data_dir, MULTIPLE_LINK_RE_PATTERN, MULTIPLE_LINK_LABELS_LIST)
# decomposed_dataset = decompose_complex_data(multiple_df, MULTIPLE_LINK_LABELS_LIST)
# decomposed_dataset = decompose_complex_data(single_df, SINGLE_LINK_LABELS_LIST)
# l = 4
