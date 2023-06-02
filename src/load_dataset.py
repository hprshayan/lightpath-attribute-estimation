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


# accessible_data_dir = pathlib.Path(ACCESSIBLE_DATA_DIR)
# single_link_data_dir = accessible_data_dir / SINGLE_LINK_DATA_DIR
# single_link_data_optimal_dir = single_link_data_dir / SINGLE_LINK_DATA_OPTIMAL_DIR
# multiple_link_data_dir = accessible_data_dir / MULTIPLE_LINK_DATA_DIR
#
# single = label_extractor('consts_13span', SINGLE_LINK_RE_PATTERN)
# multiple = label_extractor('in2_consts_560km_links_power-1dBm', MULTIPLE_LINK_RE_PATTERN)
# single_df = load_dataset(single_link_data_optimal_dir, SINGLE_LINK_RE_PATTERN, SINGLE_LINK_LABELS_LIST)
# multiple_df = load_dataset(multiple_link_data_dir, MULTIPLE_LINK_RE_PATTERN, MULTIPLE_LINK_LABELS_LIST)
# l = 4
