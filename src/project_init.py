import logging
import os
import pathlib
import shutil
import sys
from zipfile import ZipFile

from src.constants import (
    ACCESSIBLE_DATA_DIR,
    DATA_DIR,
    EXP_DIR,
    MULTIPLE_LINK_DATA_DIR,
    MULTIPLE_LINK_DATA_ZIP_DIR,
    ORIGINAL_DATA_ZIP_DIR,
    SINGLE_LINK_DATA_DIR,
    SINGLE_LINK_DATA_ZIP_DIR,
)


def get_logger(log_path: pathlib.Path):
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(log_path.stem)
    logger.addHandler(logging.FileHandler(log_path.with_suffix('.log')))
    return logger


def organize_dirs() -> None:
    data_path = pathlib.Path(DATA_DIR)
    accessible_data_dir = data_path / ACCESSIBLE_DATA_DIR
    original_data_zip_dir = data_path / ORIGINAL_DATA_ZIP_DIR
    single_data_dir = accessible_data_dir / SINGLE_LINK_DATA_DIR
    multiple_data_dir = accessible_data_dir / MULTIPLE_LINK_DATA_DIR
    shutil.rmtree(EXP_DIR)
    os.mkdir(EXP_DIR)
    shutil.rmtree(accessible_data_dir)
    os.mkdir(accessible_data_dir)
    with ZipFile(original_data_zip_dir) as f:
        f.extractall(accessible_data_dir)
    with ZipFile(accessible_data_dir / SINGLE_LINK_DATA_ZIP_DIR) as f:
        f.extractall(single_data_dir)
    with ZipFile(accessible_data_dir / MULTIPLE_LINK_DATA_ZIP_DIR) as f:
        f.extractall(multiple_data_dir)
