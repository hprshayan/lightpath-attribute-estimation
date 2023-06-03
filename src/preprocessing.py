from __future__ import annotations

from functools import reduce, singledispatch
from typing import Callable, TypeVar

import numpy as np
import pandas as pd

from src.constants import DISTANCE_FEATURE, ROADM_SIDE_FEATURE

ArrayLike = TypeVar('ArrayLike', pd.DataFrame, pd.Series, np.ndarray)
NdArrayNone = np.ndarray | None


def mean_std_calculator(data: pd.DataFrame | pd.Series) -> tuple[float, float]:
    return data.to_numpy().flatten().mean(), data.to_numpy().flatten().std()


@singledispatch
def fit_helper(data: pd.DataFrame, column_features: list[str] | None) -> tuple[float, float, NdArrayNone, NdArrayNone]:
    mean_matrix, std_matrix = mean_std_calculator(data.loc[:, ~data.columns.isin(column_features)])
    mean_column = data[column_features].mean().to_numpy()
    std_column = data[column_features].std().to_numpy()
    return mean_matrix, std_matrix, mean_column, std_column


@fit_helper.register
def _(data: pd.Series, column_features: list[str] | None) -> tuple[float, float, NdArrayNone, NdArrayNone]:
    mean_matrix, std_matrix = mean_std_calculator(data)
    return mean_matrix, std_matrix, None, None


@singledispatch
def transform_helper(
    data: pd.DataFrame,
    _mean_matrix: float,
    _std_matrix: float,
    _mean_column: NdArrayNone,
    _std_column: NdArrayNone,
    _column_features: list[str] | None,
) -> ArrayLike:
    data.loc[:, ~data.columns.isin(_column_features)] = (
        data.loc[:, ~data.columns.isin(_column_features)] - _mean_matrix
    ) / _std_matrix
    data[_column_features] = (data[_column_features] - _mean_column) / _std_column
    return data


@transform_helper.register
def _(
    data: pd.Series,
    _mean_matrix: float,
    _std_matrix: float,
    _mean_column: NdArrayNone,
    _std_column: NdArrayNone,
    _column_features: list[str] | None,
) -> ArrayLike:
    return (data - _mean_matrix) / _std_matrix


@singledispatch
def inv_transform_helper(
    data: pd.DataFrame,
    _mean_matrix: float,
    _std_matrix: float,
    _mean_column: NdArrayNone,
    _std_column: NdArrayNone,
    _column_features: list[str] | None,
) -> ArrayLike:
    data.loc[:, ~data.columns.isin(_column_features)] = (
        data.loc[:, ~data.columns.isin(_column_features)] * _std_matrix + _mean_matrix
    )
    data[_column_features] = data[_column_features] * _std_column + _mean_column
    return data


@inv_transform_helper.register
def _(
    data: pd.Series,
    _mean_matrix: float,
    _std_matrix: float,
    _mean_column: NdArrayNone,
    _std_column: NdArrayNone,
    _column_features: list[str] | None,
) -> ArrayLike:
    return data * _std_matrix + _mean_matrix


class CustomStandardScaler:
    '''A custom standard scaler model'''

    def __init__(self, column_features: list[str] | None = None):
        self._mean_matrix: float = 0
        self._std_matrix: float = 0
        self._mean_column: NdArrayNone = None
        self._std_column: NdArrayNone = None
        if column_features is None:
            self._column_features = []
        else:
            self._column_features = column_features

    def fit(self, data: ArrayLike) -> CustomStandardScaler:
        self._mean_matrix, self._std_matrix, self._mean_column, self._std_column = fit_helper(
            data, self._column_features
        )
        return self

    def transform(self, data: ArrayLike) -> ArrayLike:
        return transform_helper(data, **self.__dict__)

    def inv_transform(self, data: ArrayLike) -> ArrayLike:
        return inv_transform_helper(data, **self.__dict__)


def create_fit_transfrom_standard_scaler(
    data: ArrayLike, column_features: list[str] | None = None
) -> tuple[CustomStandardScaler, ArrayLike]:
    '''creates a CustomScaler, fits it on data and returns the scaler, and scaled data'''
    scaler = CustomStandardScaler(column_features).fit(data)
    return scaler, scaler.transform(data)


def make_pipeline(steps: list[Callable]) -> Callable:
    '''makes a pipeline of functions'''
    return reduce(lambda x, y: lambda arg: y(x(arg)), steps)


def remove_column_features(data: pd.DataFrame) -> pd.DataFrame:
    '''removes non-constellation columns'''
    return data.loc[:, ~data.columns.isin([DISTANCE_FEATURE, ROADM_SIDE_FEATURE])]
