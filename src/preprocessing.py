from __future__ import annotations

from functools import reduce, singledispatch
from typing import Callable, TypeVar

import numpy as np
import pandas as pd

ArrayLike = TypeVar('ArrayLike', pd.DataFrame, np.ndarray)


@singledispatch
def mean_std_calculator(data: pd.DataFrame) -> tuple[float, float]:
    return data.to_numpy().flatten().mean(), data.to_numpy().flatten().std()


@mean_std_calculator.register
def _(data: np.ndarray) -> tuple[float, float]:
    return data.mean(), data.std()


class CustomStandardScaler:
    '''normalizes the whole data frame'''

    def __ini__(self):
        self._mean: float = 0
        self._std: float = 0

    def fit(self, data: ArrayLike) -> CustomStandardScaler:
        self._mean, self._std = mean_std_calculator(data)
        return self

    def transform(self, data: ArrayLike) -> ArrayLike:
        return (data - self._mean) / self._std

    def inv_transform(self, data: ArrayLike) -> ArrayLike:
        return data * self._std + self._mean


def create_fit_transfrom_standard_scaler(data: ArrayLike) -> tuple[CustomStandardScaler, ArrayLike]:
    '''creates a CustomScaler, fits it on data and returns the scaler, and scaled data'''
    scaler = CustomStandardScaler().fit(data)
    return scaler, scaler.transform(data)


def make_pipeline(steps: list[Callable]) -> Callable:
    '''makes a pipeline of functions'''
    return reduce(lambda x, y: lambda arg: y(x(arg)), steps)
