from __future__ import annotations

from functools import reduce, singledispatch
from itertools import chain
from typing import Callable, TypeVar

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split

MyArrayLike = TypeVar('MyArrayLike', pd.DataFrame, pd.Series, np.ndarray)


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

    def fit(self, data: MyArrayLike) -> CustomStandardScaler:
        self._mean, self._std = mean_std_calculator(data)
        return self

    def transform(self, data: MyArrayLike) -> pd.DataFrame:
        return pd.DataFrame((data - self._mean) / self._std)

    def inv_transform(self, data: MyArrayLike) -> pd.DataFrame:
        return pd.DataFrame(data * self._std + self._mean)


class StandardScalerDf(StandardScaler):
    '''StandardScaler with DataFrame output'''

    def transform(self, X):
        return pd.DataFrame(super().transform(X), columns=X.columns)

    def inverse_transform(self, X):
        return pd.DataFrame(super().inverse_transform(X), columns=X.columns)


def create_fit_transfrom_standard_scaler(
    data: MyArrayLike, column_wise: bool = False
) -> tuple[CustomStandardScaler, MyArrayLike]:
    '''creates a CustomScaler, fits it on data and returns the scaler, and scaled data'''
    if column_wise:
        scaler = StandardScalerDf().fit(data)
    else:
        scaler = CustomStandardScaler().fit(data)
    return scaler, scaler.transform(data)


def make_pipeline(steps: list[Callable]) -> Callable:
    '''makes a pipeline of functions'''
    return reduce(lambda x, y: lambda arg: y(x(arg)), steps)


def custom_train_test_split(*dfs: MyArrayLike, test_size: float, random_state: int) -> tuple[MyArrayLike, ...]:
    row_count = dfs[0].shape[0]
    train_indices, test_indices = train_test_split(range(row_count), test_size=test_size, random_state=random_state)
    return tuple(chain(*((df.iloc[train_indices], df.iloc[test_indices]) for df in dfs)))
