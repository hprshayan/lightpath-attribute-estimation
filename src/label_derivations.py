import pandas as pd

from src.constants import (
    LINK_LENGTH_FEATURE,
    LOCATION_FEATURE,
    MULTIPLE_LINKS_MODE_SPEC,
    POWER_FEATURE,
    SINGLE_LINK_MODE_FEATURE,
    SINGLE_LINK_MODE_SPEC,
    SINGLE_LINK_SPAN_COUNT_FEATURE,
    SPAN_LENGTH,
)
from src.load_dataset import concat_helper


def distance_calculator(span_count: int, offset: int = 0) -> int:
    return offset + (span_count - 1) * SPAN_LENGTH


def calculate_single_link_lp_length(labels: pd.DataFrame) -> pd.Series:
    return labels.apply(
        lambda row: distance_calculator(
            int(row[SINGLE_LINK_SPAN_COUNT_FEATURE]),
            SINGLE_LINK_MODE_SPEC[row[SINGLE_LINK_MODE_FEATURE]].first_span_length,
        ),
        axis=1,
    )


def calculate_multiple_link_label_derivations(labels: pd.DataFrame) -> pd.DataFrame:
    distance = labels.apply(
        lambda row: int(row[LINK_LENGTH_FEATURE]) * MULTIPLE_LINKS_MODE_SPEC[row[LOCATION_FEATURE]].span_count, axis=1
    )
    roadm_side = labels.apply(lambda row: int(MULTIPLE_LINKS_MODE_SPEC[row[LOCATION_FEATURE]].is_ingress), axis=1)
    power = labels.apply(lambda row: int(row[POWER_FEATURE]) - 1, axis=1)
    return concat_helper(distance, roadm_side, power)
