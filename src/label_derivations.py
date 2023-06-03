import pandas as pd

from src.const import (
    SINGLE_LINK_MODE_FEATURE,
    SINGLE_LINK_MODE_SPEC,
    SINGLE_LINK_SPAN_COUNT_FEATURE,
    SPAN_LENGTH,
)


def distance_calculator(span_count: int, offset: int) -> int:
    return offset + (span_count - 1) * SPAN_LENGTH


def calculate_single_link_lp_length(labels: pd.DataFrame) -> pd.DataFrame:
    labels[SINGLE_LINK_SPAN_COUNT_FEATURE] = labels[SINGLE_LINK_SPAN_COUNT_FEATURE].astype(int)
    return labels.apply(
        lambda row: distance_calculator(
            row[SINGLE_LINK_SPAN_COUNT_FEATURE], SINGLE_LINK_MODE_SPEC[row[SINGLE_LINK_MODE_FEATURE]].first_span_length
        ),
        axis=1,
    )
