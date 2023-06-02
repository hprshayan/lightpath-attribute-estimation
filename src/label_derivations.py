import pandas as pd

from src.const import SINGLE_LINK_MODE_SPEC, SPAN_LENGTH


def distance_calculator(span_count: int, offset: int) -> int:
    return offset + (span_count - 1) * SPAN_LENGTH


def calculate_single_link_lp_length(labels: pd.DataFrame) -> pd.DataFrame:
    labels['span_count'] = labels['span_count'].astype(int)
    return labels.apply(
        lambda row: distance_calculator(row['span_count'], SINGLE_LINK_MODE_SPEC[row['mode']].first_span_length),
        axis=1,
    )
