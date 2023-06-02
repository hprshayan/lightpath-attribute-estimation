from dataclasses import dataclass


@dataclass(frozen=True)
class SingleLinkModeSpec:
    first_span_length: int  # in km
    attenuator_loss: float  # in dB


# dataset constants
SPAN_LENGTH = 80
ACCESSIBLE_DATA_DIR = 'data/accessible_dataset'
SINGLE_LINK_DATA_DIR = 'single_link_scenario'
MULTIPLE_LINK_DATA_DIR = 'multiple_link_scenario'

SINGLE_LINK_RE_PATTERN = 'consts_(\d+)span'
SINGLE_LINK_LABELS_LIST = ['span_count']
MULTIPLE_LINK_RE_PATTERN = '(.+)_consts_(\d+)km_links_power-*(\d+)dBm'
MULTIPLE_LINK_LABELS_LIST = ['location', 'distance', 'power']

SINGLE_LINK_MODE_SPEC = {
    'optimal': SingleLinkModeSpec(80, 0),
    'sub-optimal': SingleLinkModeSpec(60, -4),
    'degradation': SingleLinkModeSpec(40, -8),
}

# reproducibility
SEED = 0

# PCA
N_COMPONENTS = 50

# linear regression
