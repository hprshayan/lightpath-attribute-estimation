from dataclasses import dataclass


@dataclass(frozen=True)
class SingleLinkModeSpec:
    first_span_length: int  # in km
    attenuator_loss: float  # in dB


@dataclass(frozen=True)
class MultipleLinksModeSpec:
    span_count: int
    is_ingress: bool


# dataset constants
SPAN_LENGTH = 80
DATA_DIR = 'data'
ACCESSIBLE_DATA_DIR = 'accessible_dataset'
DISTANCE_FEATURE = 'distance'
POWER_FEATURE = 'power'
LOCATION_FEATURE = 'location'
ROADM_SIDE_FEATURE = 'roadm_side'
LINK_LENGTH_FEATURE = 'link_length'
ORIGINAL_DATA_ZIP_DIR = 'constellation-dataset.zip'

# single scenario constants
SINGLE_LINK_DATA_DIR = 'single_link_scenario'
SINGLE_LINK_DATA_ZIP_DIR = '16QAM-singleLink-s1_data.zip'
SINGLE_LINK_RE_PATTERN = 'consts_(\d+)span'
SINGLE_LINK_MODE_FEATURE = 'mode'
SINGLE_LINK_SPAN_COUNT_FEATURE = 'span_count'
SINGLE_LINK_MODE_SPEC = {
    'optimal': SingleLinkModeSpec(80, 0),
    'sub-optimal': SingleLinkModeSpec(60, -4),
    'degradation': SingleLinkModeSpec(40, -8),
}

# multiple links scenario constants
MULTIPLE_LINK_DATA_DIR = 'multiple_link_scenario'
MULTIPLE_LINK_DATA_ZIP_DIR = '16QAM-multipleLink-s1_data.zip'
MULTIPLE_LINK_RE_PATTERN = '(.+)_consts_(\d+)km_links_power-*(\d+)dBm'
MULTIPLE_LINKS_MODE_SPEC = {
    **{'init': MultipleLinksModeSpec(0, True)},
    **{f'in{i}': MultipleLinksModeSpec(i, True) for i in range(1, 5)},
    **{f'out{i}': MultipleLinksModeSpec(i - 1, False) for i in range(1, 6)},
}


# reproducibility
SEED = 0
TEST_SIZE_RATIO = 0.33

# PCA
N_COMPONENTS = 50
COMPRESSION_METHOD = 'PCA'
EMBEDDING_DEMO_DIM = 2

# reports
TEST_DEPICTION_COUNT = 10
EXP_DIR = 'exports'
DPI = 500
