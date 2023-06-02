SPAN_LENGTH = 80
ACCESSIBLE_DATA_DIR = 'data/accessible_dataset'
SINGLE_LINK_DATA_DIR = 'single_link_scenario'
SINGLE_LINK_DATA_OPTIMAL_DIR = 'optimal'
MULTIPLE_LINK_DATA_DIR = 'multiple_link_scenario'

SINGLE_LINK_RE_PATTERN = 'consts_(\d+)span'
SINGLE_LINK_LABELS_LIST = ['span_count']
MULTIPLE_LINK_RE_PATTERN = '(.+)_consts_(\d+)km_links_power-*(\d+)dBm'
MULTIPLE_LINK_LABELS_LIST = ['location', 'distance', 'power']
