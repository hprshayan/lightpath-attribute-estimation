import time
import warnings

from matplotlib.axes._axes import _log as matplotlib_axes_logger

from src.project_init import organize_dirs
from src.result_exporter import print_announcement

# avoid annoying warnings
matplotlib_axes_logger.setLevel('ERROR')
warnings.filterwarnings('ignore')


def main():
    # initialize project structure
    print('initializing the demo...')
    organize_dirs()
    print('demo project is initialized')
    # import scenario executors
    from src.multiple_links_sc_classification import execute_multiple_links_scenario
    from src.single_link_sc_regression import execute_single_link_scenario

    start_time = time.perf_counter()
    '''single link scenario'''
    print_announcement('single link scenario: lightpath distance prediction with constellation samples')
    execute_single_link_scenario()

    '''multiple link scenario'''
    print_announcement(
        'multiple links scenario: launch power prediction with constellation samples and sample location'
    )
    execute_multiple_links_scenario()

    print_announcement(f'All done in {int(time.perf_counter() - start_time)} seconds!')


if __name__ == '__main__':
    main()
