from src.multiple_links_sc_classification import execute_multiple_links_scenario
from src.result_exporter import print_announcement
from src.single_link_sc_regression import execute_single_link_scenario


def main():
    '''single link scenario'''
    print_announcement('single link scenario: lightpath distance prediction with constellation samples')
    execute_single_link_scenario()

    '''multiple link scenario'''
    print_announcement(
        'multiple links scenario: launch power prediction with constellation samples and sample location'
    )
    execute_multiple_links_scenario()

    print_announcement('All done!')


if __name__ == '__main__':
    main()
