"""Wrapper around v1/2/3_to_areas"""

import logging

from clefts.manual_label.v1_to_areas import main as v1_to_areas
from clefts.manual_label.v2_to_areas import main as v2_to_areas
from clefts.manual_label.v3_to_areas import main as v3_to_areas
from clefts.manual_label.table_to_v3 import main as v1v2_skeletons


logger = logging.getLogger(__name__)


def main():
    v1_to_areas()
    v2_to_areas()
    v1v2_skeletons()

    v3_to_areas()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
