import numpy as np


########################################################################
#                         Download information                         #
########################################################################

DOWNLOAD_URL = 'https://zenodo.org/records/8287792'

# FOR-instance in LAS format
LAS_ZIP_NAME = 'FORinstance_dataset.zip'
LAS_UNZIP_NAME = "forinstance"


########################################################################
#                              Data splits                             #
########################################################################

# The validation set was arbitrarily chosen as the x last train tiles of each
# dataset. When there is only one tile, it is used for training.
TILES = {
    'train': [
        'CULS/plot_3_annotated.las',
        'NIBIO/plot_10_annotated.las',
        'NIBIO/plot_11_annotated.las',
        'NIBIO/plot_12_annotated.las',
        'NIBIO/plot_13_annotated.las',
        'NIBIO/plot_16_annotated.las',
        'NIBIO/plot_19_annotated.las',
        'NIBIO/plot_2_annotated.las',
        'NIBIO/plot_21_annotated.las',
        'NIBIO/plot_3_annotated.las',
        'NIBIO/plot_4_annotated.las',
        'NIBIO/plot_6_annotated.las',
        'NIBIO/plot_7_annotated.las',
        'RMIT/train.las',
        'SCION/plot_35_annotated.las',
        'SCION/plot_39_annotated.las',
        'TUWIEN/train.las'
    ],

    'val': [
        'CULS/plot_1_annotated.las',
        'NIBIO/plot_8_annotated.las',
        'NIBIO/plot_9_annotated.las',
        'SCION/plot_87_annotated.las'
    ],

    'test': [
        'CULS/plot_2_annotated.las',
        'NIBIO/plot_1_annotated.las',
        'NIBIO/plot_17_annotated.las',
        'NIBIO/plot_18_annotated.las',
        'NIBIO/plot_22_annotated.las',
        'NIBIO/plot_23_annotated.las',
        'NIBIO/plot_5_annotated.las',
        'RMIT/test.las',
        'SCION/plot_31_annotated.las',
        'SCION/plot_61_annotated.las',
        'TUWIEN/test.las'
    ]
}

TILES_WITH_NIBIO2 = {
    'train': [
        'CULS/plot_1_annotated.las',
        'CULS/plot_3_annotated.las',
        'NIBIO/plot_10_annotated.las',
        'NIBIO/plot_11_annotated.las',
        'NIBIO/plot_12_annotated.las',
        'NIBIO/plot_13_annotated.las',
        'NIBIO/plot_16_annotated.las',
        'NIBIO/plot_19_annotated.las',
        'NIBIO/plot_2_annotated.las',
        'NIBIO/plot_21_annotated.las',
        'NIBIO/plot_3_annotated.las',
        'NIBIO/plot_4_annotated.las',
        'NIBIO/plot_6_annotated.las',
        'NIBIO/plot_7_annotated.las',
        'NIBIO/plot_8_annotated.las',
        'NIBIO/plot_9_annotated.las',
        'NIBIO2/plot12_annotated.las',
        'NIBIO2/plot13_annotated.las',
        'NIBIO2/plot14_annotated.las',
        'NIBIO2/plot16_annotated.las',
        'NIBIO2/plot2_annotated.las',
        'NIBIO2/plot20_annotated.las',
        'NIBIO2/plot21_annotated.las',
        'NIBIO2/plot22_annotated.las',
        'NIBIO2/plot23_annotated.las',
        'NIBIO2/plot24_annotated.las',
        'NIBIO2/plot25_annotated.las',
        'NIBIO2/plot26_annotated.las',
        'NIBIO2/plot28_annotated.las',
        'NIBIO2/plot31_annotated.las',
        'NIBIO2/plot33_annotated.las',
        'NIBIO2/plot38_annotated.las',
        'NIBIO2/plot39_annotated.las',
        'NIBIO2/plot40_annotated.las',
        'NIBIO2/plot41_annotated.las',
        'NIBIO2/plot42_annotated.las',
        'NIBIO2/plot43_annotated.las',
        'NIBIO2/plot44_annotated.las',
        'NIBIO2/plot45_annotated.las',
        'NIBIO2/plot46_annotated.las',
        'NIBIO2/plot47_annotated.las',
        'NIBIO2/plot50_annotated.las',
        'NIBIO2/plot51_annotated.las',
        'NIBIO2/plot54_annotated.las',
        'NIBIO2/plot55_annotated.las',
        'NIBIO2/plot56_annotated.las',
        'NIBIO2/plot57_annotated.las',
        'NIBIO2/plot59_annotated.las',
        'NIBIO2/plot61_annotated.las',
        'NIBIO2/plot8_annotated.las',
        'NIBIO2/plot9_annotated.las',
        'RMIT/train.las',
        'SCION/plot_35_annotated.las',
        'SCION/plot_39_annotated.las',
        'SCION/plot_87_annotated.las',
        'TUWIEN/train.las'
    ],

    'val': [],

    'test': [
        'CULS/plot_2_annotated.las',
        'NIBIO/plot_1_annotated.las',
        'NIBIO/plot_17_annotated.las',
        'NIBIO/plot_18_annotated.las',
        'NIBIO/plot_22_annotated.las',
        'NIBIO/plot_23_annotated.las',
        'NIBIO/plot_5_annotated.las',
        'NIBIO2/plot1_annotated.las',
        'NIBIO2/plot10_annotated.las',
        'NIBIO2/plot15_annotated.las',
        'NIBIO2/plot27_annotated.las',
        'NIBIO2/plot3_annotated.las',
        'NIBIO2/plot32_annotated.las',
        'NIBIO2/plot34_annotated.las',
        'NIBIO2/plot35_annotated.las',
        'NIBIO2/plot48_annotated.las',
        'NIBIO2/plot49_annotated.las',
        'NIBIO2/plot52_annotated.las',
        'NIBIO2/plot53_annotated.las',
        'NIBIO2/plot58_annotated.las',
        'NIBIO2/plot6_annotated.las',
        'NIBIO2/plot60_annotated.las',
        'RMIT/test.las',
        'SCION/plot_31_annotated.las',
        'SCION/plot_61_annotated.las',
        'TUWIEN/test.las'
    ]
}

########################################################################
#                                Labels                                #
########################################################################
FORInstance_NUM_CLASSES = 3

ID2TRAINID = np.asarray([3, 1, 0, 3, 2, 2, 2])

CLASS_NAMES = [
    'Terrain',          # 2 Terrain
    'Low vegetation',   # 1 Low vegetation
    'Tree',             # 4 Stem, 5 Live branches, 6 Woody branches
    'Unknown'           # 0 Unclassified, 3 Out-points
]

CLASS_COLORS = np.asarray([
    [243, 214, 171],    # Terrain
    [141, 222, 29],     # Low vegetation
    [ 70, 115,  66],    # Tree
    [  0,   8, 116]     # Unknown
])

# For instance segmentation
MIN_OBJECT_SIZE = 100
THING_CLASSES = [2]
STUFF_CLASSES = [0, 1]