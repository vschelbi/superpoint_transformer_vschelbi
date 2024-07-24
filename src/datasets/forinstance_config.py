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
        'CULS/plot_3_annotated',
        'NIBIO/plot_10_annotated',
        'NIBIO/plot_11_annotated',
        'NIBIO/plot_12_annotated',
        'NIBIO/plot_13_annotated',
        'NIBIO/plot_16_annotated',
        'NIBIO/plot_19_annotated',
        'NIBIO/plot_2_annotated',
        'NIBIO/plot_21_annotated',
        'NIBIO/plot_3_annotated',
        'NIBIO/plot_4_annotated',
        'NIBIO/plot_6_annotated',
        'NIBIO/plot_7_annotated',
        'RMIT/train',
        'SCION/plot_35_annotated',
        'SCION/plot_39_annotated',
        'TUWIEN/train'
    ],

    'val': [
        'CULS/plot_1_annotated',
        'NIBIO/plot_8_annotated',
        'NIBIO/plot_9_annotated',
        'SCION/plot_87_annotated'
    ],

    'test': [
        'CULS/plot_2_annotated',
        'NIBIO/plot_1_annotated',
        'NIBIO/plot_17_annotated',
        'NIBIO/plot_18_annotated',
        'NIBIO/plot_22_annotated',
        'NIBIO/plot_23_annotated',
        'NIBIO/plot_5_annotated',
        'RMIT/test',
        'SCION/plot_31_annotated',
        'SCION/plot_61_annotated',
        'TUWIEN/test'
    ]
}

TILES_WITH_NIBIO2 = {
    'train': [
        'CULS/plot_1_annotated',
        'CULS/plot_3_annotated',
        'NIBIO/plot_10_annotated',
        'NIBIO/plot_11_annotated',
        'NIBIO/plot_12_annotated',
        'NIBIO/plot_13_annotated',
        'NIBIO/plot_16_annotated',
        'NIBIO/plot_19_annotated',
        'NIBIO/plot_2_annotated',
        'NIBIO/plot_21_annotated',
        'NIBIO/plot_3_annotated',
        'NIBIO/plot_4_annotated',
        'NIBIO/plot_6_annotated',
        'NIBIO/plot_7_annotated',
        'NIBIO/plot_8_annotated',
        'NIBIO/plot_9_annotated',
        'NIBIO2/plot12_annotated',
        'NIBIO2/plot13_annotated',
        'NIBIO2/plot14_annotated',
        'NIBIO2/plot16_annotated',
        'NIBIO2/plot2_annotated',
        'NIBIO2/plot20_annotated',
        'NIBIO2/plot21_annotated',
        'NIBIO2/plot22_annotated',
        'NIBIO2/plot23_annotated',
        'NIBIO2/plot24_annotated',
        'NIBIO2/plot25_annotated',
        'NIBIO2/plot26_annotated',
        'NIBIO2/plot28_annotated',
        'NIBIO2/plot31_annotated',
        'NIBIO2/plot33_annotated',
        'NIBIO2/plot38_annotated',
        'NIBIO2/plot39_annotated',
        'NIBIO2/plot40_annotated',
        'NIBIO2/plot41_annotated',
        'NIBIO2/plot42_annotated',
        'NIBIO2/plot43_annotated',
        'NIBIO2/plot44_annotated',
        'NIBIO2/plot45_annotated',
        'NIBIO2/plot46_annotated',
        'NIBIO2/plot47_annotated',
        'NIBIO2/plot50_annotated',
        'NIBIO2/plot51_annotated',
        'NIBIO2/plot54_annotated',
        'NIBIO2/plot55_annotated',
        'NIBIO2/plot56_annotated',
        'NIBIO2/plot57_annotated',
        'NIBIO2/plot59_annotated',
        'NIBIO2/plot61_annotated',
        'NIBIO2/plot8_annotated',
        'NIBIO2/plot9_annotated',
        'RMIT/train',
        'SCION/plot_35_annotated',
        'SCION/plot_39_annotated',
        'SCION/plot_87_annotated',
        'TUWIEN/train'
    ],

    'val': [],

    'test': [
        'CULS/plot_2_annotated',
        'NIBIO/plot_1_annotated',
        'NIBIO/plot_17_annotated',
        'NIBIO/plot_18_annotated',
        'NIBIO/plot_22_annotated',
        'NIBIO/plot_23_annotated',
        'NIBIO/plot_5_annotated',
        'NIBIO2/plot1_annotated',
        'NIBIO2/plot10_annotated',
        'NIBIO2/plot15_annotated',
        'NIBIO2/plot27_annotated',
        'NIBIO2/plot3_annotated',
        'NIBIO2/plot32_annotated',
        'NIBIO2/plot34_annotated',
        'NIBIO2/plot35_annotated',
        'NIBIO2/plot48_annotated',
        'NIBIO2/plot49_annotated',
        'NIBIO2/plot52_annotated',
        'NIBIO2/plot53_annotated',
        'NIBIO2/plot58_annotated',
        'NIBIO2/plot6_annotated',
        'NIBIO2/plot60_annotated',
        'RMIT/test',
        'SCION/plot_31_annotated',
        'SCION/plot_61_annotated',
        'TUWIEN/test'
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