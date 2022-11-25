import numpy as np


########################################################################
#                         Download information                         #
########################################################################


########################################################################
#                              Data splits                             #
########################################################################

# Credit: https://github.com/torch-points3d/torch-points3d

ROOM_TYPES = {
    "conferenceRoom": 0,
    "copyRoom": 1,
    "hallway": 2,
    "office": 3,
    "pantry": 4,
    "WC": 5,
    "auditorium": 6,
    "storage": 7,
    "lounge": 8,
    "lobby": 9,
    "openspace": 10}

VALIDATION_ROOMS = [
    "hallway_1",
    "hallway_6",
    "hallway_11",
    "office_1",
    "office_6",
    "office_11",
    "office_16",
    "office_21",
    "office_26",
    "office_31",
    "office_36",
    "WC_2",
    "storage_1",
    "storage_5",
    "conferenceRoom_2",
    "auditorium_1"]


########################################################################
#                                Labels                                #
########################################################################

# Credit: https://github.com/torch-points3d/torch-points3d

S3DIS_NUM_CLASSES = 13

INV_OBJECT_LABEL = {
    0: "ceiling",
    1: "floor",
    2: "wall",
    3: "beam",
    4: "column",
    5: "window",
    6: "door",
    7: "chair",
    8: "table",
    9: "bookcase",
    10: "sofa",
    11: "board",
    12: "clutter"}

CLASS_NAMES = [INV_OBJECT_LABEL[i] for i in range(S3DIS_NUM_CLASSES)] + ['ignored']

OBJECT_COLOR = np.asarray([
    [233, 229, 107],  # 'ceiling'   ->  yellow
    [95, 156, 196],   # 'floor'     ->  blue
    [179, 116, 81],   # 'wall'      ->  brown
    [241, 149, 131],  # 'beam'      ->  salmon
    [81, 163, 148],   # 'column'    ->  bluegreen
    [77, 174, 84],    # 'window'    ->  bright green
    [108, 135, 75],   # 'door'      ->  dark green
    [41, 49, 101],    # 'chair'     ->  darkblue
    [79, 79, 76],     # 'table'     ->  dark grey
    [223, 52, 52],    # 'bookcase'  ->  red
    [89, 47, 95],     # 'sofa'      ->  purple
    [81, 109, 114],   # 'board'     ->  grey
    [233, 233, 229],  # 'clutter'   ->  light grey
    [0, 0, 0]])       # unlabelled  -> black

OBJECT_LABEL = {name: i for i, name in INV_OBJECT_LABEL.items()}

def object_name_to_label(object_class):
    """Convert from object name to int label.
    """
    object_label = OBJECT_LABEL.get(object_class, OBJECT_LABEL["clutter"])  #TODO: default to "clutter"=12 or to "ignored"=-1 ?
    return object_label
