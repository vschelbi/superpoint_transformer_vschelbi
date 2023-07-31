import os
import torch
import logging
import os.path as osp
from src.data import Data, InstanceData
from src.utils.scannet import read_one_scan, read_one_test_scan
from src.datasets.scannet_config import *
from torch_geometric.nn.pool.consecutive import consecutive_cluster


DIR = os.path.dirname(os.path.realpath(__file__))
log = logging.getLogger(__name__)


# Occasional Dataloader issues on some machines. Hack to solve this:
# https://stackoverflow.com/questions/73125231/pytorch-dataloaders-bad-file-descriptor-and-eof-for-workers0
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


__all__ = ['ScanNet', 'MiniScanNet']


########################################################################
#                                 Utils                                #
########################################################################

def read_scannet_scan(
        scan_dir,
        xyz=True,
        rgb=True,
        semantic=True,
        instance=True,
        remap=False):
    """Read a ScanNet scan.

    Expects the data to be saved under the following structure:

        raw/
            └── scannetv2-labels.combined.tsv
            └── scans/
            |   └── {{scan_name}}/
            |       |── {{scan_name}}_vh_clean_2.ply
            |       |── {{scan_name}}.aggregation.json
            |       |── {{scan_name}}.txt
            |       └── {{scan_name}}_vh_clean_2.0.010000.segs.json
            └── scans_test/
                └── {{scan_name}}/
                    └── {{scan_name}}_vh_clean_2.ply

    :param scan_dir: str
        Absolute path to the directory
        `raw/{{scans, scans_test}}/{{scan_name}}/{{scan_name}}`
    :param xyz: bool
        Whether XYZ coordinates should be saved in the output Data.pos
    :param rgb: bool
        Whether RGB colors should be saved in the output Data.rgb
    :param semantic: bool
        Whether semantic labels should be saved in the output Data.y
    :param instance: bool
        Whether instance labels should be saved in the output Data.obj
    :param remap: bool
        Whether semantic labels should be mapped from their NYU40 ID
        to their ScanNet ID
    """
    # Remove trailing slash, just in case
    scan_dir = scan_dir[:-1] if scan_dir[-1] == '/' else scan_dir

    # Extract the parent directory and the scan name from the scan
    # directory path. The parent directory will be used to identify test
    # scans
    scan_name = osp.basename(scan_dir)
    stage_dir = osp.dirname(scan_dir)
    stage_dirname = osp.basename(stage_dir)
    raw_dir = osp.dirname(stage_dir)

    # Build the path to the label mapping .tsv file, it is expected to
    # be in the raw/ folder
    label_map_file = osp.join(raw_dir, "scannetv2-labels.combined.tsv")

    # Scans are expected in the 'raw_dir/{scans, scans_test}/scan_name'
    # structure
    if stage_dirname not in ['scans', 'scans_test']:
        raise ValueError(
            "Expected the data to be in a "
            "'raw_dir/{scans, scans_test}/scan_name' structure, but parent "
            f"directory is {stage_dirname}")

    # Read the scan. Different reading methods for train/val scanes and
    # test scans
    if stage_dirname == 'scans':
        pos, color, y, obj = read_one_scan(stage_dir, scan_name, label_map_file)
        y = torch.from_numpy(NYU40_2_SCANNET)[y] if remap else y
        data = Data(pos=pos, rgb=color, y=y)
        idx = torch.arange(data.num_points)
        obj = consecutive_cluster(obj)[0]
        count = torch.ones_like(obj)
        data.obj = InstanceData(idx, obj, count, y, dense=True)
    else:
        pos, color = read_one_test_scan(stage_dir, scan_name)
        data = Data(pos=pos, rgb=color)

    # Remove unneeded attributes
    if not xyz:
        data.pos = None
    if not rgb:
        data.rgb = None
    if not semantic:
        data.y = None
    if not instance:
        data.obj = None

    return data


########################################################################
#                               ScanNet                                #
########################################################################

