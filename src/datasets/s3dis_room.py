import logging
from src.datasets.s3dis_config import *
from src.datasets.s3dis import read_s3dis_room, S3DIS

DIR = osp.dirname(osp.realpath(__file__))
log = logging.getLogger(__name__)


__all__ = ['S3DISRoom', 'MiniS3DISRoom']


########################################################################
#                               S3DIS                               #
########################################################################

class S3DISRoom(S3DIS):
    """S3DIS dataset, for aligned room-wise prediction.

    Dataset website: http://buildingparser.stanford.edu/dataset.html

    Parameters
    ----------
    root : `str`
        Root directory where the dataset should be saved.
    fold : `int`
        Integer in [1, ..., 6] indicating the Test Area
    stage : {'train', 'val', 'test', 'trainval'}, optional
    transform : `callable`, optional
        transform function operating on data.
    pre_transform : `callable`, optional
        pre_transform function operating on data.
    pre_filter : `callable`, optional
        pre_filter function operating on data.
    on_device_transform: `callable`, optional
        on_device_transform function operating on data, in the
        'on_after_batch_transfer' hook. This is where GPU-based
        augmentations should be, as well as any Transform you do not
        want to run in CPU-based DataLoaders
    """

    _zip_name = ALIGNED_ZIP_NAME
    _unzip_name = ALIGNED_UNZIP_NAME

    @property
    def all_cloud_ids(self):
        """Dictionary holding lists of clouds ids, for each
        stage.

        The following structure is expected:
            `{'train': [...], 'val': [...], 'test': [...]}`
        """
        return {
            'train': [
                f'Area_{i}/{r}' for i in range(1, 7) if i != self.fold
                for r in ROOMS[f'Area_{i}']],
            'val': [
                f'Area_{i}/{r}' for i in range(1, 7) if i != self.fold
                for r in ROOMS[f'Area_{i}']],
            'test': [
                f'Area_{self.fold}/{r}' for r in ROOMS[f'Area_{self.fold}']]}

    def read_single_raw_cloud(self, raw_cloud_path):
        """Read a single raw cloud and return a Data object, ready to
        be passed to `self.pre_transform`.
        """
        return read_s3dis_room(
            raw_cloud_path, xyz=True, rgb=True, semantic=True, instance=False,
            is_val=True, verbose=False)

    def id_to_relative_raw_path(self, id):
        """Given a cloud id as stored in `self.cloud_ids`, return the
        path (relative to `self.raw_dir`) of the corresponding raw
        cloud.
        """
        return id


########################################################################
#                              MiniS3DIS                               #
########################################################################

class MiniS3DISRoom(S3DISRoom):
    """A mini version of S3DIS with only 2 areas per stage for
    experimentation.
    """
    _NUM_MINI = 1

    @property
    def all_cloud_ids(self):
        return {k: v[:self._NUM_MINI] for k, v in super().all_cloud_ids.items()}

    @property
    def data_subdir_name(self):
        return self.__class__.__bases__[0].__name__.lower()

    # We have to include this method, otherwise the parent class skips
    # processing
    def process(self):
        super().process()

    # We have to include this method, otherwise the parent class skips
    # processing
    def download(self):
        super().download()
