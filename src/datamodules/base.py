import torch
import logging
from pytorch_lightning import LightningDataModule
from src.transforms import instantiate_transforms
from src.loader import DataLoader
from src.data import NAGBatch
from src.transforms import SampleGraphs, NAGSaveNodeIndex


log = logging.getLogger(__name__)


# List of transforms not allowed for test-time augmentation
_TTA_CONFLICTS = [SampleGraphs]


class BaseDataModule(LightningDataModule):
    """Base LightningDataModule class.

    Child classes should overwrite:

    ```
    MyDataModule(BaseDataModule):

        _DATASET_CLASS = ...
        _MINIDATASET_CLASS = ...
    ```

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """
    _DATASET_CLASS = None
    _MINIDATASET_CLASS = None

    def __init__(
            self, data_dir='', pre_transform=None, train_transform=None,
            val_transform=None, test_transform=None,
            on_device_train_transform=None, on_device_val_transform=None,
            on_device_test_transform=None, dataloader=None, mini=False,
            trainval=False, val_on_test=False, tta_runs=None, tta_val=False,
            **kwargs):
        super().__init__()

        # This line allows to access init params with 'self.hparams'
        # attribute also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.kwargs = kwargs

        # Make sure `_DATASET_CLASS` and `_MINIDATASET_CLASS` have been
        # specified
        if self.dataset_class is None:
            raise NotImplementedError

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Do not set the transforms directly, use self.set_transforms()
        # instead to parse the input configs
        self.pre_transform = None
        self.train_transform = None
        self.val_transform = None
        self.test_transform = None
        self.on_device_train_transform = None
        self.on_device_val_transform = None
        self.on_device_test_transform = None

        # Instantiate the transforms
        self.set_transforms()

        # Check TTA and transforms conflicts
        self.check_tta_conflicts()

    @property
    def dataset_class(self):
        """Return the LightningDataModule's Dataset class.
        """
        if self.hparams.mini:
            return self._MINIDATASET_CLASS
        return self._DATASET_CLASS

    @property
    def train_stage(self):
        """Return either 'train' or 'trainval' depending on how
        `self.hparams.trainval` is configured.
        """
        return 'trainval' if self.hparams.trainval else 'train'

    @property
    def val_stage(self):
        """Return either 'val' or 'test' depending on how
        `self.hparams.val_on_test` is configured.
        """
        return 'test' if self.hparams.val_on_test else 'val'

    def prepare_data(self):
        """Download and heavy preprocessing of data should be triggered
        here.

        However, do not use it to assign state (e.g. self.x = y) because
        it will not be preserved outside this scope.
        """
        self.dataset_class(
            self.hparams.data_dir, stage=self.train_stage,
            transform=self.train_transform, pre_transform=self.pre_transform,
            on_device_transform=self.on_device_train_transform, **self.kwargs)

        self.dataset_class(
            self.hparams.data_dir, stage=self.val_stage,
            transform=self.val_transform, pre_transform=self.pre_transform,
            on_device_transform=self.on_device_val_transform, **self.kwargs)

        self.dataset_class(
            self.hparams.data_dir, stage='test',
            transform=self.test_transform, pre_transform=self.pre_transform,
            on_device_transform=self.on_device_test_transform, **self.kwargs)

    def setup(self, stage=None):
        """Load data. Set variables: `self.train_dataset`,
        `self.val_dataset`, `self.test_dataset`.

        This method is called by lightning with both `trainer.fit()`
        and `trainer.test()`, so be careful not to execute things like
        random split twice!
        """
        self.train_dataset = self.dataset_class(
            self.hparams.data_dir, stage=self.train_stage,
            transform=self.train_transform, pre_transform=self.pre_transform,
            on_device_transform=self.on_device_train_transform, **self.kwargs)

        self.val_dataset = self.dataset_class(
            self.hparams.data_dir, stage=self.val_stage,
            transform=self.val_transform, pre_transform=self.pre_transform,
            on_device_transform=self.on_device_val_transform, **self.kwargs)

        self.test_dataset = self.dataset_class(
            self.hparams.data_dir, stage='test',
            transform=self.test_transform, pre_transform=self.pre_transform,
            on_device_transform=self.on_device_test_transform, **self.kwargs)

        self.predict_dataset = None

    def set_transforms(self):
        """Parse in self.hparams in search for '*transform*' keys and
        instantiate the corresponding transforms.

        Credit: https://github.com/torch-points3d/torch-points3d
        """
        for key_name in self.hparams.keys():
            if "transform" in key_name:
                name = key_name.replace("transforms", "transform")
                params = getattr(self.hparams, key_name, None)
                if params is None:
                    continue
                try:
                    transform = instantiate_transforms(params)
                except Exception:
                    log.exception(f"Error trying to create {name}, {params}")
                    continue
                setattr(self, name, transform)

    def check_tta_conflicts(self):
        """Make sure the transforms are Test-Time Augmentation-friendly
        """
        # Skip if not TTA
        if self.hparams.tta_runs is None or self.hparams.tta_runs == 1:
            return

        # Make sure all transforms are test-time augmentation friendly
        transforms = self.test_transform.transforms
        transforms += self.on_device_test_transform.transforms
        if self.hparams.tta_val:
            transforms += self.val_transform.transforms
            transforms += self.on_device_val_transform.transforms
        for t in transforms:
            if t in _TTA_CONFLICTS:
                raise NotImplementedError(
                    f"Cannot use {t} with test-time augmentation. The "
                    f"following transforms are not supported: {_TTA_CONFLICTS}")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.dataloader.batch_size,
            num_workers=self.hparams.dataloader.num_workers,
            pin_memory=self.hparams.dataloader.pin_memory,
            shuffle=True)

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.dataloader.batch_size,
            num_workers=self.hparams.dataloader.num_workers,
            pin_memory=self.hparams.dataloader.pin_memory,
            shuffle=False)

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.dataloader.batch_size,
            num_workers=self.hparams.dataloader.num_workers,
            pin_memory=self.hparams.dataloader.pin_memory,
            shuffle=False)

    def predict_dataloader(self):
        raise NotImplementedError

    def teardown(self, stage=None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict):
        """Things to do when loading checkpoint."""
        pass

    @torch.no_grad()
    def on_after_batch_transfer(self, nag_list, dataloader_idx):
        """Intended to call on-device operations. Typically,
        NAGBatch.from_nag_list and some Transforms like SampleSubNodes
        and SampleSegments are faster on GPU, and we may prefer
        executing those on GPU rather than in CPU-based DataLoader.

        Use self.on_device_<stage>_transform, to benefit from this hook.
        """
        # Since NAGBatch.from_nag_list takes a bit of time, we asked
        # src.loader.DataLoader to simply pass a list of NAG objects,
        # waiting for to be batched on device.
        nag = NAGBatch.from_nag_list(nag_list)
        del nag_list

        # Here we run on_device_transform, which contains NAG transforms
        # that we could not / did not want to run using CPU-based
        # DataLoaders
        if self.trainer.training:
            on_device_transform = self.on_device_train_transform
        elif self.trainer.validating:
            on_device_transform = self.on_device_val_transform
        elif self.trainer.testing:
            on_device_transform = self.on_device_test_transform
        elif self.trainer.predicting:
            raise NotImplementedError('No on_device_predict_transform yet...')
        else:
            log.warning(
                'Unsure which stage we are in, defaulting to '
                'self.on_device_train_transform')
            on_device_transform = self.on_device_train_transform

        # Skip on_device_transform if None
        if on_device_transform is None:
            return nag

        # Apply on_device_transform only once when in training mode and
        # if no test-time augmentation is required
        if self.trainer.training \
                or self.hparams.tta_runs is None \
                or self.hparams.tta_runs == 1 or \
                (self.trainer.validating and not self.hparams.tta_val):
            return on_device_transform(nag)

        # TODO : TTA
        #  - add node identifier (which level ?)
        #  - run on device transform multiple times and accumulate into list of NAGs
        #  - make SURE ALL INPUT NODES ARE COVERED !!!
        # Run test-time augmentations and produce a list of NAGs for
        # each inference run. Since the augmentations may change the
        # sampling of the nodes, we save their input id here before
        # anything. This will allow us to fuse the multiple predictions
        # for each node in the `LightningModule.step()`. We return the
        # input NAG as well as the list of augmented NAGs. Passing the
        # input NAG will help us make sure all nodes have received a
        # prediction
        nag = NAGSaveNodeIndex()(nag)
        nag_list = []
        for i_run in range(self.hparams.tta_runs):
            nag_list.append(on_device_transform(nag.clone()))
        return nag, nag_list

    def __repr__(self):
        return f'{self.__class__.__name__}'
