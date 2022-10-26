import logging
from pytorch_lightning import LightningDataModule
from src.transforms import instantiate_transforms
from src.datasets import KITTI360
from src.loader import DataLoader
from src.data import NAGBatch


log = logging.getLogger(__name__)


class KITTI360DataModule(LightningDataModule):
    """LightningDataModule for KITTI360 dataset.

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

    def __init__(
            self, data_dir='', x32=True, y_to_csr=True, pre_transform=None,
            train_transform=None, val_transform=None, test_transform=None,
            on_device_train_transform=None, on_device_val_transform=None,
            on_device_test_transform=None, dataloader=None):
        super().__init__()

        # this line allows to access init params with 'self.hparams'
        # attribute also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

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

    def prepare_data(self):
        """Download and heavy preprocessing of data should be triggered
        here.

        However, do not use it to assign state (eg self.x = y) because
        it will not be preserved outside of this scope.
        """
        KITTI360(
            self.hparams.data_dir, stage='train',
            transform=self.train_transform, pre_transform=self.pre_transform,
            on_device_transform=self.on_device_train_transform,
            x32=self.hparams.x32, y_to_csr=self.hparams.y_to_csr)

        KITTI360(
            self.hparams.data_dir, stage='val',
            transform=self.val_transform, pre_transform=self.pre_transform,
            on_device_transform=self.on_device_val_transform,
            x32=self.hparams.x32, y_to_csr=self.hparams.y_to_csr)

        KITTI360(
            self.hparams.data_dir, stage='test',
            transform=self.test_transform, pre_transform=self.pre_transform,
            on_device_transform=self.on_device_test_transform,
            x32=self.hparams.x32, y_to_csr=self.hparams.y_to_csr)

    def setup(self, stage=None):
        """Load data. Set variables: `self.train_dataset`,
        `self.val_dataset`, `self.test_dataset`.

        This method is called by lightning with both `trainer.fit()`
        and `trainer.test()`, so be careful not to execute things like
        random split twice!
        """
        self.train_dataset = KITTI360(
            self.hparams.data_dir, stage='train',
            transform=self.train_transform, pre_transform=self.pre_transform,
            on_device_transform=self.on_device_train_transform,
            x32=self.hparams.x32, y_to_csr=self.hparams.y_to_csr)

        self.val_dataset = KITTI360(
            self.hparams.data_dir, stage='val',
            transform=self.val_transform, pre_transform=self.pre_transform,
            on_device_transform=self.on_device_val_transform,
            x32=self.hparams.x32, y_to_csr=self.hparams.y_to_csr)

        self.test_dataset = KITTI360(
            self.hparams.data_dir, stage='test',
            transform=self.test_transform, pre_transform=self.pre_transform,
            on_device_transform=self.on_device_test_transform,
            x32=self.hparams.x32, y_to_csr=self.hparams.y_to_csr)

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

    def on_after_batch_transfer(self, nag_list, dataloader_idx):
        """Intended to call on-device operations. Typically,
        NAGBatch.from_nag_list and some Transforms like SampleSegments
        and DropoutSegments are faster on GPU and we may prefer
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
        if on_device_transform is None:
            return nag
        else:
            return on_device_transform(nag).detach()

    def __repr__(self):
        return f'{self.__class__.__name__}'


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "kitti360.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
