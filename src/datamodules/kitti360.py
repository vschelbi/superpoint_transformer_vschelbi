import logging
from pytorch_lightning import LightningDataModule
from src.transforms import instantiate_transforms
from src.datasets import KITTI360
from src.loader import DataLoader


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
            dataloader=None):
        super().__init__()

        # this line allows to access init params with 'self.hparams'
        # attribute also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.pre_transform = None
        self.test_transform = None
        self.train_transform = None
        self.val_transform = None

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
            x32=self.hparams.x32, y_to_csr=self.hparams.y_to_csr)

        KITTI360(
            self.hparams.data_dir, stage='val',
            transform=self.val_transform, pre_transform=self.pre_transform,
            x32=self.hparams.x32, y_to_csr=self.hparams.y_to_csr)

        KITTI360(
            self.hparams.data_dir, stage='test',
            transform=self.test_transform, pre_transform=self.pre_transform,
            x32=self.hparams.x32, y_to_csr=self.hparams.y_to_csr)

    def setup(self, stage=None):
        """Load data. Set variables: `self.data_train`, `self.data_val`,
        `self.data_test`.

        This method is called by lightning with both `trainer.fit()`
        and `trainer.test()`, so be careful not to execute things like
        random split twice!
        """
        if stage is None or stage == 'train':
            self.train_dataset = KITTI360(
                self.hparams.data_dir, stage='train',
                transform=self.train_transform, pre_transform=self.pre_transform,
                x32=self.hparams.x32, y_to_csr=self.hparams.y_to_csr)

        if stage is None or stage == 'val':
            self.val_dataset = KITTI360(
                self.hparams.data_dir, stage='val',
                transform=self.val_transform, pre_transform=self.pre_transform,
                x32=self.hparams.x32, y_to_csr=self.hparams.y_to_csr)

        if stage is None or stage == 'test':
            self.test_dataset = KITTI360(
                self.hparams.data_dir, stage='test',
                transform=self.test_transform, pre_transform=self.pre_transform,
                x32=self.hparams.x32, y_to_csr=self.hparams.y_to_csr)

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
            dataset=self.data_train,
            batch_size=self.hparams.dataloader.batch_size,
            num_workers=self.hparams.dataloader.num_workers,
            pin_memory=self.hparams.dataloader.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.dataloader.batch_size,
            num_workers=self.hparams.dataloader.num_workers,
            pin_memory=self.hparams.dataloader.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.dataloader.batch_size,
            num_workers=self.hparams.dataloader.num_workers,
            pin_memory=self.hparams.dataloader.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage=None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "kitti360.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
