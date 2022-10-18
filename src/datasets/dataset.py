import pytorch_lightning as pl
from src.loader import DataLoader


class BaseDataSet(pl.LightningDataModule):
    def __init__(self, root, batch_size=4):
        super().__init__()
        self.root = root
        self.batch_size = batch_size

    def prepare_data(self):
        """Here do the first instantiation of the train, val and test
        sets. This is when download should occur and any heavy
        preprocessing that might require multiple CPUs or the GPU.

        However, DO NOT set self attributes here, they will be out of
        scope.
        """
        raise NotImplementedError

    def setup(self, stage=None):
        """Instantiate self.train_set, self.val_set, self.test_set,
        and self.predict_set here.
        """
        raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(self.train_set, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.predict_set, batch_size=self.batch_size)
