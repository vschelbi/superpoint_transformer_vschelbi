from typing import Any, List
import torch
import logging
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from src.metrics import ConfusionMatrix
from src.utils import loss_with_target_histogram, atomic_to_histogram


log = logging.getLogger(__name__)


class PointNetModule(LightningModule):
    """A LightningModule for PointNet."""

    def __init__(
            self, net, criterion, optimizer, scheduler, num_classes,
            class_names=None, pointwise_loss=True, weighted_loss=True):
        super().__init__()

        # Allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        # nn.Module that will do the actual computation
        self.net = net

        # Loss function. We add `ignore_index=num_classes` to account
        # for unclassified/ignored points, which are given num_classes
        # labels
        self.criterion = criterion
        self.pointwise_loss = pointwise_loss
        self.weighted_loss = weighted_loss

        # Metric objects for calculating and averaging accuracy across
        # batches. We add `ignore_index=num_classes` to account for
        # unclassified/ignored points, which are given num_classes
        # labels
        self.num_classes = num_classes
        self.class_names = class_names if class_names is not None \
            else [f'class-{i}' for i in range(num_classes)]
        self.train_cm = ConfusionMatrix(
            num_classes, ignore_index=num_classes)
        self.val_cm = ConfusionMatrix(
            num_classes, ignore_index=num_classes)
        self.test_cm = ConfusionMatrix(
            num_classes, ignore_index=num_classes)

        # For averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # For tracking best-so-far validation metrics
        self.val_miou_best = MaxMetric()
        self.val_oa_best = MaxMetric()
        self.val_macc_best = MaxMetric()

    def forward(self, pos, x, idx):
        return self.net(pos, x, idx)

    def on_fit_start(self):
        # This is a bit of a late initialization for the LightningModule
        # At this point, we can access some LightningDataModule-related
        # parameters that were not available beforehand. So we take this
        # opportunity to catch the number of classes or class weights
        # from the LightningDataModule

        # Get the LightningDataModule number of classes and make sure it
        # matches self.num_classes. We could also forcefully update the
        # LightningModule with this new information but it could easily
        # become tedious to track all places where num_classes affects
        # the LightningModule object.
        num_classes = self.trainer.datamodule.train_dataset.num_classes
        assert num_classes == self.num_classes, \
            f'LightningModule has {self.num_classes} classes while the ' \
            f'LightningDataModule has {num_classes} classes.'

        self.class_names = self.trainer.datamodule.train_dataset.class_names

        if not self.weighted_loss:
            return

        if not hasattr(self.criterion, 'weight'):
            log.warning(
                f"{self.criterion} does not have a 'weight' attribute. "
                f"Class weights will be ignored...")
            return

        # Set class weights for the
        weight = self.trainer.datamodule.train_dataset.get_class_weight()
        self.criterion.weight = weight.to(self.device)

    def on_train_start(self):
        # By default lightning executes validation step sanity checks
        # before training starts, so we need to make sure `*_best`
        # metrics do not store anything from these checks
        self.val_cm.reset()
        self.val_miou_best.reset()
        self.val_oa_best.reset()
        self.val_macc_best.reset()

    def step(self, nag):
        # Recover level-0 features, position, segment indices and labels
        x = nag[0].x
        pos = nag[0].pos
        idx = nag[0].super_index
        y = nag[0].y

        # Convert level-0 labels to segment-level histograms, while
        # accounting for the extra class for unlabeled/ignored points
        y_hist = atomic_to_histogram(y, idx, n_bins=self.num_classes + 1)

        # Remove the last bin of the histogram, accounting for
        # unlabeled/ignored points
        y_hist = y_hist[:, :self.num_classes]

        # Inference on the batch
        logits = self.forward(pos, x, idx)
        preds = torch.argmax(logits, dim=1)

        # Compute the loss either in a point-wise or segment-wise
        # fashion
        if self.pointwise_loss:
            loss = loss_with_target_histogram(self.criterion, logits, y_hist)
        else:
            loss = self.criterion(logits, y_hist.argmax(dim=1))

        return loss, preds, y_hist

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_cm(preds, targets)
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True,
            prog_bar=True)

        # we can return here dict with any tensors and then read it in
        # some callback or in `training_epoch_end()` below
        # Remember to always return loss from `training_step()` or
        # backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        self.log("train/miou", self.train_cm.miou(), prog_bar=True)
        self.log("train/oa", self.train_cm.oa(), prog_bar=True)
        self.log("train/macc", self.train_cm.macc(), prog_bar=True)
        for iou, seen, name in zip(*self.train_cm.iou(), self.class_names):
            if seen:
                self.log(f"train/iou_{name}", iou, prog_bar=True)
        self.train_cm.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_cm(preds, targets)
        self.log(
            "val/loss", self.val_loss, on_step=False, on_epoch=True,
            prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `validation_step()`
        miou = self.val_cm.miou()
        oa = self.val_cm.oa()
        macc = self.val_cm.macc()

        self.log("val/miou", miou, prog_bar=True)
        self.log("val/oa", oa, prog_bar=True)
        self.log("val/macc", macc, prog_bar=True)
        for iou, seen, name in zip(*self.val_cm.iou(), self.class_names):
            if seen:
                self.log(f"val/iou_{name}", iou, prog_bar=True)

        self.val_miou_best(miou)  # update best-so-far metric
        self.val_oa_best(oa)  # update best-so-far metric
        self.val_macc_best(macc)  # update best-so-far metric

        # log `*_best` metrics this way, using `.compute()` instead of
        # passing the whole torchmetric object, because otherwise metric
        # would be reset by lightning after each epoch
        self.log("val/miou_best", self.val_miou_best.compute(), prog_bar=True)
        self.log("val/oa_best", self.val_oa_best.compute(), prog_bar=True)
        self.log("val/macc_best", self.val_macc_best.compute(), prog_bar=True)

        self.val_cm.reset()

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_cm(preds, targets)
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True,
            prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `test_step()`
        self.log("test/miou", self.test_cm.miou(), prog_bar=True)
        self.log("test/oa", self.test_cm.oa(), prog_bar=True)
        self.log("test/macc", self.test_cm.macc(), prog_bar=True)
        for iou, seen, name in zip(*self.test_cm.iou(), self.class_names):
            if seen:
                self.log(f"test/iou_{name}", iou, prog_bar=True)
        self.test_cm.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1}}
        return {"optimizer": optimizer}


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "pointnet.yaml")
    _ = hydra.utils.instantiate(cfg)
