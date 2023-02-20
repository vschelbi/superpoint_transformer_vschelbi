import torch
import logging
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from src.metrics import ConfusionMatrix
from src.utils import loss_with_target_histogram, atomic_to_histogram, \
    init_weights, wandb_confusion_matrix, knn_2
from src.nn import Classifier
from src.optim.lr_scheduler import ON_PLATEAU_SCHEDULERS
from pytorch_lightning.loggers.wandb import WandbLogger
from src.data import NAG
from src.transforms import NAGSaveNodeIndex


log = logging.getLogger(__name__)


class PointSegmentationModule(LightningModule):
    """A LightningModule for semantic segmentation of point clouds."""

    def __init__(
            self, net, criterion, optimizer, scheduler, num_classes,
            class_names=None, sampling_loss=False, pointwise_loss=True,
            weighted_loss=True, custom_init=True, transformer_lr_scale=1,
            **kwargs):
        super().__init__()

        # Allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=['net', 'criterion'])

        # Network that will do the actual computation
        self.net = net

        # Segmentation head
        self.head = Classifier(self.net.out_dim, num_classes)

        # Custom weight initialization. In particular, this applies
        # Xavier / Glorot initialization on Linear layers
        if custom_init:
            self.net.apply(init_weights)
            self.head.apply(init_weights)

        # Loss function. We add `ignore_index=num_classes` to account
        # for unclassified/ignored points, which are given num_classes
        # labels
        self.criterion = criterion

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

    def forward(self, nag):
        x = self.net(nag)
        return self.head(x)

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

        if not self.hparams.weighted_loss:
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
        # By default, lightning executes validation step sanity checks
        # before training starts, so we need to make sure `*_best`
        # metrics do not store anything from these checks
        self.val_cm.reset()
        self.val_miou_best.reset()
        self.val_oa_best.reset()
        self.val_macc_best.reset()

    def step(self, batch):
        # Forward step on the input batch. If a (NAG, List(NAG)) is
        # passed, the multi-run inference will be triggered
        if isinstance(batch, NAG):
            logits, preds, y_hist = self.step_single_run_inference(batch)
        else:
            logits, preds, y_hist = self.step_multi_run_inference(*batch)

        # Compute the loss either in a point-wise or segment-wise
        # fashion
        if self.hparams.pointwise_loss:
            loss = loss_with_target_histogram(self.criterion, logits, y_hist)
        else:
            loss = self.criterion(logits, y_hist.argmax(dim=1))

        return loss, preds, y_hist

    def step_single_run_inference(self, nag):
        """Single-run inference
        """
        y_hist = self.step_get_y_hist(nag)
        logits = self.forward(nag)
        preds = torch.argmax(logits, dim=1)
        return logits, preds, y_hist

    def step_multi_run_inference(self, nag, transform, num_runs):
        """Multi-run inference, typically with test-time augmentation.
        See `BaseDataModule.on_after_batch_transfer`
        """
        # Since the transform may change the sampling of the nodes, we
        # save their input id here before anything. This will allow us
        # to fuse the multiple predictions for each node
        transform.transforms = [NAGSaveNodeIndex()] + transform.transforms

        # Recover the target labels from the reference NAG
        y_hist = self.step_get_y_hist(nag)

        # Build the global logits, in which the multi-run
        # logits will be accumulated, before computing their final
        # prediction
        logits = torch.zeros_like(y_hist, dtype=torch.float)
        seen = torch.zeros_like(y_hist[:, 0], dtype=torch.bool)

        for i_run in range(num_runs):
            # Apply transform
            nag_ = transform(nag.clone())

            # Recover the node identifier that should have been
            # implanted by `BaseDataModule.on_after_batch_transfer`
            node_id = nag_[1][NAGSaveNodeIndex.KEY]

            # Forward on the augmented data and update the global
            # logits of the node
            logits[node_id] += self.forward(nag_)
            seen[node_id] = True

        # If some nodes were not seen across any of the multi-runs,
        # search their nearest seen neighbor
        unseen_idx = torch.where(~seen)[0]
        if unseen_idx.shape[0] > 0:
            seen_idx = torch.where(seen)[0]
            x_search = nag[1].pos[seen_idx]
            x_query = nag[1].pos[unseen_idx]
            neighbors = knn_2(x_search, x_query, 1, r_max=2)[0]
            num_unseen = unseen_idx.shape[0]
            num_seen = seen_idx.shape[0]
            num_left_out = (neighbors == -1).sum().long()
            if num_left_out > 0:
                log.warning(
                    f"Could not find a neighbor for all unseen nodes: num_seen="
                    f"{num_seen}, num_unseen={num_unseen}, num_left_out="
                    f"{num_left_out}. These left out nodes will default to "
                    f"label-0 class prediction. Consider sampling less nodes "
                    f"in the augmentations, or increase the search radius")
            logits[unseen_idx] = logits[seen_idx][neighbors]

        # Compute the global prediction
        preds = torch.argmax(logits, dim=1)

        return logits, preds, y_hist

    def step_get_y_hist(self, nag):
        # Recover level-1 label histograms, either from the level-0
        # sampled points (ie sampling will affect the loss and metrics)
        # or directly from the precomputed level-1 label histograms (ie
        # true annotations)
        if self.hparams.sampling_loss:
            idx = nag[0].super_index
            y = nag[0].y

            # Convert level-0 labels to segment-level histograms, while
            # accounting for the extra class for unlabeled/ignored points
            y_hist = atomic_to_histogram(y, idx, n_bins=self.num_classes + 1)
        else:
            y_hist = nag[1].y

        # Remove the last bin of the histogram, accounting for
        # unlabeled/ignored points
        y_hist = y_hist[:, :self.num_classes]

        return y_hist

    def training_step(self, batch, batch_idx):
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

    def training_epoch_end(self, outputs):
        # `outputs` is a list of dicts returned from `training_step()`
        self.log("train/miou", self.train_cm.miou(), prog_bar=True)
        self.log("train/oa", self.train_cm.oa(), prog_bar=True)
        self.log("train/macc", self.train_cm.macc(), prog_bar=True)
        for iou, seen, name in zip(*self.train_cm.iou(), self.class_names):
            if seen:
                self.log(f"train/iou_{name}", iou, prog_bar=True)
        self.train_cm.reset()

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_cm(preds, targets)
        self.log(
            "val/loss", self.val_loss, on_step=False, on_epoch=True,
            prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs):
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

    def test_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_cm(preds, targets)
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True,
            prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs):
        # `outputs` is a list of dicts returned from `test_step()`
        self.log("test/miou", self.test_cm.miou(), prog_bar=True)
        self.log("test/oa", self.test_cm.oa(), prog_bar=True)
        self.log("test/macc", self.test_cm.macc(), prog_bar=True)
        for iou, seen, name in zip(*self.test_cm.iou(), self.class_names):
            if seen:
                self.log(f"test/iou_{name}", iou, prog_bar=True)

        # Log confusion matrix to wandb
        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.log({
                "test/cm": wandb_confusion_matrix(
                    self.test_cm.confmat, class_names=self.class_names)})

        self.test_cm.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        # Differential learning rate for transformer blocks
        t_names = ['transformer_blocks', 'down_pool_block']
        lr = self.hparams.optimizer.keywords['lr']
        t_lr = lr * self.hparams.transformer_lr_scale
        param_dicts = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if all([t not in n for t in t_names]) and p.requires_grad]},
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any([t in n for t in t_names]) and p.requires_grad],
                "lr": t_lr}]
        optimizer = self.hparams.optimizer(params=param_dicts)

        # Return the optimizer if no scheduler in the config
        if self.hparams.scheduler is None:
            return {"optimizer": optimizer}

        # Build the scheduler, with special attention for plateau-like
        # schedulers, which
        scheduler = self.hparams.scheduler(optimizer=optimizer)
        reduce_on_plateau = isinstance(scheduler, ON_PLATEAU_SCHEDULERS)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
                "reduce_on_plateau": reduce_on_plateau}}

    def load_state_dict(self, state_dict, strict=True):
        # Little bit of acrobatics due to `criterion.weight`. This
        # attribute, when present in the `state_dict`, causes
        # `load_state_dict` to crash.
        try:
            super().load_state_dict(state_dict, strict=strict)
        except:
            class_weight = state_dict.pop('criterion.weight', None)
            super().load_state_dict(state_dict, strict=strict)
            self.criterion.weight = class_weight


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = str(pyrootutils.setup_root(__file__, pythonpath=True))
    cfg = omegaconf.OmegaConf.load(root + "/configs/model/s1.yaml")
    _ = hydra.utils.instantiate(cfg)
