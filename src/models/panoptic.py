import torch
import logging
from torchmetrics import MaxMetric, MeanMetric
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from src.utils import init_weights
from src.metrics import MeanAveragePrecision3D, PanopticQuality3D
from src.models.segmentation import SemanticSegmentationOutput, \
    SemanticSegmentationModule
from src.loss import OffsetLoss
from src.nn import FFN

log = logging.getLogger(__name__)


__all__ = ['PanopticSegmentationOutput', 'PanopticSegmentationModule']


class PanopticSegmentationOutput(SemanticSegmentationOutput):
    def __init__(
            self,
            logits,
            edge_affinity_logits,
            node_offset_pred,
            y_hist=None,
            obj_edge_index=None,
            obj_edge_affinity=None,
            pos=None,
            obj_pos=None):
        # We set the child class attributes before calling the parent
        # class constructor, because the parent constructor calls
        # `self.debug()`, which needs all attributes to be initialized
        self.edge_affinity_logits = edge_affinity_logits
        self.node_offset_pred = node_offset_pred
        self.obj_edge_index = obj_edge_index
        self.obj_edge_affinity = obj_edge_affinity
        self.pos = pos
        self.obj_pos = obj_pos
        super().__init__(logits, y_hist=y_hist)

    def debug(self):
        # Parent class debugger
        super().debug()

        # Instance predictions
        assert self.node_offset_pred.dim() == 2
        assert self.node_offset_pred.shape[0] == self.num_nodes
        assert self.edge_affinity_logits.dim() == 1

        # Instance targets
        items = [
            self.obj_edge_index, self.obj_edge_affinity, self.pos, self.obj_pos]
        without_instance_targets = all(x is None for x in items)
        with_instance_targets = all(x is not None for x in items)
        assert without_instance_targets or with_instance_targets

        if without_instance_targets:
            return

        assert self.obj_edge_index.dim() == 2
        assert self.obj_edge_index.shape[0] == 2
        assert self.obj_edge_index.shape[1] == self.num_edges
        assert self.obj_edge_affinity.dim() == 1
        assert self.obj_edge_affinity.shape[0] == self.num_edges
        assert self.pos.shape == self.node_offset_pred.shape
        assert self.obj_pos.shape == self.node_offset_pred.shape

    @property
    def has_target(self):
        """Check whether `self` contains target data for panoptic
        segmentation.
        """
        items = [
            self.obj_edge_index, self.obj_edge_affinity, self.pos, self.obj_pos]
        return super().has_target and all(x is not None for x in items)

    @property
    def num_edges(self):
        """Number for edges in the instance graph.
        """
        return self.edge_affinity_logits.shape[1]

    @property
    def node_offset(self):
        """Target node offset: `offset = obj_pos - pos`.
        """
        if not self.has_target:
            return
        return self.obj_pos - self.pos

    @property
    def edge_affinity_preds(self):
        """Simply applies a sigmoid on `edge_affinity_logits` to produce
        the actual affinity predictions to be used for superpoint
        graph clustering.
        """
        return torch.sigmoid(self.edge_affinity_logits)

    @property
    def void_edge_mask(self):
        """Returns a mask on the edges indicating those connecting two
        void nodes.
        """
        if not self.has_target:
            return

        mask = self.void_mask[self.obj_edge_index]
        return mask[0] and mask[1]


class PanopticSegmentationModule(SemanticSegmentationModule):
    """A LightningModule for panoptic segmentation of point clouds.
    """

    _IGNORED_HYPERPARAMETERS = [
        'net', 'criterion', 'edge_affinity_criterion', 'node_offset_criterion']

    def __init__(
            self,
            net,
            criterion,
            optimizer,
            scheduler,
            num_classes,
            stuff_classes,
            class_names=None,
            sampling_loss=False,
            loss_type='ce_kl',
            weighted_loss=True,
            init_linear=None,
            init_rpe=None,
            transformer_lr_scale=1,
            multi_stage_loss_lambdas=None,
            edge_affinity_criterion=None,
            edge_affinity_loss_lambda=1,
            node_offset_criterion=None,
            node_offset_loss_lambda=1,
            gc_every_n_steps=0,
            min_instance_size=0,
            **kwargs):
        super().__init__(
            net,
            criterion,
            optimizer,
            scheduler,
            num_classes,
            class_names=None,
            sampling_loss=False,
            loss_type=True,
            weighted_loss=True,
            init_linear=None,
            init_rpe=None,
            transformer_lr_scale=1,
            multi_stage_loss_lambdas=None,
            gc_every_n_steps=0,
            **kwargs)

        # Store the stuff class indices
        self.stuff_classes = stuff_classes

        # Loss functions for edge affinity and node offset predictions.
        # NB: the semantic loss is already accounted for in the
        # SemanticSegmentationModule constructor
        self.edge_affinity_criterion = torch.nn.BCEWithLogitsLoss() \
            if edge_affinity_criterion is None else edge_affinity_criterion
        self.node_offset_criterion = OffsetLoss(spatial_mean=False) \
            if node_offset_criterion is None else node_offset_criterion

        # Model heads for edge affinity and node offset predictions
        self.edge_affinity_head = FFN(self.net.out_dim, hidden_dim=32, out_dim=1)
        self.node_offset_head = FFN(self.net.out_dim, hidden_dim=32, out_dim=3)

        # Custom weight initialization. In particular, this applies
        # Xavier / Glorot initialization on Linear and RPE layers by
        # default, but can be tuned
        init = lambda m: init_weights(m, linear=init_linear, rpe=init_rpe)
        self.edge_affinity_head.apply(init)
        self.node_offset_head.apply(init)

        # Metric objects for calculating scores on each dataset split.
        # We add `ignore_index=num_classes` to account for
        # void/unclassified/ignored points, which are given
        # `num_classes` labels
        self.train_instance = MeanAveragePrecision3D(
            self.num_classes,
            stuff_classes=self.stuff_classes,
            min_size=min_instance_size,
            compute_on_cpu=True,
            remove_void=True,
            **kwargs)
        self.train_panoptic = PanopticQuality3D(
            self.num_classes,
            ignore_unseen_classes=True,
            stuff_classes=self.stuff_classes,
            compute_on_cpu=True,
            **kwargs)
        self.val_instance = MeanAveragePrecision3D(
            self.num_classes,
            stuff_classes=self.stuff_classes,
            min_size=min_instance_size,
            compute_on_cpu=True,
            remove_void=True,
            **kwargs)
        self.val_panoptic = PanopticQuality3D(
            self.num_classes,
            ignore_unseen_classes=True,
            stuff_classes=self.stuff_classes,
            compute_on_cpu=True,
            **kwargs)
        self.test_instance = MeanAveragePrecision3D(
            self.num_classes,
            stuff_classes=self.stuff_classes,
            min_size=min_instance_size,
            compute_on_cpu=True,
            remove_void=True,
            **kwargs)
        self.test_panoptic = PanopticQuality3D(
            self.num_classes,
            ignore_unseen_classes=True,
            stuff_classes=self.stuff_classes,
            compute_on_cpu=True,
            **kwargs)

        # For averaging loss across batches
        self.train_edge_affinity_loss = MeanMetric()
        self.train_node_offset_loss = MeanMetric()
        self.val_edge_affinity_loss = MeanMetric()
        self.val_node_offset_loss = MeanMetric()
        self.test_edge_affinity_loss = MeanMetric()
        self.test_node_offset_loss = MeanMetric()

        # For tracking best-so-far validation metrics
        self.val_map_best = MaxMetric()
        self.val_pq_best = MaxMetric()
        self.val_pqmod_best = MaxMetric()
        self.val_mprec_best = MaxMetric()
        self.val_mrec_best = MaxMetric()

    def forward(self, nag):
        # Extract features
        x = self.net(nag)

        # Compute level-1 or multi-level semantic predictions
        semantic_pred = [head(x_) for head, x_ in zip(self.head, x)] \
            if self.multi_stage_loss else self.head(x)

        # Recover level-1 features only
        x = x[0] if self.multi_stage_loss else x

        # Compute node offset predictions
        node_offset_pred = self.node_offset_head(x)

        # Forcefully set 0-offset for nodes with stuff predictions.
        # By convention, we want stuff nodes to have 0 offset. Two
        # reasons for that:
        #   - defining a stuff target center is ambiguous
        #   - by predicting 0 offsets, the corresponding nodes are
        #     likely to be isolated by the superpoint clustering step.
        #     This is what we want, because the predictions will be
        #     merged as a post-processing step, to ensure there is a
        #     most one prediction per batch item for each stuff class
        y = semantic_pred[0].argmax(dim=1) \
            if self.multi_stage_loss else semantic_pred.argmax(dim=1)
        stuff_classes = torch.tensor(
            self.stuff_classes, dtype=y.dtype, device=y.device)
        is_stuff = torch.isin(y, stuff_classes)
        node_offset_pred[is_stuff] = 0

        # Compute edge affinity predictions
        # NB: we make edge features symmetric, since we want to compute
        # edge affinity, which is not directed
        x_edge = x[nag[1].obj_edge_index]
        x_edge = torch.cat(
            ((x_edge[0] - x_edge[1]).abs() / 2, x_edge[0] + x_edge[1]), dim=1)
        edge_affinity_logits = self.edge_affinity_head(x_edge)

        return PanopticSegmentationOutput(
            semantic_pred, edge_affinity_logits, node_offset_pred)

    def on_fit_start(self):
        super().on_fit_start()

        # Get the LightningDataModule stuff classes and make sure it
        # matches self.stuff_classes. We could also forcefully update
        # the LightningModule with this new information but it could
        # easily become tedious to track all places where stuff_classes
        # affects the LightningModule object.
        stuff_classes = self.trainer.datamodule.train_dataset.stuff_classes
        assert sorted(stuff_classes) == sorted(self.stuff_classes), \
            f'LightningModule has the following stuff classes ' \
            f'{self.stuff_classes} while the LightningDataModule has ' \
            f'{stuff_classes}.'

    def on_train_start(self):
        # By default, lightning executes validation step sanity checks
        # before training starts, so we need to make sure `*_best`
        # metrics do not store anything from these checks
        super().on_train_start()
        self.val_map_best.reset()
        self.val_pq_best.reset()
        self.val_pqmod_best.reset()
        self.val_mprec_best.reset()
        self.val_mrec_best.reset()

    def _create_empty_output(self, nag):
        """Local helper method to initialize an empty output for
        multi-run prediction.
        """
        # Prepare empty output for semantic segmentation
        output_semseg = super()._create_empty_output(nag)

        # Prepare empty edge affinity and node offset outputs
        num_edges = nag[1].obj_edge_index.shape[1]
        edge_affinity_logits = torch.zeros(num_edges, device=nag.device)
        node_offset_pred = torch.zeros_like(nag[1].pos)

        return PanopticSegmentationOutput(
            output_semseg.logits,
            edge_affinity_logits,
            node_offset_pred)

    @staticmethod
    def _update_output_multi(output_multi, nag, output, nag_transformed, key):
        """Local helper method to accumulate multiple predictions on
        the same -or part of the same- point cloud.
        """
        # Update semantic segmentation logits only
        output_multi = super()._update_output_multi(
            output_multi, nag, output, nag_transformed, key)

        # Update node-wise predictions
        # TODO: this is INCORRECT accumulation of node offsets. Need to
        #  define the mean, not the mean of the successive predictions
        node_id = nag_transformed[1][key]
        output_multi.node_offset_pred[node_id] = \
            (output_multi.node_offset_pred[node_id]
             + output.node_offset_pred) / 2

        # Update edge-wise predictions
        edge_index_1 = nag[1].obj_edge_index
        edge_index_2 = node_id[nag_transformed[1].obj_edge_index]
        base = nag[1].num_points + 1
        edge_id_1 = edge_index_1[0] * base + edge_index_1[1]
        edge_id_2 = edge_index_2[0] * base + edge_index_2[1]
        edge_id_cat = consecutive_cluster(torch.cat((edge_id_1, edge_id_2)))[0]
        edge_id_1 = edge_id_cat[:edge_id_1.numel()]
        edge_id_2 = edge_id_cat[edge_id_1.numel():]
        pivot = torch.zeros(base**2, device=output.edge_affinity_logits)
        pivot[edge_id_1] = output_multi.edge_affinity_logits
        # TODO: this is INCORRECT accumulation of node offsets. Need to
        #  define the mean, not the mean of the successive predictions
        pivot[edge_id_2] = (pivot[edge_id_2] + output.edge_affinity_logits) / 2
        output_multi.edge_affinity_logits = pivot[edge_id_1]

        return output_multi

    @staticmethod
    def _propagate_output_to_unseen_neighbors(output, nag, seen, neighbors):
        """Local helper method to propagate predictions to unseen
        neighbors.
        """
        # Propagate semantic segmentation to neighbors
        output = super()._propagate_output_to_unseen_neighbors(
            output, nag, seen, neighbors)

        # Heuristic for unseen node offsets: unseen nodes take the same
        # offset as their nearest neighbor
        seen_idx = torch.where(seen)[0]
        unseen_idx = torch.where(~seen)[0]
        output.node_offset_pred[unseen_idx] = \
            output.node_offset_pred[seen_idx][neighbors]

        # Heuristic for unseen edge affinity predictions: we set the
        # edge affinity to 0.5
        seen_edge = nag[1].obj_edge_index[seen]
        unseen_edge_idx = torch.where(~seen_edge)[0]
        output.edge_affinity_logits[unseen_edge_idx] = 0.5

        return output

    def get_target(self, nag, output):
        """Recover the target data for semantic and panoptic
        segmentation and store it in the `output` object.

        More specifically:
          - label histogram(s) for semantic segmentation will be saved
            in `output.y_hist`
          - instance graph data `obj_edge_index` and `obj_edge_affinity`
            will be saved in `output.obj_edge_index` and
            `output.obj_edge_affinity`, respectively
          - node positions `pos` and `obj_pos` will be saved in
            `output.pos` and `output.obj_pos`, respectively. Besides,
            the `output.obj_offset` will carry the target offset,
            computed from those
        """
        # Recover targets for semantic segmentation
        output = super().get_target(nag, output)

        # Recover targets for instance/panoptic segmentation
        output.obj_edge_index = getattr(nag[1], 'obj_edge_index', None)
        output.obj_edge_affinity = getattr(nag[1], 'obj_edge_affinity', None)
        output.pos = nag[1].pos
        output.obj_pos = getattr(nag[1], 'obj_pos', None)

        return output

    def model_step(self, batch):
        # Loss and predictions for semantic segmentation
        semantic_loss, output = super().model_step(batch)

        # Cannot compute losses if some target data are missing
        if not output.has_target:
            return None, output

        # TODO: offset soft-assigned to 0 based on the predicted
        #  stuff/thing probas. A stuff/thing classification loss could
        #  provide additional supervision

        # Prepare predicted and target node offsets
        # NB: By convention, we want stuff nodes to have 0 offset. Two
        # reasons for that:
        #   - defining a stuff target center is ambiguous
        #   - by predicting 0 offsets, the corresponding nodes are
        #     likely to be isolated by the superpoint clustering step.
        #     This is what we want, because the predictions will be
        #     merged as a post-processing step, to ensure there is a
        #     most one prediction per batch item for each stuff class
        # For that reason, `self.forward()` forcefully sets to 0 the
        # PREDICTED offsets for nodes PREDICTED to stuff classes.
        # Here, we forcefully set to 0 the TARGET offsets of nodes with
        # TARGET stuff class
        y = output.y_hist[0].argmax(dim=1) \
            if output.multi_stage else output.y_hist.argmax(dim=1)
        stuff_classes = torch.tensor(
            self.stuff_classes, dtype=y.dtype, device=y.device)
        is_stuff = torch.isin(y, stuff_classes)
        node_offset = output.node_offset
        node_offset[is_stuff] = 0

        # We exclude the void nodes from loss computation
        idx = torch.where(~output.void_mask)[0]
        node_offset_pred = output.node_offset_pred[idx]
        node_offset = node_offset[idx]

        # Compute the node offset loss, weighted by the node size
        size = batch[1].y_hist[idx].sum(dim=1)
        node_offset_loss = self.node_offset_criterion(
            node_offset_pred, node_offset, weight=size)

        # Compute the edge affinity loss
        # NB: we pass the edge affinity logits to the criterion and not
        # the actual sigmoid-normalized predictions used for graph
        # clustering. The reason for this is that we expect
        # `edge_affinity_criterion` to be `torch.nn.BCEWithLogitsLoss`
        # NB2: we exclude the edges connecting two void nodes from loss
        # computation
        idx = torch.where(~output.void_edge_mask)[0]
        edge_affinity_loss = self.edge_affinity_criterion(
            output.edge_affinity_logits[idx], output.edge_affinity[idx])

        # Combine the losses together
        loss = semantic_loss \
               + self.hparams.edge_affinity_loss_lambda * edge_affinity_loss \
               + self.hparams.node_offset_loss_lambda * node_offset_loss

        return loss, output

# TODO: metrics for panoptic + metrics to track accuracy / fscore on affinity
#  prediction + metrics to track offset error

# TODO: handle loss in output object (helps using offset loss as metric when
#  logging metrics)

# TODO: make sure void and stuff properly treated for semantic loss, offset loss
#  and affinty loss

# TODO: run, every K epochs, the instance partition + stuff aggregation
#  prost-proc + put the resulting instance pred in the output object +
#  instance/panoptic metrics computation
#  (NB: do we compute on train (ie potential latencies) or only on val ?)

