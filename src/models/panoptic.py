import torch
import logging
from torch_scatter import scatter_mean
from torchmetrics import MinMetric, MaxMetric, MeanMetric
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from src.data import InstanceData
from src.utils import init_weights, get_stuff_mask, scatter_mean_weighted
from src.metrics import MeanAveragePrecision3D, PanopticQuality3D, \
    WeightedMeanSquaredError
from src.models.semantic import SemanticSegmentationOutput, \
    SemanticSegmentationModule
from src.loss import OffsetLoss
from src.nn import FFN, InstancePartitioner

log = logging.getLogger(__name__)


__all__ = ['PanopticSegmentationOutput', 'PanopticSegmentationModule']


class PanopticSegmentationOutput(SemanticSegmentationOutput):
    """A simple holder for panoptic segmentation model output, with a
    few helper methods for manipulating the predictions and targets
    (if any).
    """

    def __init__(
            self,
            logits,
            stuff_classes,
            edge_affinity_logits,
            node_offset_pred,
            node_size,
            y_hist=None,
            obj=None,
            obj_edge_index=None,
            obj_edge_affinity=None,
            pos=None,
            obj_pos=None,
            obj_index_pred=None,
            semantic_loss=None,
            node_offset_loss=None,
            edge_affinity_loss=None):
        # We set the child class attributes before calling the parent
        # class constructor, because the parent constructor calls
        # `self.debug()`, which needs all attributes to be initialized
        device = edge_affinity_logits.device
        self.stuff_classes = torch.tensor(stuff_classes, device=device).long() \
            if stuff_classes is not None \
            else torch.empty(0, device=device).long()
        self.edge_affinity_logits = edge_affinity_logits
        self.node_offset_pred = node_offset_pred
        self.node_size = node_size
        self.obj = obj
        self.obj_edge_index = obj_edge_index
        self.obj_edge_affinity = obj_edge_affinity
        self.pos = pos
        self.obj_pos = obj_pos
        self.obj_index_pred = obj_index_pred
        self.semantic_loss = semantic_loss
        self.node_offset_loss = node_offset_loss
        self.edge_affinity_loss = edge_affinity_loss
        super().__init__(logits, y_hist=y_hist)

    def debug(self):
        # Parent class debugger
        super().debug()

        # Instance predictions
        assert self.node_offset_pred.dim() == 2
        assert self.node_offset_pred.shape[0] == self.num_nodes
        assert self.edge_affinity_logits.dim() == 1

        # Node properties
        assert self.node_size.dim() == 1
        assert self.node_size.shape[0] == self.num_nodes

        if self.has_instance_pred:
            assert self.obj_index_pred.dim() == 1
            assert self.obj_index_pred.shape[0] == self.num_nodes

        # Instance targets
        items = [
            self.obj_edge_index, self.obj_edge_affinity, self.pos, self.obj_pos]
        without_instance_targets = all(x is None for x in items)
        with_instance_targets = all(x is not None for x in items)
        assert without_instance_targets or with_instance_targets

        if without_instance_targets:
            return

        assert isinstance(self.obj, InstanceData)
        assert self.obj.num_clusters == self.num_nodes
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
            self.obj,
            self.obj_edge_index,
            self.obj_edge_affinity,
            self.pos,
            self.obj_pos]
        return super().has_target and all(x is not None for x in items)

    @property
    def has_instance_pred(self):
        """Check whether `self` contains predicted data for panoptic
        segmentation `obj_index_pred`.
        """
        return self.obj_index_pred is not None

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
        return self.edge_affinity_logits.sigmoid()

    @property
    def void_edge_mask(self):
        """Returns a mask on the edges indicating those connecting two
        void nodes.
        """
        if not self.has_target:
            return

        mask = self.void_mask[self.obj_edge_index]
        return mask[0] & mask[1]

    @property
    def sanitized_node_offsets(self):
        """Return the predicted and target node offsets, along with node
        size, sanitized for node offset loss and metrics computation.

        By convention, we want stuff nodes to have 0 offset. Two
        reasons for that:
          - defining a stuff target center is ambiguous
          - by predicting 0 offsets, the corresponding nodes are
            likely to be isolated by the superpoint clustering step.
            This is what we want, because the predictions will be
            merged as a post-processing step, to ensure there is a
            most one prediction per batch item for each stuff class

        Besides, we choose to exclude nodes/superpoints with more than
        50% 'void' points from node offset loss and metrics computation.

        To this end, the present function does the following:
          - ASSUME predicted offsets are 0 when predicted semantic class
            is of type 'stuff'
          - set target offsets to 0 when target semantic class is of
            type 'stuff'
          - remove predicted and target offsets for 'void' nodes (see
            `self.void_mask`)
        """
        if not self.has_target:
            return None, None, None

        # We exclude the void nodes from loss computation
        idx = torch.where(~self.void_mask)[0]

        # Set target offsets to 0 when predicted semantic is stuff
        y_hist = self.targets
        is_stuff = get_stuff_mask(y_hist, self.stuff_classes)
        node_offset = self.node_offset
        node_offset[is_stuff] = 0

        return self.node_offset_pred[idx], node_offset[idx], self.node_size[idx]

    @property
    def sanitized_edge_affinities(self):
        """Return the predicted and target edge affinities, sanitized
        for edge affinity loss and metrics computation.

        We return the edge affinity logits to the criterion and not
        the actual sigmoid-normalized predictions used for graph
        clustering. The reason for this is that we expect the edge
        affinity loss to be computed using `torch.nn.BCEWithLogitsLoss`.

        We choose to exclude edges connecting nodes/superpoints with
        more than 50% 'void' points from edge affinity loss and metrics
        computation.

        To this end, the present function does the following:
          - remove predicted and target edges connecting two 'void'
            nodes (see `self.void_edge_mask`)
        """
        idx = torch.where(~self.void_edge_mask)[0]
        return self.edge_affinity_logits[idx], self.obj_edge_affinity[idx]

    @property
    def instance_predictions(self):
        """Return the predicted InstanceData, and the predicted instance
        semantic label and score.
        """
        if not self.has_instance_pred:
            return None, None, None

        # Merge the InstanceData based on the predicted instances and
        # target instances
        instance_data = self.obj.merge(self.obj_index_pred) if self.has_target \
            else None

        # Compute the mean logits for each predicted object, weighted by
        # the node sizes
        node_logits = self.logits[0] if self.multi_stage else self.logits
        obj_logits = scatter_mean_weighted(
            node_logits, self.obj_index_pred, self.node_size)

        # Compute the predicted semantic label and proba for each node
        obj_semantic_score, obj_y = obj_logits.softmax(dim=1).max(dim=1)

        # Compute the mean node offset, weighted by node sizes, for each
        # object
        node_x = self.pos + self.node_offset_pred
        obj_x = scatter_mean_weighted(
            node_x, self.obj_index_pred, self.node_size)

        # Compute the mean squared distance to the mean predicted offset
        # for each object
        node_x_error = ((node_x - obj_x[self.obj_index_pred])**2).sum(dim=1)
        obj_x_error = scatter_mean_weighted(
            node_x_error, self.obj_index_pred, self.node_size).squeeze()

        # Compute the node offset prediction score
        obj_x_score = 1 / (1 + obj_x_error)

        # Compute, for each predicted object, the mean inter-object and
        # intra-object predicted edge affinity
        ie = self.obj_index_pred[self.obj_edge_index]
        intra = ie[0] == ie[1]
        idx = ie.flatten()
        intra = intra.repeat(2)
        a = self.edge_affinity_preds.repeat(2)
        n = self.obj_index_pred.max() + 1
        obj_mean_intra = scatter_mean(a[intra], idx[intra], dim_size=n)
        obj_mean_inter = scatter_mean(a[~intra], idx[~intra], dim_size=n)

        # Compute the inter-object and intra-object scores
        obj_intra_score = obj_mean_intra
        obj_inter_score = 1 / (1 + obj_mean_inter)

        # Final prediction score is the product of individual scores
        obj_score = \
            obj_semantic_score * obj_x_score * obj_intra_score * obj_inter_score

        return obj_score, obj_y, instance_data


class PanopticSegmentationModule(SemanticSegmentationModule):
    """A LightningModule for panoptic segmentation of point clouds.
    """

    _IGNORED_HYPERPARAMETERS = [
        'net', 'criterion', 'edge_affinity_criterion', 'node_offset_criterion']

    def __init__(
            self,
            net,
            partitioner,
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
            partition_every_n_epochs=10,
            **kwargs):
        super().__init__(
            net,
            criterion,
            optimizer,
            scheduler,
            num_classes,
            class_names=class_names,
            sampling_loss=sampling_loss,
            loss_type=loss_type,
            weighted_loss=weighted_loss,
            init_linear=init_linear,
            init_rpe=init_rpe,
            transformer_lr_scale=transformer_lr_scale,
            multi_stage_loss_lambdas=multi_stage_loss_lambdas,
            gc_every_n_steps=gc_every_n_steps,
            **kwargs)

        # Instance partition head, epects a fully-fledged
        # InstancePartitioner module as input.
        # This module is only called when the actual instance/panoptic
        # segmentation is required. At train time, it is not essential,
        # since we do not propagate gradient to its parameters. However,
        # we still tune its parameters to maximize instance/panoptic
        # metrics on the train set. This tuning involves a simple
        # grid-search on a small range of parameters and needs to be
        # called at least once at the very end of training
        self.partition_every_n_epochs = partition_every_n_epochs
        self.partitioner = partitioner

        # Store the stuff class indices
        self.stuff_classes = stuff_classes

        # Loss functions for edge affinity and node offset predictions.
        # NB: the semantic loss is already accounted for in the
        # SemanticSegmentationModule constructor
        self.edge_affinity_criterion = torch.nn.BCEWithLogitsLoss() \
            if edge_affinity_criterion is None else edge_affinity_criterion
        self.node_offset_criterion = OffsetLoss() \
            if node_offset_criterion is None else node_offset_criterion

        # Model heads for edge affinity and node offset predictions
        # Initialize the model segmentation head (or heads)
        out_dim = self.net.out_dim[0] if self.multi_stage_loss \
            else self.net.out_dim
        self.edge_affinity_head = FFN(out_dim * 2, hidden_dim=32, out_dim=1)
        self.node_offset_head = FFN(out_dim, hidden_dim=32, out_dim=3)

        # Custom weight initialization. In particular, this applies
        # Xavier / Glorot initialization on Linear and RPE layers by
        # default, but can be tuned
        init = lambda m: init_weights(m, linear=init_linear, rpe=init_rpe)
        self.edge_affinity_head.apply(init)
        self.node_offset_head.apply(init)

        # Metric objects for calculating instance segmentation scores on
        # each dataset split
        self.train_instance = MeanAveragePrecision3D(
            self.num_classes,
            stuff_classes=self.stuff_classes,
            min_size=min_instance_size,
            compute_on_cpu=True,
            remove_void=True,
            **kwargs)
        self.val_instance = MeanAveragePrecision3D(
            self.num_classes,
            stuff_classes=self.stuff_classes,
            min_size=min_instance_size,
            compute_on_cpu=True,
            remove_void=True,
            **kwargs)
        self.test_instance = MeanAveragePrecision3D(
            self.num_classes,
            stuff_classes=self.stuff_classes,
            min_size=min_instance_size,
            compute_on_cpu=True,
            remove_void=True,
            **kwargs)

        # Metric objects for calculating panoptic segmentation scores on
        # each dataset split
        self.train_panoptic = PanopticQuality3D(
            self.num_classes,
            ignore_unseen_classes=True,
            stuff_classes=self.stuff_classes,
            compute_on_cpu=True,
            **kwargs)
        self.val_panoptic = PanopticQuality3D(
            self.num_classes,
            ignore_unseen_classes=True,
            stuff_classes=self.stuff_classes,
            compute_on_cpu=True,
            **kwargs)
        self.test_panoptic = PanopticQuality3D(
            self.num_classes,
            ignore_unseen_classes=True,
            stuff_classes=self.stuff_classes,
            compute_on_cpu=True,
            **kwargs)

        # Metric objects for calculating node offset prediction scores
        # on each dataset split
        self.train_offset_mse = WeightedMeanSquaredError()
        self.val_offset_mse = WeightedMeanSquaredError()
        self.test_offset_mse = WeightedMeanSquaredError()

        # Metric objects for calculating edge affinity prediction scores
        # on each dataset split
        self.train_affinity_oa = BinaryAccuracy()
        self.train_affinity_f1 = BinaryF1Score()
        self.val_affinity_oa = BinaryAccuracy()
        self.val_affinity_f1 = BinaryF1Score()
        self.test_affinity_oa = BinaryAccuracy()
        self.test_affinity_f1 = BinaryF1Score()

        # For averaging losses across batches
        self.train_semantic_loss = MeanMetric()
        self.train_edge_affinity_loss = MeanMetric()
        self.train_node_offset_loss = MeanMetric()
        self.val_semantic_loss = MeanMetric()
        self.val_edge_affinity_loss = MeanMetric()
        self.val_node_offset_loss = MeanMetric()
        self.test_semantic_loss = MeanMetric()
        self.test_edge_affinity_loss = MeanMetric()
        self.test_node_offset_loss = MeanMetric()

        # For tracking best-so-far validation metrics
        self.val_map_best = MaxMetric()
        self.val_pq_best = MaxMetric()
        self.val_pqmod_best = MaxMetric()
        self.val_mprec_best = MaxMetric()
        self.val_mrec_best = MaxMetric()
        self.val_offset_mse_best = MinMetric()
        self.val_affinity_oa_best = MaxMetric()
        self.val_affinity_f1_best = MaxMetric()

    @property
    def needs_partition(self):
        """Whether the `self.partitioner` should be called to compute
        the actual panoptic segmentation. During training, the actual
        partition is not really needed, as we do not learn to partition,
        but learn to predict inputs for the partition step instead. For
        this reason, we save compute and time during training by only
        computing the partition once in a while with
        `self.partition_every_n_epochs`.
        """
        # nth_epoch = self.current_epoch % self.partition_every_n_epochs == 0 \
        #     if self.partition_every_n_epochs > 0 and self.current_epoch > 0 \
        #     else False
        nth_epoch = self.current_epoch % self.partition_every_n_epochs == 0 \
            if self.partition_every_n_epochs > 0 else False
        last_epoch = self.current_epoch == self.trainer.max_epochs - 1
        return nth_epoch or last_epoch

    def forward(self, nag) -> PanopticSegmentationOutput:
        # Extract features
        x = self.net(nag)

        # Compute level-1 or multi-level semantic predictions
        semantic_pred = [head(x_) for head, x_ in zip(self.head, x)] \
            if self.multi_stage_loss else self.head(x)

        # Recover level-1 features only
        x = x[0] if self.multi_stage_loss else x

        # Compute node offset predictions
        node_offset_pred = self.node_offset_head(x)

        # Forcefully set 0-offset for nodes with stuff predictions
        node_logits = semantic_pred[0] if self.multi_stage_loss \
            else semantic_pred
        is_stuff = get_stuff_mask(node_logits, self.stuff_classes)
        node_offset_pred[is_stuff] = 0

        # TODO: offset soft-assigned to 0 based on the predicted
        #  stuff/thing probas. A stuff/thing classification loss could
        #  provide additional supervision

        # Compute edge affinity predictions
        # NB: we make edge features symmetric, since we want to compute
        # edge affinity, which is not directed
        x_edge = x[nag[1].obj_edge_index]
        x_edge = torch.cat(
            ((x_edge[0] - x_edge[1]).abs(), (x_edge[0] + x_edge[1]) / 2), dim=1)
        edge_affinity_logits = self.edge_affinity_head(x_edge).squeeze()

        # Gather results in an output object
        output = PanopticSegmentationOutput(
            semantic_pred,
            self.stuff_classes,
            edge_affinity_logits,
            node_offset_pred,
            nag.get_sub_size(1))

        # Compute the panoptic partition
        output = self._forward_partition(nag, output)

        return output

    def _forward_partition(self, nag, output) -> PanopticSegmentationOutput:
        """Compute the panoptic partition based on the predicted node
        offsets, node semantic logits, and edge affinity logits.

        The partition will only be computed if required. In general,
        during training, the actual partition is not needed for the
        model to be supervised. We only run it once in a while to
        evaluate the panoptic/instance segmentation metrics or tune
        the partition hyperparameters on the train set.

        :param nag: NAG object
        :param output: PanopticSegmentationOutput

        :return: output
        """
        if not self.needs_partition:
            return output

        # Recover some useful information from the NAG and
        # PanopticSegmentationOutput objects
        batch = nag[1].batch
        node_x = nag[1].pos + output.node_offset_pred
        node_size = nag.get_sub_size(1)
        node_logits = output.logits[0] if output.multi_stage else output.logits
        node_is_stuff = get_stuff_mask(node_logits, self.stuff_classes)
        edge_index = nag[1].obj_edge_index
        edge_affinity_logits = output.edge_affinity_logits

        # Compute the instance partition
        # NB: we detach the tensors here: this operation runs on CPU and
        # is non-differentiable
        obj_index = self.partitioner(
            batch,
            node_x.detach(),
            node_logits.detach(),
            node_is_stuff.detach(),
            node_size,
            edge_index,
            edge_affinity_logits.detach())

        # Store the results in the output object
        output.obj_index_pred = obj_index

        return output

    def on_fit_start(self):
        super().on_fit_start()

        # Get the LightningDataModule stuff classes and make sure it
        # matches self.stuff_classes. We could also forcefully update
        # the LightningModule with this new information, but it could
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
        self.val_instance.reset()
        self.val_panoptic.reset()
        self.val_offset_mse.reset()
        self.val_affinity_oa.reset()
        self.val_affinity_f1.reset()
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
        node_size = nag.get_sub_size(1)

        return PanopticSegmentationOutput(
            output_semseg.logits,
            self.stuff_classes,
            edge_affinity_logits,
            node_offset_pred,
            node_size)

    @staticmethod
    def _update_output_multi(output_multi, nag, output, nag_transformed, key):
        """Local helper method to accumulate multiple predictions on
        the same -or part of the same- point cloud.
        """
        raise NotImplementedError(
            "The current implementation does not properly support multi-run "
            "for instance/panoptic segmentation")

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
        output.obj = nag[1].obj

        return output

    def model_step(self, batch):
        # Loss and predictions for semantic segmentation
        semantic_loss, output = super().model_step(batch)

        # Cannot compute losses if some target data are missing
        if not output.has_target:
            return None, output

        # Compute the node offset loss, weighted by the node size
        node_offset_loss = self.node_offset_criterion(
            *output.sanitized_node_offsets)

        # Compute the edge affinity loss
        edge_affinity_loss = self.edge_affinity_criterion(
            *output.sanitized_edge_affinities)

        # Combine the losses together
        loss = semantic_loss \
               + self.hparams.edge_affinity_loss_lambda * edge_affinity_loss \
               + self.hparams.node_offset_loss_lambda * node_offset_loss

        # Save individual losses in the output object
        output.semantic_loss = semantic_loss
        output.node_offset_loss = node_offset_loss
        output.edge_affinity_loss = edge_affinity_loss

        return loss, output

    def train_step_update_metrics(self, loss, output: PanopticSegmentationOutput):
        """Update train metrics with the content of the output object.
        """
        # Update semantic segmentation metrics
        super().train_step_update_metrics(loss, output)

        # Update instance and panoptic metrics
        if self.needs_partition:
            obj_score, obj_y, instance_data = output.instance_predictions
            self.train_instance.update(obj_score, obj_y, instance_data)
            self.train_panoptic.update(obj_y, instance_data)

        # Update tracked losses
        self.train_semantic_loss(output.semantic_loss.detach().cpu())
        self.train_node_offset_loss(output.node_offset_loss.detach().cpu())
        self.train_edge_affinity_loss(output.edge_affinity_loss.detach().cpu())

        # Update node offset metrics
        self.train_offset_mse(
            *[o.detach().cpu() for o in output.sanitized_node_offsets])

        # Update edge affinity metrics
        ea = [o.detach().cpu() for o in output.sanitized_edge_affinities]
        self.train_affinity_oa(ea[0], (ea[1] > 0.5).long())
        self.train_affinity_f1(ea[0], (ea[1] > 0.5).long())

    def train_step_log_metrics(self):
        """Log train metrics after a single step with the content of the
        output object.
        """
        super().train_step_log_metrics()
        self.log(
            "train/semantic_loss", self.train_semantic_loss, on_step=False,
            on_epoch=True, prog_bar=True)
        self.log(
            "train/node_offset_loss", self.train_node_offset_loss, on_step=False,
            on_epoch=True, prog_bar=True)
        self.log(
            "train/edge_affinity_loss", self.train_edge_affinity_loss, on_step=False,
            on_epoch=True, prog_bar=True)

    def on_train_epoch_end(self):
        # Log semantic segmentation metrics and reset confusion matrix
        super().on_train_epoch_end()

        if self.needs_partition:
            # Compute the instance and panoptic metrics
            instance_results = self.train_instance.compute()
            panoptic_results = self.train_panoptic.compute()

            # Gather tracked metrics
            map = instance_results.map
            map_50 = instance_results.map_50
            map_75 = instance_results.map_75
            map_per_class = instance_results.map_per_class
            pq = panoptic_results.pq
            sq = panoptic_results.sq
            rq = panoptic_results.rq
            pq_thing = panoptic_results.pq_thing
            pq_stuff = panoptic_results.pq_stuff
            pqmod = panoptic_results.pq_modified
            mprec = panoptic_results.mean_precision
            mrec = panoptic_results.mean_recall
            pq_per_class = panoptic_results.pq_per_class

            # Log metrics
            self.log("train/map", map, prog_bar=True)
            self.log("train/map_50", map_50, prog_bar=True)
            self.log("train/map_75", map_75, prog_bar=True)
            self.log("train/pq", pq, prog_bar=True)
            self.log("train/sq", sq, prog_bar=True)
            self.log("train/rq", rq, prog_bar=True)
            self.log("train/pq_thing", pq_thing, prog_bar=True)
            self.log("train/pq_stuff", pq_stuff, prog_bar=True)
            self.log("train/pqmod", pqmod, prog_bar=True)
            self.log("train/mprec", mprec, prog_bar=True)
            self.log("train/mrec", mrec, prog_bar=True)
            enum = zip(map_per_class, pq_per_class, self.class_names)
            for map_c, pq_c, name in enum:
                self.log(f"train/map_{name}", map_c, prog_bar=True)
                self.log(f"train/pq_{name}", pq_c, prog_bar=True)

        # Log metrics
        self.log("train/offset_mse", self.train_offset_mse.compute(), prog_bar=True)
        self.log("train/affinity_oa", self.train_affinity_oa.compute(), prog_bar=True)
        self.log("train/affinity_f1", self.train_affinity_f1.compute(), prog_bar=True)

        # Reset metrics accumulated over the last epoch
        self.train_offset_mse.reset()
        self.train_affinity_oa.reset()
        self.train_affinity_f1.reset()

    def validation_step_update_metrics(self, loss, output):
        """Update validation metrics with the content of the output
        object.
        """
        # Update semantic segmentation metrics
        super().validation_step_update_metrics(loss, output)

        # Update instance and panoptic metrics
        if self.needs_partition:
            obj_score, obj_y, instance_data = output.instance_predictions
            self.val_instance.update(obj_score, obj_y, instance_data)
            self.val_panoptic.update(obj_y, instance_data)

        # Update tracked losses
        self.val_semantic_loss(output.semantic_loss.detach().cpu())
        self.val_node_offset_loss(output.node_offset_loss.detach().cpu())
        self.val_edge_affinity_loss(output.edge_affinity_loss.detach().cpu())

        # Update node offset metrics
        self.val_offset_mse(
            *[o.detach().cpu() for o in output.sanitized_node_offsets])

        # Update edge affinity metrics
        ea = [o.detach().cpu() for o in output.sanitized_edge_affinities]
        self.val_affinity_oa(ea[0], (ea[1] > 0.5).long())
        self.val_affinity_f1(ea[0], (ea[1] > 0.5).long())

    def validation_step_log_metrics(self):
        """Log validation metrics after a single step with the content
        of the output object.
        """
        super().validation_step_log_metrics()
        self.log(
            "val/semantic_loss", self.val_semantic_loss, on_step=False,
            on_epoch=True, prog_bar=True)
        self.log(
            "val/node_offset_loss", self.val_node_offset_loss, on_step=False,
            on_epoch=True, prog_bar=True)
        self.log(
            "val/edge_affinity_loss", self.val_edge_affinity_loss, on_step=False,
            on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        # Log semantic segmentation metrics and reset confusion matrix
        super().on_validation_epoch_end()

        if self.needs_partition:
            # Compute the instance and panoptic metrics
            instance_results = self.val_instance.compute()
            panoptic_results = self.val_panoptic.compute()

            # Gather tracked metrics
            map = instance_results.map
            map_50 = instance_results.map_50
            map_75 = instance_results.map_75
            map_per_class = instance_results.map_per_class
            pq = panoptic_results.pq
            sq = panoptic_results.sq
            rq = panoptic_results.rq
            pq_thing = panoptic_results.pq_thing
            pq_stuff = panoptic_results.pq_stuff
            pqmod = panoptic_results.pq_modified
            mprec = panoptic_results.mean_precision
            mrec = panoptic_results.mean_recall
            pq_per_class = panoptic_results.pq_per_class

            # Log metrics
            self.log("val/map", map, prog_bar=True)
            self.log("val/map_50", map_50, prog_bar=True)
            self.log("val/map_75", map_75, prog_bar=True)
            self.log("val/pq", pq, prog_bar=True)
            self.log("val/sq", sq, prog_bar=True)
            self.log("val/rq", rq, prog_bar=True)
            self.log("val/pq_thing", pq_thing, prog_bar=True)
            self.log("val/pq_stuff", pq_stuff, prog_bar=True)
            self.log("val/pqmod", pqmod, prog_bar=True)
            self.log("val/mprec", mprec, prog_bar=True)
            self.log("val/mrec", mrec, prog_bar=True)
            enum = zip(map_per_class, pq_per_class, self.class_names)
            for map_c, pq_c, name in enum:
                self.log(f"val/map_{name}", map_c, prog_bar=True)
                self.log(f"val/pq_{name}", pq_c, prog_bar=True)

            # Update best-so-far metrics
            self.val_map_best(map)
            self.val_pq_best(pq)
            self.val_pqmod_best(pqmod)
            self.val_mprec_best(mprec)
            self.val_mrec_best(mrec)

        # Compute the metrics tracked for model selection on validation
        offset_mse = self.val_offset_mse.compute()
        affinity_oa = self.val_affinity_oa.compute()
        affinity_f1 = self.val_affinity_f1.compute()

        # Log metrics
        self.log("val/offset_mse", offset_mse, prog_bar=True)
        self.log("val/affinity_oa", affinity_oa, prog_bar=True)
        self.log("val/affinity_f1", affinity_f1, prog_bar=True)

        # Update best-so-far metrics
        self.val_offset_mse_best(offset_mse)
        self.val_affinity_oa_best(affinity_oa)
        self.val_affinity_f1_best(affinity_f1)

        # Log best-so-far metrics, using `.compute()` instead of passing
        # the whole torchmetrics object, because otherwise metric would
        # be reset by lightning after each epoch
        self.log("val/offset_mse_best", self.val_offset_mse_best.compute(), prog_bar=True)
        self.log("val/affinity_oa_best", self.val_affinity_oa_best.compute(), prog_bar=True)
        self.log("val/affinity_f1_best", self.val_affinity_f1_best.compute(), prog_bar=True)

        # Reset metrics accumulated over the last epoch
        self.val_offset_mse.reset()
        self.val_affinity_oa.reset()
        self.val_affinity_f1.reset()

    def test_step_update_metrics(self, loss, output):
        """Update test metrics with the content of the output object.
        """
        if not output.has_target:
            return

        # Update semantic segmentation metrics
        super().test_step_update_metrics(loss, output)

        # Update instance and panoptic metrics
        if self.needs_partition:
            obj_score, obj_y, instance_data = output.instance_predictions
            self.test_instance.update(obj_score, obj_y, instance_data)
            self.test_panoptic.update(obj_y, instance_data)

        # Update tracked losses
        self.test_semantic_loss(output.semantic_loss.detach().cpu())
        self.test_node_offset_loss(output.node_offset_loss.detach().cpu())
        self.test_edge_affinity_loss(output.edge_affinity_loss.detach().cpu())

        # Update node offset metrics
        self.test_offset_mse(
            *[o.detach().cpu() for o in output.sanitized_node_offsets])

        # Update edge affinity metrics
        ea = [o.detach().cpu() for o in output.sanitized_edge_affinities]
        self.test_affinity_oa(ea[0], (ea[1] > 0.5).long())
        self.test_affinity_f1(ea[0], (ea[1] > 0.5).long())

    def test_step_log_metrics(self):
        """Log test metrics after a single step with the content of the
        output object.
        """
        super().test_step_log_metrics()
        self.log(
            "test/semantic_loss", self.test_semantic_loss, on_step=False,
            on_epoch=True, prog_bar=True)
        self.log(
            "test/node_offset_loss", self.test_node_offset_loss, on_step=False,
            on_epoch=True, prog_bar=True)
        self.log(
            "test/edge_affinity_loss", self.test_edge_affinity_loss, on_step=False,
            on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        # Log semantic segmentation metrics and reset confusion matrix
        super().on_test_epoch_end()

        if self.needs_partition:
            # Compute the instance and panoptic metrics
            instance_results = self.test_instance.compute()
            panoptic_results = self.test_panoptic.compute()

            # Gather tracked metrics
            map = instance_results.map
            map_50 = instance_results.map_50
            map_75 = instance_results.map_75
            map_per_class = instance_results.map_per_class
            pq = panoptic_results.pq
            sq = panoptic_results.sq
            rq = panoptic_results.rq
            pq_thing = panoptic_results.pq_thing
            pq_stuff = panoptic_results.pq_stuff
            pqmod = panoptic_results.pq_modified
            mprec = panoptic_results.mean_precision
            mrec = panoptic_results.mean_recall
            pq_per_class = panoptic_results.pq_per_class

            # Log metrics
            self.log("test/map", map, prog_bar=True)
            self.log("test/map_50", map_50, prog_bar=True)
            self.log("test/map_75", map_75, prog_bar=True)
            self.log("test/pq", pq, prog_bar=True)
            self.log("test/sq", sq, prog_bar=True)
            self.log("test/rq", rq, prog_bar=True)
            self.log("test/pq_thing", pq_thing, prog_bar=True)
            self.log("test/pq_stuff", pq_stuff, prog_bar=True)
            self.log("test/pqmod", pqmod, prog_bar=True)
            self.log("test/mprec", mprec, prog_bar=True)
            self.log("test/mrec", mrec, prog_bar=True)
            enum = zip(map_per_class, pq_per_class, self.class_names)
            for map_c, pq_c, name in enum:
                self.log(f"test/map_{name}", map_c, prog_bar=True)
                self.log(f"test/pq_{name}", pq_c, prog_bar=True)

        # Log metrics
        self.log("test/offset_mse", self.test_offset_mse.compute(), prog_bar=True)
        self.log("test/affinity_oa", self.test_affinity_oa.compute(), prog_bar=True)
        self.log("test/affinity_f1", self.test_affinity_f1.compute(), prog_bar=True)

        # Reset metrics accumulated over the last epoch
        self.test_offset_mse.reset()
        self.test_affinity_oa.reset()
        self.test_affinity_f1.reset()

# TODO: gridsearch instance partition parameters

if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = str(pyrootutils.setup_root(__file__, pythonpath=True))
    cfg = omegaconf.OmegaConf.load(root + "/configs/model/panoptic/spt-2.yaml")
    _ = hydra.utils.instantiate(cfg)
