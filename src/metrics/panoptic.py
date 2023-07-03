import torch
import logging
from torch import Tensor, LongTensor
from typing import Any, Dict, List, Optional, Sequence, Tuple
from torchmetrics.metric import Metric
from torchmetrics.detection.mean_ap import BaseMetricResults
from torch_geometric.nn.pool.consecutive import consecutive_cluster

from src.data import InstanceData, InstanceBatch
from src.utils import arange_interleave, sizes_to_pointers


log = logging.getLogger(__name__)


__all__ = ['PanopticQuality3D']


class PanopticMetricResults(BaseMetricResults):
    """Class to wrap the final metric results for Panoptic Segmentation.
    """
    __slots__ = (
        "pq",
        "pq_thing",
        "pq_stuff",
        "pq_per_class",
        "sq",
        "sq_thing",
        "sq_stuff",
        "sq_per_class",
        "rq",
        "rq_thing",
        "rq_stuff",
        "rq_per_class",
        "pq_modified",
        "pq_modified_thing",
        "pq_modified_stuff",
        "pq_modified_per_class",
        "sq_modified",
        "sq_modified_thing",
        "sq_modified_stuff",
        "sq_modified_per_class",
        "rq_modified",
        "rq_modified_thing",
        "rq_modified_per_class",
        "mean_precision",
        "mean_recall")


class PanopticQuality3D(Metric):
    """Computes the `Panoptic Quality (PQ) and associated metrics`_ for
    3D panoptic segmentation.
    Optionally, the metrics can be calculated per class.

    Importantly, this implementation expects predictions and targets to
    be passed as InstanceData, which assumes predictions and targets
    form two PARTITIONS of the scene: all points belong to one and only
    one prediction and one and only one target ('stuff' included).

    Predicted instances and targets have to be passed to
    :meth:``forward`` or :meth:``update`` within a custom format. See
    the :meth:`update` method for more information about the input
    format to this metric.

    As output of ``forward`` and ``compute`` the metric returns the
    following output:

    - ``pq_dict``: A dictionary containing the following key-values:

        - pq: (:class:`~torch.Tensor`)
        - pq_thing: (:class:`~torch.Tensor`)
        - pq_stuff: (:class:`~torch.Tensor`)
        - pq_per_class: (:class:`~torch.Tensor`)
        - sq: (:class:`~torch.Tensor`)
        - sq_thing: (:class:`~torch.Tensor`)
        - sq_stuff: (:class:`~torch.Tensor`)
        - sq_per_class: (:class:`~torch.Tensor`)
        - rq: (:class:`~torch.Tensor`)
        - rq_thing: (:class:`~torch.Tensor`)
        - rq_stuff: (:class:`~torch.Tensor`)
        - rq_per_class: (:class:`~torch.Tensor`)
        - pq_modified: (:class:`~torch.Tensor`)
        - pq_modified_thing: (:class:`~torch.Tensor`)
        - pq_modified_stuff: (:class:`~torch.Tensor`)
        - pq_modified_per_class: (:class:`~torch.Tensor`)
        - sq_modified: (:class:`~torch.Tensor`)
        - sq_modified_thing: (:class:`~torch.Tensor`)
        - sq_modified_stuff: (:class:`~torch.Tensor`)
        - sq_modified_per_class: (:class:`~torch.Tensor`)
        - rq_modified: (:class:`~torch.Tensor`)
        - rq_modified_thing: (:class:`~torch.Tensor`)
        - rq_modified_per_class: (:class:`~torch.Tensor`)
        - mean_precision: (:class:`~torch.Tensor`)
        - mean_recall: (:class:`~torch.Tensor`)

    :param **************
    """
    prediction_semantic: List[LongTensor]
    instance_data: List[InstanceData]

    def __init__(
            self,
            class_metrics: bool = False,
            stuff_classes: Optional[List[int]] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # TODO: deal with ignored classes
        # TODO: parallelize per-class computation with starmap

        # If class_metrics=True the per-class metrics will also be
        # returned, although this may impact overall speed
        if not isinstance(class_metrics, bool):
            raise ValueError("Expected argument `class_metrics` to be a bool")
        self.class_metrics = class_metrics

        # Stuff classes may be specified, to be properly accounted for
        # in metrics computation
        self.stuff_classes = stuff_classes

        # All torchmetric's Metrics have internal states they use to
        # store predictions and ground truths. Those are updated when
        # `self.forward()` or `self.update()` are called, and used for
        # computing the actual metrics when `self.compute()` is called.
        # Every time we want to restart our metric measurements, we
        # need to reset these internal states to their initial values.
        # This happens when calling `self.reset()`. For `self.reset()`
        # to know what to reset and to which value, these states to be
        # declared with `self.add_state()`
        self.add_state("prediction_semantic", default=[], dist_reduce_fx=None)
        self.add_state("instance_data", default=[], dist_reduce_fx=None)

    def update(
            self,
            prediction_semantic: LongTensor,
            instance_data: InstanceData,
    ) -> None:
        """Update the internal state of the metric.

        :param prediction_semantic: LongTensor
             1D tensor of size N_pred holding the semantic label of the
             predicted instances.
        :param instance_data: InstanceData
             InstanceData object holding all information required for
             computing the iou between predicted and ground truth
             instances, as well as the target semantic label.
             Importantly, ALL PREDICTION AND TARGET INSTANCES ARE
             ASSUMED TO BE REPRESENTED in THIS InstanceData, even
             'stuff' classes and 'too small' instances, which will be
             accounted for in this metric. Besides the InstanceData
             assumes the predictions and targets form two PARTITIONS of
             the scene: all points belong to one and only one prediction
             and one and only one target object ('stuff' included).
        :return:
        """
        # Sanity checks
        self._input_validator(prediction_semantic, instance_data)

        # Store in the internal states
        self.prediction_semantic.append(prediction_semantic)
        self.instance_data.append(instance_data)

    @staticmethod
    def _input_validator(
            prediction_semantic: LongTensor,
            instance_data: InstanceData):
        """Sanity checks executed on the input of `self.update()`.
        """
        if not isinstance(prediction_semantic, Tensor):
            raise ValueError(
                "Expected argument `prediction_semantic` to be of type Tensor")
        if not prediction_semantic.dtype == torch.long:
            raise ValueError(
                "Expected argument `prediction_semantic` to have dtype=long")
        if not isinstance(instance_data, InstanceData):
            raise ValueError(
                "Expected argument `instance_data` to be of type InstanceData")

        if prediction_semantic.dim() != 1:
            raise ValueError(
                "Expected argument `prediction_semantic` to have dim=1")
        if prediction_semantic.numel() != instance_data.num_clusters:
            raise ValueError(
                "Expected argument `prediction_semantic` and `instance_data` to "
                "have the same number of size")

    def compute(self) -> dict:
        """Metrics computation.
        """

        # Batch together the values stored in the internal states.
        # Importantly, the InstanceBatch mechanism ensures there is no
        # collision between object labels of the stored scenes
        pred_semantic = torch.cat(self.prediction_semantic)
        pair_data = InstanceBatch.from_list(self.instance_data)
        device = pred_semantic.device

        # Recover the target index, IoU, sizes, and target label for
        # each pair. Importantly, the way the pair_pred_idx is built
        # guarantees the prediction indices are contiguous in
        # [0, pred_id_max]. Besides it is IMPORTANT that all points are
        # accounted for in the InstanceData, regardless of their class.
        # This is because the InstanceData will infer the total size
        # of each segment, as well as the IoUs from these values.
        pair_pred_idx = pair_data.indices
        pair_gt_idx = pair_data.obj
        pair_gt_semantic = pair_data.y
        pair_iou = pair_data.iou_and_size()[0]

        # To alleviate memory and compute, we would rather store
        # ground-truth-specific attributes in tensors of size
        # num_gt rather than for all prediction-ground truth
        # pairs. We will keep track of the prediction and ground truth
        # indices for each pair, to be able to recover relevant
        # information when need be. To this end, since there is no
        # guarantee the ground truth indices are contiguous in
        # [0, gt_idx_max], we contract those and gather associated
        # pre-ground-truth attributes
        pair_gt_idx, gt_idx = consecutive_cluster(pair_gt_idx)
        gt_semantic = pair_gt_semantic[gt_idx]
        del gt_idx, pair_gt_semantic

        # Recover the classes of interest for this metric. These are the
        # classes whose labels appear at least once in the predicted
        # semantic label or in the ground truth semantic labels.
        # Besides, if `stuff_classes` was provided, the corresponding
        # labels are ignored
        class_ids = self._get_classes()  # TODO: this should NOT remove the STUFF clases for panoptic...

        # For each class, each size range (and each IoU threshold),
        # compute the prediction-ground truth matches. Importantly, the
        # output of this step is formatted so as to be passed to
        # `MeanAveragePrecision.__calculate_recall_precision_scores`
        evaluations = [
            self._evaluate(
                class_id,
                pair_iou,
                pair_pred_idx,
                pair_gt_idx,
                pred_semantic,
                gt_semantic)
            for class_id in class_ids]

        num_iou = len(self.iou_thresholds)
        num_rec = len(self.rec_thresholds)
        num_classes = len(class_ids)
        num_sizes = len(self.size_ranges)
        precision = -torch.ones((num_iou, num_rec, num_classes, num_sizes, 1), device=device)
        recall = -torch.ones((num_iou, num_classes, num_sizes, 1), device=device)
        scores = -torch.ones((num_iou, num_rec, num_classes, num_sizes, 1), device=device)

        # Compute the recall, precision, and score for each IoU
        # threshold, class, and size range
        for idx_cls, _ in enumerate(class_ids):
            for i_size, _ in enumerate(self.size_ranges):
                recall, precision, scores = self.__calculate(
                    recall,
                    precision,
                    scores,
                    idx_cls,
                    i_size,
                    evaluations)

        # Compute all AP and AR metrics
        metrics

        # If class_metrics is enabled, also evaluate all metrics for
        # each class of interest
        pass

        return metrics

    def _evaluate(
            self,
            class_id: int,
            size_range: Tuple[int, int],
            pred_score: Tensor,
            pair_iou: Tensor,
            pair_pred_idx: Tensor,
            pair_gt_idx: Tensor,
            pred_semantic: Tensor,
            gt_semantic: Tensor,
            pred_size: Tensor,
            gt_size: Tensor
    ) -> Optional[dict]:
        """Perform evaluation for single class and a single size range.
        The output evaluations cover all required IoU thresholds.
        Concretely, these "evaluations" are the prediction-target
        assignments, with respect to constraints on the semantic class
        of interest, the IoU threshold (for AP computation), the target
        size, etc.

        The following rules apply:
          - at most 1 prediction per target
          - predictions are assigned by order of decreasing score and to
            the not-already-matched target with highest IoU (within IoU
            threshold)

        NB: the input prediction-wise and pair-wise data is assumed to
        be ALREADY SORTED by descending prediction scores.

        The output is formatted so as to be passed to torchmetrics'
        `MeanAveragePrecision.__calculate_recall_precision_scores`.

        :param class_id: int
            Index of the class on which to compute the evaluations.
        :param size_range: List
            Upper and lower bounds for the size range of interest.
            Target objects outside those bounds will be ignored. As well
            as non-matched predictions outside those bounds.
        :param pred_score: Tensor of shape [N_pred]
        :param pair_iou: Tensor of shape [N_pred_gt_overlaps]
        :param pair_pred_idx: Tensor of shape [N_pred_gt_overlaps]
        :param pair_gt_idx: Tensor of shape [N_pred_gt_overlaps]
        :param pred_semantic: Tensor of shape [N_pred]
        :param gt_semantic: Tensor of shape [N_gt]
        :param pred_size: Tensor of shape [N_pred]
        :param gt_size: Tensor of shape [N_gt]
        :return:
        """
        device = gt_semantic.device

        # Compute masks on the prediction-target pairs, based on their
        # semantic label, as well as their size
        is_gt_class = gt_semantic == class_id
        is_pred_class = pred_semantic == class_id
        is_gt_in_size_range = (size_range[0] <= gt_size.float()) \
                              & (gt_size.float() <= size_range[1])
        is_pred_in_size_range = (size_range[0] <= pred_size.float()) \
                                & (pred_size.float() <= size_range[1])

        # Count the number of ground truths and predictions with the
        # class at hand
        num_gt = is_gt_class.count_nonzero().item()
        num_pred = is_pred_class.count_nonzero().item()

        # Get the number of IoU thresholds
        num_iou = len(self.iou_thresholds)

        # If no ground truth and no detection carry the class of
        # instance, return None
        if num_gt == 0 and num_pred == 0:
            return None

        # Some targets have the class at hand but no prediction does
        if num_pred == 0:
            return {
                "dtMatches": torch.zeros(
                    num_iou, 0, dtype=torch.bool, device=device),
                "gtMatches": torch.zeros(
                    num_iou, num_gt, dtype=torch.bool, device=device),
                "dtScores": torch.zeros(0, device=device),
                "gtIgnore": ~is_gt_in_size_range[is_gt_class],
                "dtIgnore": torch.zeros(
                    num_iou, 0, dtype=torch.bool, device=device)}

        # Some predictions have the class at hand but no target does
        if num_gt == 0:
            pred_ignore = ~is_pred_in_size_range[is_pred_class]
            return {
                "dtMatches": torch.zeros(
                    num_iou, num_pred, dtype=torch.bool, device=device),
                "gtMatches": torch.zeros(
                    num_iou, 0, dtype=torch.bool, device=device),
                "dtScores": pred_score[is_pred_class],
                "gtIgnore": torch.zeros(
                    0, dtype=torch.bool, device=device),
                "dtIgnore": pred_ignore.view(1, -1).repeat(num_iou, 1)}

        # Compute the global indices of the prediction and ground truth
        # for the class at hand. These will be used to search for
        # relevant pairs
        gt_class_idx = torch.where(is_gt_class)[0]
        pred_class_idx = torch.where(is_pred_class)[0]
        is_pair_gt_class = torch.isin(pair_gt_idx, gt_class_idx)

        # Build the tensors used to track which ground truth and which
        # prediction has found a match, for each IoU threshold. This is
        # the data structure expected by torchmetrics'
        # `MeanAveragePrecision.__calculate_recall_precision_scores()`
        gt_matches = torch.zeros(
            num_iou, num_gt, dtype=torch.bool, device=device)
        det_matches = torch.zeros(
            num_iou, num_pred, dtype=torch.bool, device=device)
        gt_ignore = ~is_gt_in_size_range[is_gt_class]
        det_ignore = torch.zeros(
            num_iou, num_pred, dtype=torch.bool, device=device)

        # Each pair is associated with a prediction index and a ground
        # truth index. Except these indices are global, spanning across
        # all objects in the data, regardless of their class. Here, we
        # are also going to need to link a pair with the local ground
        # truth index (tracking objects with the current class of
        # interest), to be able to index and update the above-created
        # gt_matches and gt_ignore. To this end, we will compute a
        # simple mapping. NB: we do not need to build such mapping for
        # prediction indices, because we will simply enumerate
        # pred_class_idx, which provides both local and global indices
        gt_idx_to_i_gt = torch.full_like(gt_semantic, -1)
        gt_idx_to_i_gt[gt_class_idx] = torch.arange(num_gt, device=device)

        # Match each prediction to a ground truth
        # NB: the assumed pre-ordering of predictions by decreasing
        # score ensures we are assigning high-confidence predictions
        # first
        for i_pred, pred_idx in enumerate(pred_class_idx):

            # Get the indices of pairs which involve the prediction at
            # hand and whose ground truth has the class at hand
            _pair_idx = torch.where(
                (pair_pred_idx == pred_idx) & is_pair_gt_class)[0]

            # Detection will be ignored if no candidate overlapping gt
            # is found
            if _pair_idx.numel() == 0:
                continue

            # Gather the pairs' ground truth information for candidate
            # ground truth matches
            _pair_gt_idx = pair_gt_idx[_pair_idx]
            _pair_i_gt = gt_idx_to_i_gt[_pair_gt_idx]
            _pair_iou = pair_iou[_pair_idx]

            # Sort the candidates by decreasing gt size. In case the
            # prediction has multiple candidate ground truth matches
            # with equal IoU, we will select the one with the largest
            # size in priority
            if _pair_gt_idx.numel() > 1:
                order = gt_size[_pair_gt_idx].argsort(descending=True)
                _pair_idx = _pair_idx[order]
                _pair_gt_idx = _pair_gt_idx[order]
                _pair_i_gt = _pair_i_gt[order]
                _pair_iou = _pair_iou[order]
                del order

            # Among potential ground truth matches, remove those which
            # are already matched with a prediction.
            # NB: we do not remove the 'ignored' ground truth yet: if a
            # ground truth is 'ignored', we still want to match it to a
            # good prediction, if any. This way, the prediction in
            # question will also be marked as to be 'ignored', else it
            # would be unfairly penalized as a False Positive
            _iou_pair_gt_matched = gt_matches[:, _pair_i_gt]

            # For each IoU and each candidate ground truth, search the
            # available candidates with large-enough IoU.
            # NB: clamping the thresholds to 0 will facilitate searching
            # for the best match for each prediction while identifying
            # False Positives
            iou_thresholds = torch.tensor(self.iou_thresholds, device=device)

            # Get the best possible matching pair for each IoU threshold
            _iou_match, _iou_match_idx = \
                (~_iou_pair_gt_matched * _pair_iou.view(1, -1)).max(dim=1)

            # Check if the match found for each IoU threshold is valid.
            # A match is valid if:
            #   - the match's IoU is above the IoU threshold
            #   - the ground truth is not already matched
            # A match may be valid but still be ignored if:
            #   - the ground truth is marked as ignored
            _iou_match_ok = _iou_match > iou_thresholds

            # For each IoU threshold, get the corresponding ground truth
            # index. From there, we can update the det_ignore,
            # det_matches and gt_matches.
            _iou_match_i_gt = _pair_i_gt[_iou_match_idx]
            det_ignore[:, i_pred] = gt_ignore[_iou_match_i_gt] * _iou_match_ok
            det_matches[:, i_pred] = _iou_match_ok

            #  Special attention must be paid to gt_matches in case the
            #  prediction tried to match an already-assigned gt. In
            #  which case the prediction will not match (ie
            #  _iou_match_ok is False). To avoid re-setting the
            #  corresponding gt_matches to False, we need to make sure
            #  gt_matches was not already matched
            _iou_match_gt_ok = \
                gt_matches.gather(1, _iou_match_i_gt.view(-1, 1)).squeeze()
            _iou_match_gt_ok = _iou_match_gt_ok | _iou_match_ok
            gt_matches.scatter_(
                1, _iou_match_i_gt.view(-1, 1), _iou_match_gt_ok.view(-1, 1))

        # The above procedure may leave some predictions without match.
        # Those should count as False Positives, unless their size is
        # outside the size_range of interest, in which case it should be
        # ignored from metrics computation
        det_ignore = det_ignore | ~det_matches \
                     & ~is_pred_in_size_range[is_pred_class].view(1, -1)

        return {
            "dtMatches": det_matches,
            "gtMatches": gt_matches,
            "dtScores": pred_score[is_pred_class],
            "gtIgnore": gt_ignore,
            "dtIgnore": det_ignore}

    def _move_list_states_to_cpu(self) -> None:
        """Move list states to cpu to save GPU memory."""
        for key in self._defaults.keys():
            current_val = getattr(self, key)
            if isinstance(current_val, Sequence):
                setattr(self, key, [cur_v.to("cpu") for cur_v in current_val])

    def to(self, *args, **kwargs):
        """Overwrite torch.nn.Module.to() to handle the InstanceData
        stored in the internal states.
        """
        instance_data = getattr(self, 'instance_data', None)
        if instance_data is not None:
            self.instance_data = torch.zeros(1, device=self.device)
        out = super().to(*args, **kwargs)
        if instance_data is not None:
            self.instance_data = instance_data
            out.instance_data = [x.to(*args, **kwargs) for x in instance_data]
        return out

    def _get_classes(self) -> List:
        """Returns a list of unique classes found in ground truth and
        detection data, excluding 'stuff' classes if any.
        """
        if len(self.prediction_semantic) > 0 or len(self.instance_data) > 0:
            all_pred_y = self.prediction_semantic
            all_gt_y = [x.y for x in self.instance_data]
            all_y = torch.cat(all_pred_y + all_gt_y).unique().tolist()
            all_y_without_stuff = list(set(all_y) - set(self.stuff_classes))
            return all_y_without_stuff
        return []

    def _summarize_results(
            self,
            precisions: Tensor,
            recalls: Tensor
    ) -> PanopticMetricResults:
        """Summarizes ...
        """
        return PanopticMetricResults
