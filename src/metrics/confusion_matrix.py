# Source: https://github.com/torch-points3d/torch-points3d

import numpy as np
import torch
import os
from torchmetrics.classification import MulticlassConfusionMatrix
from src.utils import histogram_to_atomic


class ConfusionMatrix(MulticlassConfusionMatrix):
    """TorchMetrics's MulticlassConfusionMatrix but tailored to our
    needs. In particular, new methods allow computing OA, mAcc, mIoU
    and per-class IoU.

    :param num_classes: int
        Number of classes in the confusion matrix
    :param ignore_index: int
        Specifies a target value that is ignored and does not
        contribute to the metric calculation
    :param pointwise: bool
        If True and when target is a 2D tensor. Target will be treated
        as a histogram and metrics will be computed at a point
        level. Otherwise, any 2D target will be argmaxed along its
        2nd dimension to recover the dominant label in each
        histogram and the metrics will be computed at the segment
        level. See `self.update()`
    """

    def __init__(self, num_classes, ignore_index=None, pointwise=True):
        super().__init__(
            num_classes, ignore_index=ignore_index, normalize=None,
            validate_args=False)
        self.pointwise = pointwise

    def update(self, preds, target):
        """Update state with predictions and targets. Extends the
        `MulticlassConfusionMatrix.update()` with the possibility to
        pass histograms as targets. This is needed when computing
        point-level metrics from segments classification.

        :param preds: Tensor
            Predictions
        :param target: Tensor
            Ground truth
        """
        # If we want to compute poitnwise metrics and received
        # histograms as target labels
        if self.pointwise and target.dim() > 1 and target.shape[1] > 1:
            target, preds = histogram_to_atomic(target, preds)

        # If we received histograms as target labels but want to compute
        # segment-level metrics only
        elif not self.pointwise and target.dim() > 1:
            if target.shape[1] == 1:
                target = target.squeeze()
            else:
                target = target.argmax(dim=1)

        super().update(preds, target)

    @classmethod
    def from_matrix(cls, confusion_matrix):
        assert confusion_matrix.shape[0] == confusion_matrix.shape[1]
        matrix = cls(confusion_matrix.shape[0])
        matrix.confmat = confusion_matrix
        return matrix

    @classmethod
    def from_histogram(cls, h, class_names=None, verbose=False):
        assert h.ndim == 2
        device = h.device

        # Initialization
        num_nodes, num_classes = h.shape

        # Instantiate the ConfusionMatrix object
        cm = cls(num_classes)

        # Compute the dominant class in each superpoint
        pred_super = h.argmax(dim=1)

        # Compute the corresponding pointwise prediction and ground truth
        gt = torch.arange(
            num_classes, device=device).repeat(num_nodes).repeat_interleave(
            h.flatten())
        pred = pred_super.repeat_interleave(h.sum(dim=1))

        # Store in the ConfusionMatrix
        cm(pred, gt)

        if not verbose:
            return cm

        # Compute and print the metrics
        print(f'OA: {100 * cm.oa():0.2f}')
        print(f'mIoU: {100 * cm.miou():0.2f}')
        class_iou = cm.iou()
        class_names = class_names if class_names else range(num_nodes)
        for c, iou, seen in zip(class_names, class_iou[0], class_iou[1]):
            if not seen:
                print(f'  {c:<13}: not seen')
                continue
            print(f'  {c:<13}: {100 * iou:0.2f}')

        return cm

    def iou(self, as_percent=True):
        """Computes the Intersection over Union of each class in the
        confusion matrix

        :param as_percent: bool
            If True, the returned metric is expressed in [0, 100]

        Return:
            (iou, missing_class_mask) - iou for class as well as a mask
            highlighting existing classes
        """
        TP_plus_FN = self.confmat.sum(dim=0)
        TP_plus_FP = self.confmat.sum(dim=1)
        TP = self.confmat.diag()
        union = TP_plus_FN + TP_plus_FP - TP
        iou = 1e-8 + TP / (union + 1e-8)
        existing_class_mask = union > 1e-3
        if as_percent:
            iou *= 100
        return iou, existing_class_mask

    def oa(self, as_percent=True):
        """Compute the Overall Accuracy of the confusion matrix.

        :param as_percent: bool
            If True, the returned metric is expressed in [0, 100]
        """
        confusion_matrix = self.confmat
        matrix_diagonal = 0
        all_values = 0
        for row in range(self.num_classes):
            for column in range(self.num_classes):
                all_values += confusion_matrix[row][column]
                if row == column:
                    matrix_diagonal += confusion_matrix[row][column]
        if all_values == 0:
            all_values = 1
        if as_percent:
            matrix_diagonal *= 100
        return float(matrix_diagonal) / all_values

    def miou(self, missing_as_one=False, as_percent=True):
        """Computes the mean Intersection over Union of the confusion
        matrix. Get the mIoU metric by ignoring missing labels.

        :param missing_as_one: bool
            If True, then treats missing classes in the IoU as 1
        :param as_percent: bool
            If True, the returned metric is expressed in [0, 100]
        """
        values, existing_classes_mask = self.iou(as_percent=as_percent)
        if existing_classes_mask.sum() == 0:
            return 0
        if missing_as_one:
            values[~existing_classes_mask] = 1
            existing_classes_mask[:] = True
        return values[existing_classes_mask].sum() / existing_classes_mask.sum()

    def macc(self, as_percent=True):
        """Compute the mean of per-class accuracy in the confusion
        matrix.

        :param as_percent: bool
            If True, the returned metric is expressed in [0, 100]
        """
        re = 0
        label_presents = 0
        for i in range(self.num_classes):
            total_gt = self.confmat[i, :].sum()
            if total_gt:
                label_presents += 1
                re = re + self.confmat[i][i] / max(1, total_gt)
        if label_presents == 0:
            return 0
        if as_percent:
            re *= 100
        return re / label_presents


def save_confusion_matrix(cm, path2save, ordered_names):
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set(font_scale=5)

    if isinstance(cm, torch.Tensor):
        cm = cm.cpu().float().numpy()

    template_path = os.path.join(path2save, "{}.svg")
    # PRECISION
    cmn = cm.astype("float") / cm.sum(axis=-1)[:, np.newaxis]
    cmn[np.isnan(cmn) | np.isinf(cmn)] = 0
    fig, ax = plt.subplots(figsize=(31, 31))
    sns.heatmap(
        cmn, annot=True, fmt=".2f", xticklabels=ordered_names,
        yticklabels=ordered_names, annot_kws={"size": 20})
    # g.set_xticklabels(g.get_xticklabels(), rotation = 35, fontsize = 20)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    path_precision = template_path.format("precision")
    plt.savefig(path_precision, format="svg")

    # RECALL
    cmn = cm.astype("float") / cm.sum(axis=0)[np.newaxis, :]
    cmn[np.isnan(cmn) | np.isinf(cmn)] = 0
    fig, ax = plt.subplots(figsize=(31, 31))
    sns.heatmap(
        cmn, annot=True, fmt=".2f", xticklabels=ordered_names,
        yticklabels=ordered_names, annot_kws={"size": 20})
    # g.set_xticklabels(g.get_xticklabels(), rotation = 35, fontsize = 20)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    path_recall = template_path.format("recall")
    plt.savefig(path_recall, format="svg")
