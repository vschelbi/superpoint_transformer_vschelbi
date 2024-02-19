import torch
import logging
import src

log = logging.getLogger(__name__)


__all__ = ['SemanticSegmentationOutput']


class SemanticSegmentationOutput:
    """A simple holder for semantic segmentation model output, with a
    few helper methods for manipulating the predictions and targets
    (if any).
    """

    def __init__(self, logits, y_hist=None):
        self.logits = logits
        self.y_hist = y_hist
        if src.is_debug_enabled():
            self.debug()

    def debug(self):
        """Runs a series of sanity checks on the attributes of self.
        """
        assert isinstance(self.logits, torch.Tensor) \
               or all(isinstance(l, torch.Tensor) for l in self.logits)
        if self.has_target:
            if self.multi_stage:
                assert len(self.y_hist) == len(self.logits)
                assert all(
                    y.shape[0] == l.shape[0]
                    for y, l in zip(self.y_hist, self.logits))
            else:
                assert self.y_hist.shape[0] == self.logits.shape[0]

    @property
    def has_target(self):
        """Check whether `self` contains target data for semantic
        segmentation.
        """
        return self.y_hist is not None

    @property
    def multi_stage(self):
        """If the semantic segmentation `logits` are stored in an
        enumerable, then the model output is multi-stage.
        """
        return not isinstance(self.logits, torch.Tensor)

    @property
    def num_classes(self):
        """Number for semantic classes in the output predictions.
        """
        logits = self.logits[0] if self.multi_stage else self.logits
        return logits.shape[1]

    @property
    def num_nodes(self):
        """Number for nodes/superpoints in the output predictions. By
        default, for a hierarchical partition, this means counting the
        number of level-1 nodes/superpoints.
        """
        logits = self.logits[0] if self.multi_stage else self.logits
        return logits.shape[0]

    @property
    def preds(self):
        """Final semantic segmentation predictions are the argmax of the
        first-level partition logits.
        """
        logits = self.logits[0] if self.multi_stage else self.logits
        return torch.argmax(logits, dim=1)

    @property
    def targets(self):
        """Final semantic segmentation targets are the label histogram
        of the first-level partition logits.
        """
        return self.y_hist[0] if self.multi_stage else self.y_hist

    @property
    def void_mask(self):
        """Returns a mask on the level-1 nodes indicating which is void.
        By convention, nodes/superpoints are void if they contain
        more than 50% void points. By convention in this project, void
        points have the label `num_classes`. In label histograms, void
        points are counted in the last column.
        """
        if not self.has_target:
            return

        # For simplicity, we only return the mask for the level-1
        y_hist = self.targets
        total_count = y_hist.sum(dim=1)
        void_count = y_hist[:, -1]
        return void_count / total_count > 0.5

    def __repr__(self):
        return f"{self.__class__.__name__}()"
