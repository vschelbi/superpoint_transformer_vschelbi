from superpoint_transformer.transforms import Transform
from superpoint_transformer.utils.neighbors import inliers_split, outliers_split


__all__ = ['Inliers', 'Outliers']


class Inliers(Transform):
    """Search for points with `k_min` OR MORE neighbors within a
    radius of `r_max`.

    Since removing outliers may cause some points to become outliers
    themselves, this problem can be tackled with the `recursive` option.
    Note that this recursive search holds no guarantee of reasonable
    convergence as one could design a point cloud for given `k_min` and
    `r_max` whose points would all recursively end up as outliers.
    """

    def __init__(
            self, k_min, r_max=1, recursive=False, update_sub=False,
            update_super=False):
        self.k_min = k_min
        self.r_max = r_max
        self.recursive = recursive
        self.update_sub = update_sub
        self.update_super = update_super

    def _process(self, data):
        # Actual outlier search, optionally recursive
        idx = inliers_split(
            data.pos, data.pos, self.k_min, r_max=self.r_max,
            recursive=self.recursive, q_in_s=True)

        # Select the points of interest in Data
        return data.select(
            idx, update_sub=self.update_sub, update_super=self.update_super)


class Outliers(Transform):
    """Search for points with LESS THAN `k_min` neighbors within a
    radius of `r_max`.

    Since removing outliers may cause some points to become outliers
    themselves, this problem can be tackled with the `recursive` option.
    Note that this recursive search holds no guarantee of reasonable
    convergence as one could design a point cloud for given `k_min` and
    `r_max` whose points would all recursively end up as outliers.
    """

    def __init__(
            self, k_min, r_max=1, recursive=False, update_sub=False,
            update_super=False):
        self.k_min = k_min
        self.r_max = r_max
        self.recursive = recursive
        self.update_sub = update_sub
        self.update_super = update_super

    def _process(self, data):
        # Actual outlier search, optionally recursive
        idx = outliers_split(
            data.pos, data.pos, self.k_min, r_max=self.r_max,
            recursive=self.recursive, q_in_s=True)

        # Select the points of interest in Data
        return data.select(
            idx, update_sub=self.update_sub, update_super=self.update_super)
