import torch
import numpy as np
import src.partition.utils.libpoint_utils as point_utils
from src.utils.features import rgb2hsv, rgb2lab
from src.transforms import Transform
from src.data import NAG


__all__ = [
    'PointFeatures', 'JitterColor', 'JitterFeatures', 'ColorAutoContrast',
    'ColorDrop', 'ColorNormalize']


class PointFeatures(Transform):
    """Compute pointwise features based on what is already available in
    the Data object.

    All local geometric features assume the input ``Data`` has a
    ``neighbors`` attribute, holding a ``(num_nodes, k)`` tensor of
    indices. All k neighbors will be used for local geometric features
    computation, unless some are missing (indicated by -1 indices). If
    the latter, only positive indices will be used.

    Parameters
    ----------
    pos: bool
        Use point position.
    rgb: bool
        Use rgb color. Assumes Data.rgb holds either [0, 1] floats or
        [0, 255] integers
    hsv: bool
        Use HSV color. Assumes Data.rgb holds either [0, 1] floats or
        [0, 255] integers
    lab: bool
        Use LAB color. Assumes Data.rgb holds either [0, 1] floats or
        [0, 255] integers
    density: bool
        Use local density. Assumes ``Data.neighbor_index`` and
        ``Data.neighbor_distance``.
    linearity: bool
        Use local linearity. Assumes ``Data.neighbor_index``.
    planarity: bool
        Use local planarity. Assumes ``Data.neighbor_index``.
    scattering: bool
        Use local scattering. Assumes ``Data.neighbor_index``.
    verticality: bool
        Use local verticality. Assumes ``Data.neighbor_index``.
    normal: bool
        Use local normal. Assumes ``Data.neighbor_index``.
    length: bool
        Use local length. Assumes ``Data.neighbor_index``.
    surface: bool
        Use local surface. Assumes ``Data.neighbor_index``.
    volume: bool
        Use local volume. Assumes ``Data.neighbor_index``.
    curvature: bool
        Use local curvature. Assumes ``Data.neighbor_index``.
    k_min: int
        Minimum number of neighbors to consider for geometric features
        computation. Points with less than k_min neighbors will receive
        0-features. Assumes ``Data.neighbor_index``.
    """

    # TODO: augment with Rep-SURF umbrella features ?
    # TODO: Random PointNet + PCA features ?

    def __init__(
            self, pos=False, radius=5, rgb=False, hsv=False, lab=False,
            density=False, linearity=False, planarity=False, scattering=False,
            verticality=False, normal=False, length=False, surface=False,
            volume=False, curvature=False, k_min=5):

        self.pos = pos
        self.radius = radius
        self.rgb = rgb
        self.hsv = hsv
        self.lab = lab
        self.density = density
        self.linearity = linearity
        self.planarity = planarity
        self.scattering = scattering
        self.verticality = verticality
        self.normal = normal
        self.length = length
        self.surface = surface
        self.volume = volume
        self.curvature = curvature
        self.k_min = k_min

    def _process(self, data):
        assert data.has_neighbors, \
            "Data is expected to have a 'neighbor_index' attribute"
        assert data.num_nodes < np.iinfo(np.uint32).max, \
            "Too many nodes for `uint32` indices"
        assert data.neighbor_index.max() < np.iinfo(np.uint32).max, \
            "Too high 'neighbor_index' indices for `uint32` indices"

        features = []

        # Add xyz normalized. The scaling factor drives the maximum
        # cluster size the partition may produce
        if self.pos and data.pos is not None:
            features.append((data.pos - data.pos.mean(dim=0)) / self.radius)

        # Add RGB to the features. If colors are stored in int, we
        # assume they are encoded in  [0, 255] and normalize them.
        # Otherwise, we assume they have already been [0, 1] normalized
        if self.rgb and data.rgb is not None:
            f = data.rgb
            if f.dtype in [torch.uint8, torch.int, torch.long]:
                f = f.float() / 255
            features.append(f)

        # Add HSV to the features. If colors are stored in int, we
        # assume they are encoded in  [0, 255] and normalize them.
        # Otherwise, we assume they have already been [0, 1] normalized.
        # Note: for all features to live in a similar range, we
        # normalize H in [0, 1]
        if self.hsv and data.rgb is not None:
            f = data.rgb
            if f.dtype in [torch.uint8, torch.int, torch.long]:
                f = f.float() / 255
            hsv = rgb2hsv(f)
            hsv[:, 0] /= 360.
            features.append(hsv)

        # Add LAB to the features. If colors are stored in int, we
        # assume they are encoded in  [0, 255] and normalize them.
        # Otherwise, we assume they have already been [0, 1] normalized.
        # Note: for all features to live in a similar range, we
        # normalize L in [0, 1] and ab in [-1, 1]
        if self.lab and data.rgb is not None:
            f = data.rgb
            if f.dtype in [torch.uint8, torch.int, torch.long]:
                f = f.float() / 255
            features.append(rgb2lab(f) / 100)

        # Add local surfacic density to the features. The local density
        # is approximated as K / D² where K is the number of nearest
        # neighbors and D is the distance of the Kth neighbor. We
        # normalize by D² since points roughly lie on a 2D manifold.
        # Note that this takes into account partial neighborhoods where
        # -1 indicates absent neighbors
        if self.density:
            dmax = data.neighbor_distance.max(dim=1).values
            k = data.neighbor_index.ge(0).sum(dim=1)
            features.append((k / dmax ** 2).view(-1, 1))

        # Add local geometric features
        needs_geof = any((
            self.linearity, self.planarity, self.scattering, self.verticality,
            self.normal))
        if needs_geof and data.pos is not None:

            # Prepare data for numpy boost interface. Note: we add each
            # point to its own neighborhood before computation
            xyz = data.pos.cpu().numpy()
            nn = torch.cat(
                (torch.arange(xyz.shape[0]).view(-1, 1), data.neighbor_index), dim=1)
            k = nn.shape[1]

            # Check for missing neighbors (indicated by -1 indices)
            n_missing = (nn < 0).sum(dim=1)
            if (n_missing > 0).any():
                sizes = k - n_missing
                nn = nn[nn >= 0]
                nn_ptr = torch.cat((torch.zeros(1), sizes.cumsum(dim=0).cpu()))
            else:
                nn = nn.flatten().cpu()
                nn_ptr = torch.arange(xyz.shape[0] + 1) * k
            nn = nn.numpy().astype('uint32')
            nn_ptr = nn_ptr.numpy().astype('uint32')

            # Make sure array are contiguous before moving to C++
            xyz = np.ascontiguousarray(xyz)
            nn = np.ascontiguousarray(nn)
            nn_ptr = np.ascontiguousarray(nn_ptr)

            # C++ geometric features computation on CPU
            f = point_utils.compute_geometric_features(
                xyz, nn, nn_ptr, self.k_min, False)
            f = torch.from_numpy(f.astype('float32'))

            # Heuristic to increase the importance of verticality
            f[:, 3] *= 2

            # Select only required features
            mask = (
                [self.linearity, self.planarity, self.scattering, self.verticality]
                + [self.normal] * 3
                + [self.length, self.surface, self.volume, self.curvature])
            features.append(f[:, mask].to(data.pos.device))

        # Save all features in the Data.x attribute
        data.x = torch.cat(features, dim=1).to(data.pos.device)

        return data


class JitterColor(Transform):
    """Add some gaussian noise to data.rgb for all data in a NAG.

    :param sigma: float or List(float)
        Standard deviation of the gaussian noise. A list may be passed
        to transform NAG levels with different parameters. Passing
        sigma <= 0 will prevent any jittering.
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG

    def __init__(self, sigma=0.05):
        assert isinstance(sigma, (int, float, list))
        self.sigma = float(sigma)

    def _process(self, nag):
        device = nag.device

        if not isinstance(self.sigma, list):
            sigma = [self.sigma] * nag.num_levels
        else:
            sigma = self.sigma

        for i_level in range(nag.num_levels):

            if sigma[i_level] <= 0 or getattr(nag[i_level], 'rgb', None) is None:
                continue

            noise = torch.randn_like(nag[i_level].rgb, device=device) * self.sigma
            nag[i_level].rgb += noise

        return nag


class JitterFeatures(Transform):
    """Add some gaussian noise to data.x for all data in a NAG.

    :param sigma: float or List(float)
        Standard deviation of the gaussian noise. A list may be passed
        to transform NAG levels with different parameters. Passing
        sigma <= 0 will prevent any jittering.
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG

    def __init__(self, sigma=0.01):
        assert isinstance(sigma, (int, float, list))
        self.sigma = float(sigma)

    def _process(self, nag):
        device = nag.device

        if not isinstance(self.sigma, list):
            sigma = [self.sigma] * nag.num_levels
        else:
            sigma = self.sigma

        for i_level in range(nag.num_levels):

            if sigma[i_level] <= 0 or getattr(nag[i_level], 'x', None) is None:
                continue

            noise = torch.randn_like(nag[i_level].x, device=device) * self.sigma
            nag[i_level].x += noise

        return nag


class ColorAutoContrast(Transform):
    """Apply some random contrast to the point colors.

    credit: https://github.com/guochengqian/openpoints

    :param p: float
        Probability of the transform to be applied
    :param blend: float (optional)
        Blend factor, controlling the contrasting intensity
    """

    def __init__(self, p=0.2, blend=None):
        self.p = p
        self.blend = blend

    def __call__(self, data):
        if getattr(data, 'rgb', None) is not None:
            return data

        device = data.device

        assert data.rgb.dtype == torch.float and 0 <= data.rgb.min() \
               and data.rgb.max() <= 1, 'expected float colors in [0, 1]'

        if torch.rand(1, device=device) < self.p:

            # Compute the contrasted colors
            lo = data.rgb.min(dim=0).values.view(1, -1)
            hi = data.rgb.max(dim=0).values.view(1, -1)
            contrast_feat = (data.rgb - lo) / (hi - lo)

            # Blend the maximum contrast with the current color
            blend = torch.rand(1, device=device) \
                if self.blend is None else self.blend
            data.rgb = (1 - blend) * data.rgb + blend * contrast_feat

        return data


class ColorDrop(Transform):
    """Randomly set point colors to 0.

    :param p: float
        Probability of the transform to be applied
    """

    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, data):
        if getattr(data, 'rgb', None) is not None:
            return data

        if torch.rand(1, device=data.device) < self.p:
            data.rgb *= 0

        return data


class ColorNormalize(Transform):
    """Normalize the colors using given means and standard deviations.

    credit: https://github.com/guochengqian/openpoints

    :param mean: list
        Channel-wise means
    :param std: list
        Channel-wise standard deviations
    """

    def __init__(
            self, mean=[0.5136457, 0.49523646, 0.44921124],
            std=[0.18308958, 0.18415008, 0.19252081]):
        self.mean = torch.tensor(mean).float().view(1, -1)
        self.std = torch.tensor(std).float().view(1, -1)
        assert self.std.gt(0).all(), "std values must be >0"

    def __call__(self, data):
        if getattr(data, 'rgb', None) is not None:
            return data

        device = data.device

        if self.mean.device != device or self.std.device != device:
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)

        assert data.rgb.dtype == torch.float and 0 <= data.rgb.min() \
               and data.rgb.max() <= 1, 'expected float colors in [0, 1]'

        data.rgb = (data.rgb - self.mean) / self.std

        return data
