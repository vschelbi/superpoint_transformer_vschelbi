import sys
from .sampling import GridSampling3D, SaveOriginalPosId, sample_clusters
from .neighbors import search_outliers, search_neighbors
from .features import compute_point_features
from .graph import compute_adjacency_graph, compute_cluster_graph
from .partition import compute_partition, compute_grid_partition
from torch_geometric.transforms import Compose

# Fuse all transforms defined in this project with the torch_geometric
# transforms
_custom_transforms = sys.modules[__name__]
_torch_geometric_transforms = sys.modules["torch_geometric.transforms"]
_intersection_names = set(_custom_transforms.__dict__) & set(_torch_geometric_transforms.__dict__)
_intersection_names = set([module for module in _intersection_names if not module.startswith("_")])
L_intersection_names = len(_intersection_names) > 0
_intersection_cls = []

for transform_name in _intersection_names:
    transform_cls = getattr(_custom_transforms, transform_name)
    if not "torch_geometric.transforms." in str(transform_cls):
        _intersection_cls.append(transform_cls)
L_intersection_cls = len(_intersection_cls) > 0

if L_intersection_names:
    if L_intersection_cls:
        raise Exception(
            f"It seems that you are overriding a transform from pytorch "
            f"geometric, this is forbidden, please rename your classes "
            f"{_intersection_names} from {_intersection_cls}")
    else:
        raise Exception(
            f"It seems you are importing transforms {_intersection_names} "
            f"from pytorch geometric within the current code base. Please, "
            f"remove them or add them within a class, function, etc.")


def instantiate_transform(transform_option, attr="transform"):
    """Create a transform from an OmegaConf dict such as

    ```
    transform: GridSampling3D
        params:
            size: 0.01
    ```
    """
    tr_name = getattr(transform_option, attr, None)
    try:
        tr_params = transform_option.get('params')  # Update to OmegaConf 2.0
    except KeyError:
        tr_params = None
    try:
        lparams = transform_option.get('lparams')  # Update to OmegaConf 2.0
    except KeyError:
        lparams = None

    cls = getattr(_custom_transforms, tr_name, None)
    if not cls:
        cls = getattr(_torch_geometric_transforms, tr_name, None)
        if not cls:
            raise ValueError(f"Transform {tr_name} is nowhere to be found")

    if tr_params and lparams:
        return cls(*lparams, **tr_params)

    if tr_params:
        return cls(**tr_params)

    if lparams:
        return cls(*lparams)

    return cls()


def instantiate_transforms(transform_options):
    """Create a torch_geometric composite transform from an OmegaConf
    list such as

    ```
    - transform: GridSampling3D
        params:
            size: 0.01
    - transform: NormaliseScale
    ```
    """
    transforms = []
    for transform in transform_options:
        transforms.append(instantiate_transform(transform))
    return Compose(transforms)
