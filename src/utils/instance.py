import sys
import torch
import numpy as np
import os.path as osp
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from torch.nn.functional import one_hot, softmax
from torch_scatter import scatter_sum
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from src.utils.neighbors import knn_2
from src.utils.graph import to_trimmed
from src.utils.cpu import available_cpu_count

src_folder = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(src_folder)
sys.path.append(osp.join(src_folder, "dependencies/grid_graph/python/bin"))
sys.path.append(osp.join(src_folder, "dependencies/parallel_cut_pursuit/python/wrappers"))

from grid_graph import edge_list_to_forward_star
from cp_kmpp_d0_dist import cp_kmpp_d0_dist


__all__ = [
    'generate_random_bbox_data', 'generate_random_segment_data',
    'instance_cut_pursuit', 'oracle_superpoint_clustering', 'get_stuff_mask']


_MAX_NUM_EDGES = 4294967295


def generate_random_bbox_data(
        num_img=1,
        num_classes=1,
        height=128,
        width=128,
        h_split=1,
        w_split=2,
        det_gt_ratio=1):
    # Create some images with a ground truth partition
    instance_images = -torch.ones(num_img, height, width, dtype=torch.long)
    label_images = -torch.ones(num_img, height, width, dtype=torch.long)

    h_gt = height // h_split
    w_gt = width // w_split

    gt_boxes = torch.zeros(num_img * h_split * w_split, 4)
    gt_labels = torch.randint(0, num_classes, (num_img * h_split * w_split,))
    iterator = product(range(num_img), range(h_split), range(w_split))
    for idx, (i_img, i, j) in enumerate(iterator):
        h1 = i * h_gt
        h2 = (i + 1) * h_gt
        w1 = j * w_gt
        w2 = (j + 1) * w_gt
        instance_images[i_img, h1:h2, w1:w2] = idx
        label_images[i_img, h1:h2, w1:w2] = gt_labels[idx]
        gt_boxes[idx] = torch.tensor([h1, w1, h2, w2])

    # Create some random detection boxes
    num_gt = (instance_images.max() + 1).item()
    num_det = int(num_gt * det_gt_ratio)

    i_center_det = torch.randint(0, height, (num_det,))
    j_center_det = torch.randint(0, width, (num_det,))
    h_det = torch.randint(int(h_gt * 0.7), int(h_gt * 1.3), (num_det,))
    w_det = torch.randint(int(w_gt * 0.7), int(w_gt * 1.3), (num_det,))

    det_boxes = torch.vstack([
        (i_center_det - h_det / 2).clamp(min=0),
        (j_center_det - w_det / 2).clamp(min=0),
        (i_center_det + h_det / 2).clamp(max=height),
        (j_center_det + w_det / 2).clamp(max=width)]).T.round()
    det_img_idx = torch.randint(0, num_img, (num_det,))
    det_labels = torch.randint(0, num_classes, (num_det,))
    det_scores = torch.rand(num_det)

    # Display the images stacked along their height (first dim) and draw
    # the box for each detection
    fig, ax = plt.subplots()
    ax.imshow(instance_images.view(-1, width), cmap='jet')
    for idx_det in range(num_det):
        i = det_boxes[idx_det, 0] + det_img_idx[idx_det] * height
        j = det_boxes[idx_det, 1]
        h = det_boxes[idx_det, 2] - det_boxes[idx_det, 0]
        w = det_boxes[idx_det, 3] - det_boxes[idx_det, 1]
        rect = patches.Rectangle(
            (j, i),
            w,
            h,
            linewidth=3,
            edgecolor=cm.nipy_spectral(idx_det / num_det),
            facecolor='none')
        ax.add_patch(rect)
    plt.show()

    # Display the images stacked along their height (first dim) and draw the
    # box for each detection
    fig, ax = plt.subplots()
    ax.imshow(label_images.view(-1, width).float() / num_classes, cmap='jet')
    for idx_det in range(num_det):
        i = det_boxes[idx_det, 0] + det_img_idx[idx_det] * height
        j = det_boxes[idx_det, 1]
        h = det_boxes[idx_det, 2] - det_boxes[idx_det, 0]
        w = det_boxes[idx_det, 3] - det_boxes[idx_det, 1]
        c = cm.nipy_spectral(det_labels[idx_det].float().item() / num_classes)
        rect = patches.Rectangle(
            (j, i),
            w,
            h,
            linewidth=3,
            edgecolor=c,
            facecolor='none')
        ax.add_patch(rect)
    plt.show()

    # Compute the metrics using torchmetrics
    iterator = zip(gt_boxes.view(num_img, -1, 4), gt_labels.view(num_img, -1))
    targets = [
        dict(boxes=boxes, labels=labels)
        for boxes, labels in iterator]

    preds = [
        dict(
            boxes=det_boxes[det_img_idx == i_img],
            labels=det_labels[det_img_idx == i_img],
            scores=det_scores[det_img_idx == i_img])
        for i_img in range(num_img)]

    # For each predicted pixel, we compute the gt object idx, and the gt
    # label, to build an InstanceData.
    # NB: we cannot build this by creating a single pred_idx image,
    # because predictions may overlap in this toy setup, unlike our 3D
    # superpoint partition paradigm...
    pred_idx = []
    gt_idx = []
    gt_y = []
    for idx_det in range(num_det):
        i_img = det_img_idx[idx_det]
        x1, y1, x2, y2 = det_boxes[idx_det].long()
        num_points = (x2 - x1) * (y2 - y1)
        pred_idx.append(torch.full((num_points,), idx_det))
        gt_idx.append(instance_images[i_img, x1:x2, y1:y2].flatten())
        gt_y.append(label_images[i_img, x1:x2, y1:y2].flatten())
    pred_idx = torch.cat(pred_idx)
    gt_idx = torch.cat(gt_idx)
    gt_y = torch.cat(gt_y)
    count = torch.ones_like(pred_idx)

    from src.data.instance import InstanceData
    instance_data = InstanceData(pred_idx, gt_idx, count, gt_y, dense=True)

    return targets, preds, gt_idx, gt_y, count, instance_data


def generate_single_random_segment_image(
        num_gt=10,
        num_pred=12,
        num_classes=3,
        height=32,
        width=64,
        shift=5,
        random_pred_label=False,
        show=True,
        iterations=20):
    """Generate an image with random ground truth and predicted instance
    and semantic segmentation data. To make the images realisitc, and to
    ensure that the instances form a PARTITION of the image, we rely on
    voronoi cells. Besides, to encourage a realistic overalp between the
    predicted and target instances, the predcition cell centers are
    sampled near the target samples.
    """
    # Generate random pixel positions for the ground truth and the
    # prediction centers. To produce predictions with "controllable"
    # overlap with the targets, we use the gt's centers as seeds for the
    # prediction centers and randomly sample shift them
    x = torch.randint(0, height, (num_gt,))
    y = torch.randint(0, width, (num_gt,))
    gt_xy = torch.vstack((x, y)).T
    if num_pred <= num_gt:
        idx_ref_gt = torch.from_numpy(
            np.random.choice(num_gt, num_pred, replace=False))
    else:
        idx_ref_gt = torch.from_numpy(
            np.random.choice(num_gt, num_pred % num_gt, replace=False))
        idx_ref_gt = torch.cat((
            torch.arange(num_gt).repeat(num_pred // num_gt), idx_ref_gt))
    xy_shift = torch.randint(0, 2 * shift, (num_pred, 2)) - shift
    pred_xy = gt_xy[idx_ref_gt] + xy_shift
    clamp_min = torch.tensor([0, 0])
    clamp_max = torch.tensor([height, width])
    pred_xy = pred_xy.clamp(min=clamp_min, max=clamp_max)

    # The above prediction center generation process may produce
    # duplicates, which can in turn generate downstream errors. To avoid
    # this, we greedily search for duplicates and shift them
    already_used_xy_ids = []
    for i_pred, xy in enumerate(pred_xy):
        xy_id = xy[0] * width + xy[1]
        count = 0

        while xy_id in already_used_xy_ids and count < iterations:
            xy_shift = torch.randint(0, 2 * shift, (2,)) - shift
            xy = gt_xy[idx_ref_gt[i_pred]] + xy_shift
            xy = xy.clamp(min=clamp_min, max=clamp_max)
            xy_id = xy[0] * width + xy[1]
            count += 1

        if count == iterations:
            raise ValueError(
                f"Reached max iterations={iterations} while resampling "
                "duplicate prediction centers")

        already_used_xy_ids.append(xy_id)
        pred_xy[i_pred] = xy

    # Generate labels and scores
    gt_labels = torch.randint(0, num_classes, (num_gt,))
    if random_pred_label:
        pred_labels = torch.randint(0, num_classes, (num_pred,))
    else:
        pred_labels = gt_labels[idx_ref_gt]
    pred_scores = torch.rand(num_pred)

    # Generate a 3D point cloud representing the pixel coordinates of the
    # image. This will be used to compute the 1-NNs and, from there, a
    # partition into voronoi cells
    x, y = torch.meshgrid(
        torch.arange(height), torch.arange(width), indexing='ij')
    x = x.flatten()
    y = y.flatten()
    z = torch.zeros_like(x)
    xyz = torch.vstack((x, y, z)).T

    # Compute a gt segmentation image from the 1-NN of each pixel, wrt the
    # gt segment centers
    gt_xyz = torch.cat((gt_xy, torch.zeros_like(gt_xy[:, [0]])), dim=1).float()
    gt_nn = knn_2(gt_xyz, xyz.float(), 1, r_max=max(width, height))[0]
    gt_seg_image = gt_nn.view(height, width)
    gt_label_image = gt_labels[gt_seg_image]

    # Compute a pred segmentation image from the 1-NN of each pixel, wrt the
    # pred segment centers
    pred_xyz = torch.cat((pred_xy, torch.zeros_like(pred_xy[:, [0]])), dim=1).float()
    pred_nn = knn_2(pred_xyz, xyz.float(), 1, r_max=max(width, height))[0]
    pred_seg_image = pred_nn.view(height, width)
    pred_label_image = pred_labels[pred_seg_image]

    # Display the segment images
    if show:
        plt.subplot(2, 2, 1)
        plt.title('Ground truth instances')
        plt.imshow(gt_seg_image)
        plt.subplot(2, 2, 2)
        plt.title('Predicted instances')
        plt.imshow(pred_seg_image)
        plt.subplot(2, 2, 3)
        plt.title('Ground truth labels')
        plt.imshow(gt_label_image)
        plt.subplot(2, 2, 4)
        plt.title('Predicted labels')
        plt.imshow(pred_label_image)
        plt.show()

    # Organize the data into torchmetric-friendly format
    tm_targets = dict(
        masks=torch.stack([gt_seg_image == i_gt for i_gt in range(num_gt)]),
        labels=gt_labels)

    tm_preds = dict(
        masks=torch.stack([pred_seg_image == i_pred for i_pred in range(num_pred)]),
        labels=pred_labels,
        scores=pred_scores)

    tm_data = (tm_preds, tm_targets)

    # Organize the data into our custom format
    pred_idx = pred_seg_image.flatten()
    gt_idx = gt_seg_image.flatten()
    gt_y = gt_label_image.flatten()
    count = torch.ones_like(pred_idx)

    from src.data.instance import InstanceData
    instance_data = InstanceData(pred_idx, gt_idx, count, gt_y, dense=True)
    spt_data = (pred_scores, pred_labels, instance_data)

    return tm_data, spt_data


def generate_random_segment_data(
        num_img=2,
        num_gt_per_img=10,
        num_pred_per_img=14,
        num_classes=2,
        height=32,
        width=64,
        shift=5,
        random_pred_label=False,
        verbose=True):
    """Generate multiple images with random ground truth and predicted
    instance and semantic segmentation data. To make the images
    realistic, and to ensure that the instances form a PARTITION of the
    image, we rely on voronoi cells. Besides, to encourage a realistic
    overlap between the predicted and and target instances, the
    prediction cell centers are sampled near the target samples.
    """
    tm_data = []
    spt_data = []

    for i_img in range(num_img):
        if verbose:
            print(f"\nImage {i_img + 1}/{num_img}")
        tm_data_, spt_data_ = generate_single_random_segment_image(
            num_gt=num_gt_per_img,
            num_pred=num_pred_per_img,
            num_classes=num_classes,
            height=height,
            width=width,
            shift=shift,
            random_pred_label=random_pred_label,
            show=verbose)
        tm_data.append(tm_data_)
        spt_data.append(spt_data_)

    return tm_data, spt_data


def instance_cut_pursuit(
        node_pos,
        node_offset,
        node_logits,
        node_size,
        edge_index,
        edge_affinity_logits,
        do_sigmoid_affinity=True,
        regularization=1e-2,
        spatial_weight=1,
        cutoff=1,
        parallel=True,
        iterations=10,
        trim=False,
        discrepancy_epsilon=1e-3,
        verbose=False):
    """Partition an instance graph using cut-pursuit.

    :param node_pos: Tensor of shape [num_nodes, num_dim]
        Node positions
    :param node_offset: Tensor of shape [num_nodes, num_dim]
        Predicted instance centroid offset for each node
    :param node_logits: Tensor of shape [num_nodes, num_classes]
        Predicted classification logits for each node
    :param node_size: Tensor of shape [num_nodes]
        Size of each node
    :param edge_index: Tensor of shape [2, num_edges]
        Edges of the graph, in torch-geometric's format
    :param edge_affinity_logits: Tensor of shape [num_edges]
        Predicted affinity logits (ie in R+, before sigmoid) of each
        edge
    :param do_sigmoid_affinity: bool
        If True, a sigmoid will be applied on the `edge_affinity_logits`
        to convert the logits to [0, 1] affinities. If False, the input
        `edge_affinity_logits` will be used as is when computing the
        discrepancies
    :param regularization: float
        Regularization parameter for the partition
    :param spatial_weight: float
        Weight used to mitigate the impact of the point position in the
        partition. The larger, the less spatial coordinates matter. This
        can be loosely interpreted as the inverse of a maximum
        superpoint radius
    :param cutoff: float
        Minimum number of points in each cluster
    :param parallel: bool
        Whether cut-pursuit should run in parallel
    :param iterations: int
        Maximum number of iterations for each partition
    :param trim: bool
        Whether the input graph should be trimmed. See `to_trimmed()`
        documentation for more details on this operation
    :param discrepancy_epsilon: float
        Mitigates the maximum discrepancy. More precisely:
        `affinity=0 ⇒ discrepancy=discrepancy_epsilon`
    :param verbose: bool
    :return:
    """

    # Sanity checks
    assert node_pos.dim() == 2, \
        "`node_pos` must have shape `[num_nodes, num_dim]`"
    assert node_offset.dim() == 2, \
        "`node_offset` must have shape `[num_nodes, num_dim]`"
    assert node_logits.dim() == 2, \
        "`node_logits` must have shape `[num_nodes, num_classes]`"
    assert node_pos.shape == node_offset.shape, \
        "`node_pos` and `node_offset` must have the same shape"
    assert node_logits.shape[0] == node_pos.shape[0], \
        "`node_logits` and `node_pos` must have the same number of points"
    assert node_size.dim() == 1, \
        "`node_size` must have shape `[num_nodes]`"
    assert node_size.shape[0] == node_pos.shape[0], \
        "`node_size` and `node_pos` must have the same number of points"
    assert edge_index.dim() == 2 and edge_index.shape[0] == 2, \
        "`edge_index` must be of shape `[2, num_edges]`"
    assert edge_affinity_logits.dim() == 1, \
        "`edge_affinity` must be of shape `[num_edges]`"
    assert edge_affinity_logits.shape[0] == edge_index.shape[1], \
        "`edge_affinity` and `edge_index` must have the same number of edges"

    device = node_pos.device
    num_nodes = node_pos.shape[0]
    num_dim = node_pos.shape[1]
    num_classes = node_logits.shape[1]
    num_edges = edge_affinity_logits.numel()

    assert num_nodes < np.iinfo(np.uint32).max, \
        "Too many nodes for `uint32` indices"
    assert num_edges < np.iinfo(np.uint32).max, \
        "Too many edges for `uint32` indices"

    # Initialize the number of threads used for parallel cut-pursuit
    num_threads = available_cpu_count() if parallel else 1

    # Exit if the graph contains only one node
    if num_nodes < 2:
        return torch.zeros(num_nodes, dtype=torch.long, device=device)

    # Trim the graph, if need be
    if trim:
        edge_index, edge_affinity_logits = to_trimmed(
            edge_index, edge_attr=edge_affinity_logits, reduce='mean')

    if verbose:
        print(
            f'Launching instance partition reg={regularization}, '
            f'cutoff={cutoff}')

    # User warning if the number of edges exceeds uint32 limits
    if num_edges > _MAX_NUM_EDGES and verbose:
        print(
            f"WARNING: number of edges {num_edges} exceeds the uint32 limit "
            f"{_MAX_NUM_EDGES}. Please update the cut-pursuit source code to "
            f"accept a larger data type for `index_t`.")

    # Convert affinity logits to discrepancies
    edge_affinity = edge_affinity_logits.sigmoid() if do_sigmoid_affinity \
        else edge_affinity_logits
    edge_discrepancy = 1 / (edge_affinity + discrepancy_epsilon)

    # Convert edges to forward-star (or CSR) representation
    source_csr, target, reindex = edge_list_to_forward_star(
        num_nodes, edge_index.T.contiguous().cpu().numpy())
    source_csr = source_csr.astype('uint32')
    target = target.astype('uint32')
    edge_weights = edge_discrepancy.cpu().numpy()[reindex] * regularization \
        if edge_discrepancy is not None else regularization

    # Convert logits to class probabilities
    node_probas = torch.nn.functional.softmax(node_logits, dim=1)

    # Mean-center the node positions, in case coordinates are very large
    center_offset = node_pos.mean(dim=0)
    node_pos = node_pos - center_offset.view(1, -1)

    # Build the node features as the concatenation of positions and
    # class probabilities
    # TODO: this is for `improve_merge` branch, need to adapt to new
    #  interface for L2+KL formulation
    loss_type = 1
    x = torch.cat((node_pos + node_offset, node_probas), dim=1)
    x = np.asfortranarray(x.cpu().numpy().T)
    node_size = node_size.float().cpu().numpy()
    coor_weights = np.ones(num_dim + num_classes, dtype=np.float32)
    coor_weights[:num_dim] *= spatial_weight

    # Partition computation
    instance_index, x_c, cluster, edges, times = cp_kmpp_d0_dist(
        loss_type,
        x,
        source_csr,
        target,
        edge_weights=edge_weights,
        vert_weights=node_size,
        coor_weights=coor_weights,
        min_comp_weight=cutoff,
        cp_dif_tol=1e-2,
        cp_it_max=iterations,
        split_damp_ratio=0.7,
        verbose=verbose,
        max_num_threads=num_threads,
        balance_parallel_split=True,
        compute_Time=True,
        compute_List=True,
        compute_Graph=True)

    if verbose:
        delta_t = (times[1:] - times[:-1]).round(2)
        print(f'Instance partition times: {delta_t}')

    # Convert the instance_index to the input format
    instance_index = torch.from_numpy(instance_index.astype('int64')).to(device)

    return instance_index


def oracle_superpoint_clustering(
        nag,
        num_classes,
        thing_classes,
        level=1,
        k_max=30,
        radius=3,
        adjacency_mode='radius-centroid',
        centroid_mode='iou',
        centroid_level=1,
        smooth_affinity=True,
        regularization=1e-5,
        spatial_weight=1e-1,
        cutoff=1,
        parallel=True,
        iterations=10,
        trim=False,
        discrepancy_epsilon=1e-1
):
    """Compute an oracle for superpoint clustering for instance and
    panoptic segmentation. This is a proxy for the highest achievable
    graph clustering performance with the superpoint partition at hand
    and the input clustering parameters.

    The output `InstanceData` can then be used to compute final
    segmentation metrics using:
      - `InstanceData.semantic_segmentation_oracle()`
      - `InstanceData.instance_segmentation_oracle()`
      - `InstanceData.panoptic_segmentation_oracle()`

    More precisely, for the optimal superpoint clustering:
      - build the instance graph on the input `NAG` `level`-partition
      - for each edge, the oracle perfectly predicts the affinity
      - for each node, the oracle perfectly predicts the offset
      - for each node, the oracle predicts the dominant label from its
        label histogram (excluding the 'void' label)
      - partition the instance graph using the oracle edge affinities,
        node offsets and node classes
      - merge superpoints if they are assigned to the same object
      - merge 'stuff' predictions together, so that there is at most 1
        prediction of each 'stuff' class per batch item

    :param nag: NAG object
    :param num_classes: int
        Number of classes in the dataset, allows differentiating between
        valid and void classes
    :param thing_classes: List[int]
        List of labels for 'thing' classes. The remaining labels are
        either 'stuff' of 'void'
    :param level: int
        Level at which to compute the superpoint clustering
    :param k_max:
    :param radius:
    :param adjacency_mode:
    :param centroid_mode:
    :param centroid_level:
    :param smooth_affinity:
    :param regularization:
    :param spatial_weight:
    :param cutoff:
    :param parallel:
    :param iterations:
    :param trim:
    :param discrepancy_epsilon:
    :return:
    """
    # Local imports to avoid import loop errors
    from src.transforms import OnTheFlyInstanceGraph
    from src.nn.instance import InstancePartitioner

    # Instance graph computation
    nag = OnTheFlyInstanceGraph(
        level=level,
        num_classes=num_classes,
        adjacency_mode=adjacency_mode,
        k_max=k_max,
        radius=radius,
        centroid_mode=centroid_mode,
        centroid_level=centroid_level,
        smooth_affinity=smooth_affinity)(nag)

    # Prepare input for instance graph partition
    # NB: we assign only to valid classes and ignore void
    node_y = nag[1].y[:, :num_classes].argmax(dim=1)
    node_pos = nag[1].pos
    node_offset = nag[1].obj_pos - nag[1].pos
    node_logits = one_hot(node_y, num_classes=num_classes).float()
    node_size = nag.get_sub_size(1)
    edge_index = nag[1].obj_edge_index
    edge_affinity = nag[1].obj_edge_affinity

    # Set node offset to 0 for stuff and void classes
    thing_classes = torch.tensor(thing_classes, device=nag.device)
    is_thing = torch.isin(node_y, thing_classes)
    node_offset = node_offset * is_thing.view(-1, 1)

    # Instance graph partition
    instance_index = instance_cut_pursuit(
        node_pos,
        node_offset,
        node_logits,
        node_size,
        edge_index,
        edge_affinity,
        do_sigmoid_affinity=False,
        regularization=regularization,
        spatial_weight=spatial_weight,
        cutoff=cutoff,
        parallel=parallel,
        iterations=iterations,
        trim=trim,
        discrepancy_epsilon=discrepancy_epsilon)

    # For each stuff class of each batch item, merge predictions
    # together
    weighted_node_logits = softmax(
        node_logits.float(), dim=1) * node_size.view(-1, 1)
    instance_label = scatter_sum(
        weighted_node_logits, instance_index, dim=0).argmax(dim=1)
    pred_instance_label = instance_label[instance_index]

    batch = nag[1].batch if nag[1].batch is not None \
        else torch.zeros_like(instance_index)
    num_batch_items = batch.max() + 1

    is_thing = torch.isin(pred_instance_label, thing_classes)

    final_instance_index = instance_index.clone()
    final_instance_index[~is_thing] = \
        instance_index.max() + 1 \
        + pred_instance_label[~is_thing] * num_batch_items \
        + batch[~is_thing]
    final_instance_index = consecutive_cluster(final_instance_index)[0]

    # Compute the instance data by merging nodes based on instance_index
    instance_data = nag[1].obj.merge(final_instance_index)

    return instance_data


def get_stuff_mask(y, stuff_classes):
    """Helper function producing a boolean mask of size `y.shape[0]`
    indicating which of the `y` (labels if 1D or logits/probas if 2D)
    are among the `stuff_classes`.
    """
    # Get labels from y, in case y are logits
    labels = y.long() if y.dim() == 1 else y.argmax(dim=1)

    # Search the labels belonging to the set of stuff classes
    stuff_classes = torch.astensor(
        stuff_classes, dtype=labels.dtype, device=labels.device)
    return torch.isin(labels, stuff_classes)
