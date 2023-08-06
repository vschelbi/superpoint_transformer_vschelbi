import sys
import torch
import numpy as np
import os.path as osp
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from torch.nn.functional import one_hot
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from src.utils.neighbors import knn_2
from src.utils.graph import to_trimmed
from src.utils.cpu import available_cpu_count
from src.utils.scatter import scatter_mean_weighted

src_folder = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(src_folder)
sys.path.append(osp.join(src_folder, "dependencies/grid_graph/python/bin"))
sys.path.append(osp.join(src_folder, "dependencies/parallel_cut_pursuit/python/wrappers"))

from grid_graph import edge_list_to_forward_star
from cp_d0_dist import cp_d0_dist


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
    overlap between the predicted and target instances, the prediction
    cell centers are sampled near the target samples.
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


def _instance_cut_pursuit(
        node_x,
        node_logits,
        node_size,
        edge_index,
        edge_affinity_logits,
        do_sigmoid_affinity=True,
        loss_type='l2_kl',
        regularization=1e-2,
        x_weight=1,
        p_weight=1,
        cutoff=1,
        parallel=True,
        iterations=10,
        trim=False,
        discrepancy_epsilon=1e-3,
        temperature=1,
        dampening=0,
        verbose=False):
    """Partition an instance graph using cut-pursuit.

    :param node_x: Tensor of shape [num_nodes, num_dim]
        Node features
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
    :param loss_type: str
        Rules the loss applied on the node features. Accepts one of
        'l2' (L2 loss on node features and probabilities),
        'l2_kl' (L2 loss on node features and Kullback-Leibler
        divergence on node probabilities)
    :param regularization: float
        Regularization parameter for the partition
    :param x_weight: float
        Weight used to mitigate the impact of the node features in the
        partition. The larger, the lesser features importance before
        the probabilities
    :param p_weight: float
        Weight used to mitigate the impact of the node probabilities in
        the partition. The larger, the lesser features importance before
        the features
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
        `affinity=1 ⇒ discrepancy=1/discrepancy_epsilon`
    :param temperature: float
        Temperature used in the softmax when converting node logits to
        probabilities
    :param dampening: float
        Dampening applied to the node probabilities to mitigate the
        impact of near-zero probabilities in the Kullback-Leibler
        divergence
    :param verbose: bool
    :return:
    """

    # Sanity checks
    assert node_x.dim() == 2, \
        "`node_x` must have shape `[num_nodes, num_dim]`"
    assert node_logits.dim() == 2, \
        "`node_logits` must have shape `[num_nodes, num_classes]`"
    assert node_logits.shape[0] == node_x.shape[0], \
        "`node_logits` and `node_x` must have the same number of points"
    assert node_size.dim() == 1, \
        "`node_size` must have shape `[num_nodes]`"
    assert node_size.shape[0] == node_x.shape[0], \
        "`node_size` and `node_x` must have the same number of points"
    assert edge_index.dim() == 2 and edge_index.shape[0] == 2, \
        "`edge_index` must be of shape `[2, num_edges]`"
    edge_affinity_logits = edge_affinity_logits.squeeze()
    assert edge_affinity_logits.dim() == 1, \
        "`edge_affinity_logits` must be of shape `[num_edges]`"
    assert edge_affinity_logits.shape[0] == edge_index.shape[1], \
        "`edge_affinity_logits` and `edge_index` must have the same number " \
        "of edges"
    loss_type = loss_type.lower()
    assert loss_type in ['l2', 'l2_kl'], \
        "`loss_type` must be one of ['l2', 'l2_kl']"
    assert 0 < discrepancy_epsilon, \
        "`discrepancy_epsilon` must be strictly positive"
    assert 0 < temperature, "`temperature` must be strictly positive"
    assert 0 <= dampening <= 1, "`dampening` must be in [0, 1]"

    device = node_x.device
    num_nodes = node_x.shape[0]
    x_dim = node_x.shape[1]
    p_dim = node_logits.shape[1]
    dim = x_dim + p_dim
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
    edge_discrepancy = edge_affinity / (1 - edge_affinity + discrepancy_epsilon)

    # Convert edges to forward-star (or CSR) representation
    source_csr, target, reindex = edge_list_to_forward_star(
        num_nodes, edge_index.T.contiguous().cpu().numpy())
    source_csr = source_csr.astype('uint32')
    target = target.astype('uint32')
    edge_weights = edge_discrepancy.cpu().numpy()[reindex] * regularization \
        if edge_discrepancy is not None else regularization

    # Convert logits to class probabilities
    node_probas = torch.nn.functional.softmax(node_logits / temperature, dim=1)

    # Apply some dampening to the probability distributions. This brings
    # the distributions closer to a uniform distribution, limiting the
    # impact of near-zero probabilities in the Kullback-Leibler
    # divergence in the partition
    num_classes = node_probas.shape[1]
    node_probas = (1 - dampening) * node_probas + dampening / num_classes

    # Mean-center the node features, in case values have a very large
    # mean. This is optional, but favors maintaining values in a
    # reasonable float32 range
    node_x = node_x - node_x.mean(dim=0).view(1, -1)

    # Build the node features as the concatenation of positions and
    # class probabilities
    x = torch.cat((node_x, node_probas), dim=1)
    x = np.asfortranarray(x.cpu().numpy().T)
    node_size = node_size.float().cpu().numpy()

    # The `loss` term will decide which portion of `x` should be treated
    # with L2 loss and which should be treated with Kullback-Leibler
    # divergence
    l2_dim = dim if loss_type == 'l2' else x_dim

    # Weighting to apply on the features and probabilities
    coor_weights_dim = dim if loss_type == 'l2' else x_dim + 1
    coor_weights = np.ones(coor_weights_dim, dtype=np.float32)
    coor_weights[:x_dim] *= x_weight
    coor_weights[x_dim:] *= p_weight

    # Partition computation
    obj_index, x_c, cluster, edges, times = cp_d0_dist(
        l2_dim,
        x,
        source_csr,
        target,
        edge_weights=edge_weights,
        vert_weights=node_size,
        coor_weights=coor_weights,
        min_comp_weight=cutoff,
        cp_dif_tol=1e-2,
        K=4,
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

    # Convert the obj_index to the input format
    obj_index = torch.from_numpy(obj_index.astype('int64')).to(device)

    return obj_index


def instance_cut_pursuit(
        batch,
        node_x,
        node_logits,
        stuff_classes,
        node_size,
        edge_index,
        edge_affinity_logits,
        do_sigmoid_affinity=True,
        loss_type='l2_kl',
        regularization=1e-2,
        x_weight=1,
        p_weight=1,
        cutoff=1,
        parallel=True,
        iterations=10,
        trim=False,
        discrepancy_epsilon=1e-3,
        temperature=1,
        dampening=0,
        verbose=False):
    """The forward step will compute the partition on the instance
    graph, based on the node features, node logits, and edge
    affinities. The partition segments will then be further merged
    so that there is at most one instance of each stuff class per
    batch item (ie per scene).

    :param batch: Tensor of shape [num_nodes]
        Batch index of each node
    :param node_x: Tensor of shape [num_nodes, num_dim]
        Predicted node embeddings
    :param node_logits: Tensor of shape [num_nodes, num_classes]
        Predicted classification logits for each node
    :param stuff_classes: List or Tensor
        List of 'stuff' class labels. These are used for merging
        stuff segments together to ensure there is at most one
        predicted instance of each 'stuff' class per batch item
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
    :param loss_type: str
        Rules the loss applied on the node features. Accepts one of
        'l2' (L2 loss on node features and probabilities),
        'l2_kl' (L2 loss on node features and Kullback-Leibler
        divergence on node probabilities)
    :param regularization: float
        Regularization parameter for the partition
    :param x_weight: float
        Weight used to mitigate the impact of the node features in the
        partition. The larger, the lesser features importance before
        the probabilities
    :param p_weight: float
        Weight used to mitigate the impact of the node probabilities in
        the partition. The larger, the lesser features importance before
        the features
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
        `affinity=1 ⇒ discrepancy=1/discrepancy_epsilon`
    :param temperature: float
        Temperature used in the softmax when converting node logits to
        probabilities
    :param dampening: float
        Dampening applied to the node probabilities to mitigate the
        impact of near-zero probabilities in the Kullback-Leibler
        divergence
    :param verbose: bool

    :return: obj_index: Tensor of shape [num_nodes]
        Indicates which predicted instance each node belongs to
    """

    # Actual partition, returns a tensor indicating which predicted
    # object each node belongs to
    obj_index = _instance_cut_pursuit(
        node_x,
        node_logits,
        node_size,
        edge_index,
        edge_affinity_logits,
        do_sigmoid_affinity=do_sigmoid_affinity,
        loss_type=loss_type,
        regularization=regularization,
        x_weight=x_weight,
        p_weight=p_weight,
        cutoff=cutoff,
        parallel=parallel,
        iterations=iterations,
        trim=trim,
        discrepancy_epsilon=discrepancy_epsilon,
        temperature=temperature,
        dampening=dampening,
        verbose=verbose)

    # Compute the mean logits for each predicted object, weighted by
    # the node sizes
    obj_logits = scatter_mean_weighted(node_logits, obj_index, node_size)
    obj_y = obj_logits.argmax(dim=1)

    # Identify, out of the predicted objects, which are of type stuff.
    # These will need to be merged to ensure there as most one instance
    # of each stuff class in each scene
    obj_is_stuff = get_stuff_mask(obj_y, stuff_classes)

    # Distribute the object-wise labels to the nodes
    node_obj_y = obj_y[obj_index]
    node_is_stuff = obj_is_stuff[obj_index]

    # Since we only want at most one prediction of each stuff class
    # per batch item (ie per scene), we assign nodes predicted as a
    # stuff class to new indices. These new indices are built in
    # such a way that there can be only one instance of each stuff
    # class per batch item
    batch = batch if batch is not None else torch.zeros_like(obj_index)
    num_batch_items = batch.max() + 1
    final_obj_index = obj_index.clone()
    final_obj_index[node_is_stuff] = \
        obj_index.max() + 1 \
        + node_obj_y[node_is_stuff] * num_batch_items \
        + batch[node_is_stuff]
    final_obj_index, perm = consecutive_cluster(final_obj_index)

    return final_obj_index


def oracle_superpoint_clustering(
        nag,
        num_classes,
        stuff_classes,
        level=1,
        k_max=30,
        radius=3,
        adjacency_mode='radius-centroid',
        centroid_mode='iou',
        centroid_level=1,
        with_offset=False,
        with_affinity=True,
        smooth_affinity=True,
        loss_type='l2_kl',
        regularization=1e-5,
        x_weight=1e-1,
        p_weight=1e-1,
        cutoff=1,
        parallel=True,
        iterations=10,
        trim=False,
        discrepancy_epsilon=1e-1,
        temperature=1,
        dampening=0):
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
    :param stuff_classes: List[int]
        List of labels for 'stuff' classes
    :param level: int
        Level at which to compute the superpoint clustering
    :param k_max:
    :param radius:
    :param adjacency_mode:
    :param centroid_mode:
    :param centroid_level:
    :param with_offset:
    :param with_affinity:
    :param smooth_affinity:
    :param loss_type:
    :param regularization:
    :param x_weight:
    :param p_weight:
    :param cutoff:
    :param parallel:
    :param iterations:
    :param trim:
    :param discrepancy_epsilon:
    :param temperature:
    :param dampening:
    :return:
    """
    # Local imports to avoid import loop errors
    from src.transforms import OnTheFlyInstanceGraph

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
    # NB2: `instance_cut_pursuit()` expects logits, which it converts to
    # probabilities using a softmax, hence the `one_hot * 10`
    node_y = nag[1].y[:, :num_classes].argmax(dim=1)
    node_logits = one_hot(node_y, num_classes=num_classes).float() * 10
    node_size = nag.get_sub_size(1)
    edge_index = nag[1].obj_edge_index

    # Prepare edge weights. If affinities are not used, we set all edge
    # weights to 0.5
    edge_affinity = nag[1].obj_edge_affinity if with_affinity \
        else torch.full(edge_index.shape[1], 0.5, device=nag.device)

    # Prepare node position features. If offsets are used, the oracle
    # perfectly predicts the offset to the object center for each node,
    # except for stuff and void classes, whose offset is set to 0
    if with_offset:
        node_x = nag[1].obj_pos
        is_stuff = get_stuff_mask(node_y, stuff_classes)
        node_x[is_stuff] = nag[1].pos[is_stuff]
    else:
        node_x = nag[1].pos

    # For each node, recover the index of the batch item it belongs to
    batch = nag[1].batch if nag[1].batch is not None \
        else torch.zeros(nag[1].num_nodes, dtype=torch.long, device=nag.device)

    # Instance graph partition
    obj_index = instance_cut_pursuit(
        batch,
        node_x,
        node_logits,
        stuff_classes,
        node_size,
        edge_index,
        edge_affinity,
        do_sigmoid_affinity=False,
        loss_type=loss_type,
        regularization=regularization,
        x_weight=x_weight,
        p_weight=p_weight,
        cutoff=cutoff,
        parallel=parallel,
        iterations=iterations,
        trim=trim,
        discrepancy_epsilon=discrepancy_epsilon,
        temperature=temperature,
        dampening=dampening)

    # Compute the instance data by merging nodes based on obj_index
    instance_data = nag[1].obj.merge(obj_index)

    return instance_data


def get_stuff_mask(y, stuff_classes):
    """Helper function producing a boolean mask of size `y.shape[0]`
    indicating which of the `y` (labels if 1D or logits/probabilities if
    2D) are among the `stuff_classes`.
    """
    # Get labels from y, in case y are logits
    labels = y.long() if y.dim() == 1 else y.argmax(dim=1)

    # Search the labels belonging to the set of stuff classes
    stuff_classes = torch.as_tensor(
        stuff_classes, dtype=labels.dtype, device=labels.device)
    return torch.isin(labels, stuff_classes)
