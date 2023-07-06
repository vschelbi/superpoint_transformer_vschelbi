import torch
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from src.utils.neighbors import knn_2


__all__ = ['generate_random_bbox_data', 'generate_random_segment_data']


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
    predicted and and target instances, the predcition cell centers are
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
