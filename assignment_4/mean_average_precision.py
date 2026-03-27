import torch
from decode import decode_predictions


def pairwise_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise IoU between two sets of boxes.

    :param boxes_a: Box parameters in (N, cx, cy, w, h) format.
    :type boxes_a: torch.Tensor
    :param boxes_b: Box parameters in (M, cx, cy, w, h) format.
    :type boxes_b: torch.Tensor
    :return: IoU matrix of shape (N, M).
    :rtype: torch.Tensor
    """
    cx_a, cy_a, w_a, h_a = boxes_a.unbind(-1)
    cx_b, cy_b, w_b, h_b = boxes_b.unbind(-1)

    half_w_a, half_h_a = w_a / 2, h_a / 2
    half_w_b, half_h_b = w_b / 2, h_b / 2

    # Find the width of the intersection by finding the minimal right sides of
    # the boxes when comparing boxes_a and boxes_b and the maximal left sides.
    # If there is a negative result set this to 0.
    inter_w = (
        torch.minimum(
            cx_a[:, None] + half_w_a[:, None], cx_b[None] + half_w_b[None]
        ) - torch.maximum(
            cx_a[:, None] - half_w_a[:, None], cx_b[None] - half_w_b[None]
        )
    ).clamp(min=0)
    
    # Find the height of the intersection by finding the minimal top sides of
    # the boxes when comparing boxes_a and boxes_b and the maximal bottom
    # sides. If there is a negative result set this to 0.
    inter_h = (
        torch.minimum(
            cy_a[:, None] + half_h_a[:, None], cy_b[None] + half_h_b[None]
        ) - torch.maximum(
            cy_a[:, None] - half_h_a[:, None], cy_b[None] - half_h_b[None]
        )
    ).clamp(min=0)

    inter_area = inter_w * inter_h
    union_area = (w_a * h_a)[:, None] + (w_b * h_b)[None] - inter_area

    # clamp union to prevent zero division
    return inter_area / union_area.clamp(min=1e-6)

def compute_map(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    iou_threshold: float,
    conf_threshold: float
) -> torch.Tensor:
    """
    
    """
    pred_x, pred_y, pred_w, pred_h, pred_conf, pred_cls = \
        decode_predictions(y_hat)
    true_x, true_y, true_w, true_h, true_conf, true_cls = decode_predictions(y)
    print(f"{true_x.shape = }")
    print(f"{true_y.shape = }")
    print(f"{true_w.shape = }")
    print(f"{true_h.shape = }")
    print(f"{true_conf.shape = }")
    print(f"{true_cls.shape = }")
    exit()

    batches, grid_size, _ = pred_x.shape
    C = pred_cls.shape[-1]
    batch_grid_cells = batches * grid_size * grid_size
    device = y_hat.device

    # Keep track of which image each cell came from.
    batch_idx = (
        torch.arange(
            batches,
            device=device
        ).unsqueeze(1).expand(batches, grid_size * grid_size).reshape(batch_grid_cells, 1).float()
    )

    pred_boxes = torch.stack(
        [pred_x, pred_y, pred_w, pred_h],
        dim=-1
    ).reshape(batch_grid_cells, 4)

    gt_boxes = torch.stack(
        [true_x, true_y, true_w, true_h],
        dim=-1
    ).reshape(batch_grid_cells, 4)

    pred_conf = pred_conf.reshape(batch_grid_cells, 1)
    gt_conf = true_conf.reshape(batch_grid_cells)
    pred_cls = pred_cls.reshape(batch_grid_cells, C)
    gt_cls = true_cls.reshape(batch_grid_cells, C)

    # ------------------------------------------------------------------ #
    # 2. Separate predictions and ground truths                          #
    # ------------------------------------------------------------------ #
    gt_mask = gt_conf > 0.5
    gt_boxes = gt_boxes[gt_mask]
    gt_cls = gt_cls[gt_mask]
    gt_batch = batch_idx[gt_mask]

    # Per-class confidence scores: objectness * class score -> (N, C)
    cls_scores = pred_conf * pred_cls

    # Keep predictions that exceed the threshold for *any* class
    keep = pred_conf.squeeze(-1) >= conf_threshold
    pred_boxes = pred_boxes[keep]
    pred_batch = batch_idx[keep]
    cls_scores = cls_scores[keep]

    P = pred_boxes.shape[0]
    G = gt_boxes.shape[0]

    # If there are no predictions or ground truths left return mAP of zero.
    if P == 0 or G == 0:
        zero = torch.zeros(C, device=device)
        return zero.mean()

    # ------------------------------------------------------------------ #
    # 3. Sort ALL predictions by confidence (per class simultaneously)   #
    #                                                                    #
    # Shape: (C, P) — each row is the confidence-sorted order for that   #
    # class.                                                             #
    # ------------------------------------------------------------------ #
    sort_idx = cls_scores.T.argsort(descending=True, dim=-1)
    # Gather sorted boxes/batch indices per class
    pred_batch_sorted = pred_batch[sort_idx]

    # ------------------------------------------------------------------ #
    # 4. Pairwise IoU: (P, G) — class-agnostic, computed once            #
    # ------------------------------------------------------------------ #
    iou_mat = pairwise_iou(pred_boxes, gt_boxes)

    # ------------------------------------------------------------------ #
    # 5. Build (C, P, G) match tensors                                    #
    # ------------------------------------------------------------------ #
    # Re-index IoU and batch matrices into confidence-sorted order
    # iou_sorted:   (C, P, G)
    # batch_mask:   (C, P, G) — True only when pred and GT share a batch
    iou_sorted = iou_mat[sort_idx]
    batch_pred = pred_batch_sorted
    batch_gt = gt_batch.T.unsqueeze(0)
    batch_mask = batch_pred == batch_gt

    # GT class mask: GT g belongs to class c if gt_cls[g, c] == 1
    # gt_cls_mask: (C, 1, G)
    gt_cls_mask = gt_cls.T.unsqueeze(1)

    # Zero out cross-image and cross-class IoU entries
    iou_sorted = iou_sorted * batch_mask * gt_cls_mask

    # Best GT match per prediction
    best_iou, best_gt = iou_sorted.max(dim=-1)
    is_match = best_iou >= iou_threshold

    # One-hot encode the matched GT index: (C, P, G)
    match_matrix = torch.zeros(C, P, G, device=device)
    c_idx, p_idx = is_match.nonzero(as_tuple=True)
    match_matrix[c_idx, p_idx, best_gt[c_idx, p_idx]] = 1.0

    # Deduplicate: each GT matched at most once (first/highest-conf hit wins)
    # cumsum along P axis; any entry where cumsum > 1 is a duplicate
    cum_matches  = match_matrix.cumsum(dim=1)
    valid_match  = (cum_matches <= 1.0) & (match_matrix == 1.0)

    tp = valid_match.any(dim=-1).float()

    # ------------------------------------------------------------------ #
    # 6. Precision-recall curve & AP via trapezoid — all classes at once  #
    # ------------------------------------------------------------------ #
    cum_tp = tp.cumsum(dim=-1)
    cum_fp = (1.0 - tp).cumsum(dim=-1)

    precision = cum_tp / (cum_tp + cum_fp).clamp(min=1e-6)

    # GT counts per class for recall denominator: (C,)
    gt_per_class = gt_cls.sum(dim=0)
    recall = cum_tp / gt_per_class.unsqueeze(-1).clamp(min=1e-6)

    # Prepend (recall=0, precision=1) sentinel to each class curve
    ones  = torch.ones(C, 1, device=device)
    zeros = torch.zeros(C, 1, device=device)
    precision = torch.cat([ones,  precision], dim=-1)
    recall = torch.cat([zeros, recall], dim=-1)

    ap_per_class = torch.trapezoid(precision, recall, dim=-1).abs()
    classes_with_gt = gt_per_class > 0

    return ap_per_class[classes_with_gt].mean()
