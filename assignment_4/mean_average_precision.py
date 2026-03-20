import torch
from decode import unpack_cube

def pairwise_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise IoU between two sets of boxes.

    :param boxes_a: Shape (N, 4) in (cx, cy, w, h) format.
    :param boxes_b: Shape (M, 4) in (cx, cy, w, h) format.
    :return: IoU matrix of shape (N, M).
    """
    cx_a, cy_a, w_a, h_a = boxes_a.unbind(-1)   # (N,)
    cx_b, cy_b, w_b, h_b = boxes_b.unbind(-1)   # (M,)

    hw_a, hh_a = w_a / 2, h_a / 2               # half-dims for a
    hw_b, hh_b = w_b / 2, h_b / 2               # half-dims for b

    # Intersection bounds directly from centers ± half-dims
    # Broadcasting: (N, 1) vs (1, M) -> (N, M)
    inter_w = (torch.minimum(cx_a[:, None] + hw_a[:, None], cx_b[None] + hw_b[None])
             - torch.maximum(cx_a[:, None] - hw_a[:, None], cx_b[None] - hw_b[None])
             ).clamp(min=0)

    inter_h = (torch.minimum(cy_a[:, None] + hh_a[:, None], cy_b[None] + hh_b[None])
             - torch.maximum(cy_a[:, None] - hh_a[:, None], cy_b[None] - hh_b[None])
             ).clamp(min=0)

    inter_area = inter_w * inter_h                              # (N, M)
    union_area = (w_a * h_a)[:, None] + (w_b * h_b)[None] - inter_area

    return inter_area / union_area.clamp(min=1e-6)

def compute_map(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean Average Precision (mAP) over all classes without any
    Python-level loops — all operations are vectorised across classes,
    predictions, and ground truths simultaneously.

    :param y_hat: Prediction cube, shape (B, S, S, 5+C).
    :param y: Target cube, same shape as y_hat.
    :param iou_threshold: IoU threshold for a detection to count as a TP.
    :param conf_threshold: Minimum objectness x class-score to keep a prediction.
    :return:
        - mAP scalar tensor
        - per-class AP tensor of shape (C,)
    """
    pred_x, pred_y, pred_w, pred_h, pred_conf, pred_cls = unpack_cube(y_hat)
    true_x, true_y, true_w, true_h, true_conf, true_cls = unpack_cube(y)

    B, S, _ = pred_x.shape
    C       = pred_cls.shape[-1]
    N       = B * S * S
    device  = y_hat.device

    # ------------------------------------------------------------------ #
    # 1. Flatten to (N, ...) tables                                       #
    # ------------------------------------------------------------------ #
    batch_idx = (
        torch.arange(B, device=device)
            .unsqueeze(1)
            .expand(B, S * S)
            .reshape(N, 1)
            .float()
    )

    pred_boxes = torch.stack(
        [pred_x, pred_y, pred_w, pred_h], dim=-1
    ).reshape(N, 4)

    gt_boxes = torch.stack(
        [true_x, true_y, true_w, true_h], dim=-1
    ).reshape(N, 4)

    pred_conf = pred_conf.reshape(N, 1)
    gt_conf   = true_conf.reshape(N)
    pred_cls  = pred_cls.reshape(N, C)
    gt_cls    = true_cls.reshape(N, C)

    # ------------------------------------------------------------------ #
    # 2. Separate predictions and ground truths                           #
    # ------------------------------------------------------------------ #
    gt_mask  = gt_conf > 0.5          # (N,)  cells that contain an object
    gt_boxes = gt_boxes[gt_mask]      # (G, 4)
    gt_cls   = gt_cls[gt_mask]        # (G, C)  one-hot class labels
    gt_batch = batch_idx[gt_mask]     # (G, 1)

    # Per-class confidence scores: objectness * class score -> (N, C)
    cls_scores = pred_conf * pred_cls                           # (N, C)

    # Keep predictions that exceed the threshold for *any* class
    keep       = (cls_scores >= conf_threshold).any(dim=-1)    # (N,)
    pred_boxes = pred_boxes[keep]     # (P, 4)
    pred_batch = batch_idx[keep]      # (P, 1)
    cls_scores = cls_scores[keep]     # (P, C)

    P = pred_boxes.shape[0]
    G = gt_boxes.shape[0]

    if P == 0 or G == 0:
        zero = torch.zeros(C, device=device)
        return zero.mean(), zero

    # ------------------------------------------------------------------ #
    # 3. Sort ALL predictions by confidence (per class simultaneously)    #
    #                                                                      #
    # Shape: (C, P) — each row is the confidence-sorted order for that   #
    # class.                                                               #
    # ------------------------------------------------------------------ #
    sort_idx   = cls_scores.T.argsort(descending=True, dim=-1)  # (C, P)
    # Gather sorted boxes/batch indices per class
    # pred_boxes_sorted: (C, P, 4),  pred_batch_sorted: (C, P, 1)
    pred_boxes_sorted = pred_boxes[sort_idx]                     # (C, P, 4)
    pred_batch_sorted = pred_batch[sort_idx]                     # (C, P, 1)

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
    iou_sorted  = iou_mat[sort_idx]                              # (C, P, G)
    batch_pred  = pred_batch_sorted                              # (C, P, 1)
    batch_gt    = gt_batch.T.unsqueeze(0)                        # (1, 1, G)
    batch_mask  = batch_pred == batch_gt                         # (C, P, G)

    # GT class mask: GT g belongs to class c if gt_cls[g, c] == 1
    # gt_cls_mask: (C, 1, G)
    gt_cls_mask = gt_cls.T.unsqueeze(1)                          # (C, 1, G)

    # Zero out cross-image and cross-class IoU entries
    iou_sorted  = iou_sorted * batch_mask * gt_cls_mask          # (C, P, G)

    # Best GT match per prediction
    best_iou, best_gt = iou_sorted.max(dim=-1)                   # (C, P)
    is_match          = best_iou >= iou_threshold                 # (C, P)

    # One-hot encode the matched GT index: (C, P, G)
    match_matrix = torch.zeros(C, P, G, device=device)
    c_idx, p_idx = is_match.nonzero(as_tuple=True)
    match_matrix[c_idx, p_idx, best_gt[c_idx, p_idx]] = 1.0

    # Deduplicate: each GT matched at most once (first/highest-conf hit wins)
    # cumsum along P axis; any entry where cumsum > 1 is a duplicate
    cum_matches  = match_matrix.cumsum(dim=1)                    # (C, P, G)
    valid_match  = (cum_matches <= 1.0) & (match_matrix == 1.0)  # (C, P, G)

    tp = valid_match.any(dim=-1).float()                         # (C, P)

    # ------------------------------------------------------------------ #
    # 6. Precision-recall curve & AP via trapezoid — all classes at once  #
    # ------------------------------------------------------------------ #
    cum_tp = tp.cumsum(dim=-1)                                   # (C, P)
    cum_fp = (1.0 - tp).cumsum(dim=-1)

    precision = cum_tp / (cum_tp + cum_fp).clamp(min=1e-6)       # (C, P)

    # GT counts per class for recall denominator: (C,)
    gt_per_class = gt_cls.sum(dim=0)                             # (C,)
    recall = cum_tp / gt_per_class.unsqueeze(-1).clamp(min=1e-6) # (C, P)

    # Prepend (recall=0, precision=1) sentinel to each class curve
    ones  = torch.ones(C, 1, device=device)
    zeros = torch.zeros(C, 1, device=device)
    precision = torch.cat([ones,  precision], dim=-1)            # (C, P+1)
    recall    = torch.cat([zeros, recall],    dim=-1)            # (C, P+1)

    ap_per_class = torch.trapezoid(precision, recall, dim=-1).abs()  # (C,)
    classes_with_gt = gt_per_class > 0

    return ap_per_class[classes_with_gt].mean()
