import torch
import torch.nn as nn

from decode import unpack_cube

class YOLOv1Loss(nn.Module):
    def __init__(self, lambda_coord: float, lambda_noobj: float):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj  = lambda_noobj

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor)-> torch.Tensor:
        """
        y_hat: (N, S, S, 7)  — model predictions
        y:     (N, S, S, 7)  — ground truth

        Tensor layout per cell (7 values, B=1, C=2):
          [x, y, w, h, conf, class0, class1]
           0  1  2  3   4      5       6
        """
        pred_x, pred_y, pred_w, pred_h, pred_conf, pred_cls = unpack_cube(
            y_hat
        )
        true_x, true_y, true_w, true_h, true_conf, true_cls = unpack_cube(y)


        obj_mask  = true_conf        # 1_ij^obj  — shape (N, S, S)
        noobj_mask = 1.0 - obj_mask  # 1_ij^noobj

        # --- loss 1 & 2: coordinate losses (only where object exists) ---
        loss_xy = self.lambda_coord * (obj_mask * (
            (pred_x - true_x) ** 2 +
            (pred_y - true_y) ** 2
        )).sum()

        loss_wh = self.lambda_coord * (obj_mask * (
            (pred_w.abs().sqrt() - true_w.sqrt()) ** 2 +   # sqrt as in paper
            (pred_h.abs().sqrt() - true_h.sqrt()) ** 2
        )).sum()

        # --- loss 3: confidence loss where object exists ---
        loss_conf_obj = (obj_mask * (pred_conf - true_conf) ** 2).sum()

        # --- loss 4: confidence loss where NO object exists ---
        loss_conf_noobj = self.lambda_noobj * (
            noobj_mask * (pred_conf - true_conf) ** 2
        ).sum()

        # --- loss 5: class probability loss (only where object exists) ---
        loss_cls = (obj_mask.unsqueeze(-1) * (pred_cls - true_cls) ** 2).sum()

        total_loss = loss_xy + loss_wh + loss_conf_obj + loss_conf_noobj + loss_cls
        return total_loss
