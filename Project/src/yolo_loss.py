# %%
import torch
from torch.nn import Module
from torch.nn.functional import mse_loss
from iou_giou import generalized_intersection_over_union

class YoloLoss(Module):
    def __init__(
        self,
        lambda_center: int=5,
        lambda_obj: int=1,
        lambda_noobj: int=0.5,
        lambda_wh: int=5,
        lambda_class: int=1,
        B: int=2, 
        S: int=7, 
        format="cxcywh"
    ):
        super().__init__()
        self.lambda_center = lambda_center
        self.lambda_noobj = lambda_noobj
        self.lambda_obj = lambda_obj
        self.lambda_wh = lambda_wh
        self.lambda_class = lambda_class
        self.B = B
        self.S = S
        self.format = format
        self.reduction = "sum"

    def forward(self, y_pred, y_true):
        # reshape y_pred (N, S*S*5*B) (N, S, S, 5*B)
        y_pred = y_pred.view(-1, self.S, self.S, (self.B*5 + self.B*2))
        # print(f"y_pred: {y_pred.shape}")
        # index for places where object exists
        obj_mask = y_true[..., 4] == 1
        noobj_mask = y_true[..., 4] == 0
        
        box_1, box_2 = y_pred[..., :5], y_pred[..., 5:10]
        
        class_1, class_2 = y_pred[..., 10:12], y_pred[..., 12:]
        # print(f"class_1: {class_1.shape}, class_2: {class_2.shape}")
        
        # print(box_1[..., :4].shape, y_true[..., :4].shape)
        iou_b1 = generalized_intersection_over_union(
            box_1[..., :4],
            y_true[..., :4],
            format=self.format,
            reduction="none"
        )
        # print(f"iou_b1: {iou_b1.shape}")
        # print(box_2[..., :4].shape, y_true[..., :4].shape)
        iou_b2 = generalized_intersection_over_union(
            box_2[..., :4],
            y_true[..., :4],
            format=self.format,
            reduction="none"
        )
        # print(f"iou_b2: {iou_b2.shape}")
        
        
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        # print(f"ious: {ious.shape}")
        best = torch.argmax(ious, dim=0).unsqueeze(-1)
        # print(f"best: {best.shape}")

        
        # Compute the loss for the center coordinates
        center_box = best * box_1[..., :2] + (1 - best) * box_2[..., :2]
        
        center_loss = mse_loss(
            center_box,
            y_true[..., :2],
            reduction=self.reduction
        )
        
        # Compute the loss for the width and height coordinates
        wh_box = (
            torch.sign(box_1[..., 2:4]) * best * torch.abs(box_1[..., 2:4] + 1e-6).sqrt()
            + 
            torch.sign(box_2[..., 2:4]) * (1 - best) * torch.abs(box_2[..., 2:4] + 1e-6).sqrt()
        )
        
        wh_real = torch.sign(y_true[..., 2:4]) * y_true[..., 2:4].sqrt()
        
        # print(f"wh_box: {wh_box.shape}")
        
        wh_loss = mse_loss(
            wh_box,
            wh_real,
            reduction=self.reduction
        )
        
        # print(f"wh_loss: {wh_loss}")
        # print(f"best: {best.shape}, box_1[4]: {box_1[..., 4:].shape}, box_2[4]: {box_2[..., 4].shape}")
        
        # compute the loss for the class probabilities where the object exists
        prob_box = best * box_1[..., 4:] + (1 - best) * box_2[..., 4:]
        # print(f"prob_box: {prob_box.shape}")
        
        # print(f"filtered: {prob_box[obj_mask].shape}")
        cls_obj_loss = mse_loss(
            prob_box[obj_mask],
            y_true[..., 4:5][obj_mask],
            reduction=self.reduction
        )
        
        class_box = best * class_1 + (1 - best) * class_2
        # print(f"class_box: {class_box.shape}")
        
        # print(cls_obj_loss)
        
        # cls_noobj_loss = mse_loss(
        #     prob_box[noobj_mask],
        #     y_true[..., 4:][noobj_mask],
        #     reduction=self.reduction
        # )
        
        cls_noobj_loss = mse_loss(
            box_1[..., 4:5][noobj_mask],
            y_true[..., 4:5][noobj_mask],
            reduction=self.reduction
        )
        
        cls_noobj_loss += mse_loss(
            box_2[..., 4:5][noobj_mask],
            y_true[..., 4:5][noobj_mask],
            reduction=self.reduction
        )
        
        
        cls_class_loss = mse_loss(
            class_box[obj_mask],
            y_true[..., 5:][obj_mask],
            reduction=self.reduction
        )
        
        # print(f"class_box {class_box.shape}, {y_true[..., 5:].shape}")
        
        # print(cls_noobj_loss)
        
        # return obj_mask, wh_box
        
        return (
            self.lambda_center * center_loss +
            self.lambda_wh * wh_loss +
            self.lambda_obj * cls_obj_loss +
            self.lambda_noobj * cls_noobj_loss +
            self.lambda_class * cls_class_loss
        )
        

# %%        
if __name__ == '__main__':
    
    torch.manual_seed(32)
    y_pred = torch.rand((3, 7 * 7 * 14))
    y_true = torch.rand((3, 7, 7, 7))
    y_true[..., 4] = torch.randint(0, 2, (3, 7, 7))
    
    loss = YoloLoss()
    out = loss(y_pred, y_true)
    print(out)
# %%
