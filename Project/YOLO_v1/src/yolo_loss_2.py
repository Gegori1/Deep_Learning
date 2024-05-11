 #%%
import torch
import torch.nn as nn
from iou_giou import intersection_over_union
from torch.nn.functional import mse_loss
# %%
class YoloLoss(nn.Module):

    def __init__(self, S=4, B=2, C=2, format="cxcywh", reduction="none"):
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")

        self.S = S
        self.B = B
        self.C = C
        
        self.format = format
        self.reduction = reduction

        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):

        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        iou_b1 = intersection_over_union(
            predictions[..., :4],
            target[..., :4],
            format=self.format,
            reduction="none"
        )
        iou_b2 = intersection_over_union(
            predictions[..., 5:9],
            target[..., :4],
            format=self.format,
            reduction="none"
        )
        
        iou_b2 = iou_b2.unsqueeze(3)
        iou_b1 = iou_b1.unsqueeze(3)
        
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        iou_maxes, bestbox = torch.max(ious, dim=0)
        exists_box = target[..., 4].unsqueeze(3)

        box_predictions = exists_box * (
            (
                bestbox * predictions[..., 5:9]
                + (1 - bestbox) * predictions[..., :4]
            )
        )

        box_targets = exists_box * target[..., :4]

        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        pred_box = (
            bestbox * predictions[..., 9:10] + (1 - bestbox) * predictions[..., 4:5]
        )

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 4:5]),
        )



        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 4:5], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 4:5], start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 9:10], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 4:5], start_dim=1)
        )


        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., 10:12], end_dim=-2,),
            torch.flatten(exists_box * target[..., 5:7], end_dim=-2,),
        )

        loss = (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )

        return loss

# %%    
    
if __name__ == '__main__':
    torch.manual_seed(0)
    S, B, C = 7, 2, 2
    predictions = torch.rand((1, 7, 7, 12))
    target = torch.rand((1, 7, 7, 7))
    target[..., 4] = torch.randint(0, 2, (1, 7, 7))
    loss = YoloLoss(S=S, B=B, C=C)
    print(loss(predictions, target))

# %%
