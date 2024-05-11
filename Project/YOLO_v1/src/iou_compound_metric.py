# %%
import torch
from iou_giou import (
    generalized_intersection_over_union,
    intersection_over_union
)
from torcheval.metrics.functional import binary_accuracy


class CompoundMetric:
    def __init__(self, S, B, C, format="cxcywh", reduction="mean", lambda_box=1, lambda_prob=1, lambda_class=1):
        self.S = S
        self.B = B
        self.C = C
        self.lbox = lambda_box
        self.lprob = lambda_prob
        self.lclass = lambda_class
        self.format = format
        self.reduction = reduction
        self.__name__ = "iou_yolo_max_box"

    def __call__(self, y_pred, y_true):
        y_pred = y_pred.view(-1, self.S, self.S, (self.B*5+self.C))
        box_1, box_2 = y_pred[..., :4], y_pred[..., 5:9]
        prob_1, prob_2 = y_pred[..., 4], y_pred[..., 9]
        class_1 = y_pred[..., 10:12]
        
        true_box = y_true[..., :4]
        
        iou_b1 = intersection_over_union(
            box_1,
            true_box,
            format=self.format,
            reduction=self.reduction,
            check_true=True
        )
        
        iou_b2 = intersection_over_union(
            box_2,
            true_box,
            format=self.format,
            reduction=self.reduction,
            check_true=True
        )
        
        prob_idx = torch.argmax(torch.stack([iou_b1, iou_b2], dim=0), dim=0)
        prob = torch.stack([prob_1, prob_2], dim=0)
        
        prob = prob[prob_idx]
        # class_ = class_[prob_idx]
        
        prob_1 = prob_1.flatten(start_dim=0)
        prob_real = y_true[..., 4].flatten(start_dim=0)
        prob_acc = binary_accuracy(prob_1, prob_real)
        
        class_1 = class_1.flatten(start_dim=0)
        class_real = y_true[..., 5:].flatten(start_dim=0)
        class_acc = binary_accuracy(class_1, class_real)
        
        return (
            prob_acc * self.lprob + 
            torch.max(iou_b1, iou_b2) * self.lbox + 
            class_acc * self.lclass
        )
    
# %%        
if __name__ == '__main__':
    torch.manual_seed(0)
    # y_pred = torch.rand((3, 7 * 7 * 10))
    y_true = torch.rand((1, 7, 7, 7))
    y_true[..., 4] = torch.randint(0, 2, (1, 7, 7))
    # y_pred_1 = y_true.clone().view(-1)
    # y_pred_2 = y_true.clone().view(-1)
    # y_pred = torch.concat([y_pred_1, y_pred_2], dim=0)
    y_pred = torch.rand((1, 7, 7, 12))
    
    loss = CompoundMetric(S=7, B=2, C=2)
    out = loss(y_pred, y_true)
    print(out)
# %%
