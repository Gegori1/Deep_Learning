# %%
import torch

def intersection_over_union(b_predict: torch.Tensor, b_truth: torch.Tensor, format: str="cxcywh", reduction: str="sum", check_true: bool=False):
    """
    Receives bounding box predictions and ground truth
    Args:
    b_predict: torch.Tensor: Bounding box predictions.
        format: "xyxy" [x_min, y_min, x_max, y_max]
        format: "cxcywh" [x_center, y_center, width, height]
    
    reduction: str: Reduction method. "mean", "sum" or "none".
        
    b_truth: torch.Tensor: Bounding box ground truth.
    
    format: str: Format of bounding box coordinates. "xyxy" or "cxcywh".
    
    check_true: bool: Check for non-zero bounding boxes to speed up computation.
    
    Returns:
    IoU: torch.Tensor: Intersection over Union.
    
    """
    if check_true:
        # get bounding box where x, y are center of bounding box
        zero_idx = torch.where(torch.any(b_truth[..., :4] != 0, dim=-1))
        b_predict, b_truth = b_predict[zero_idx], b_truth[zero_idx]
    
    if format == "cxcywh":
        b_predict = torch.cat(
            (
                b_predict[..., :2] - b_predict[..., 2:]/2,
                b_predict[..., :2] + b_predict[..., 2:]/2), dim=-1
        )
        
        b_truth = torch.cat(
            (
                b_truth[..., :2] - b_truth[..., 2:]/2,
                b_truth[..., :2] + b_truth[..., 2:]/2), dim=-1
        )

    # compute coordinates of intersection rectangle
    x_left = torch.max(b_predict[..., 0], b_truth[..., 0])
    y_top = torch.max(b_predict[..., 1], b_truth[..., 1])
    x_right = torch.min(b_predict[..., 2], b_truth[..., 2])
    y_bottom = torch.min(b_predict[..., 3], b_truth[..., 3])
    
    # compute area of intersection rectangle
    A_inter = torch.clamp(x_right - x_left, min=0) * torch.clamp(y_bottom - y_top, min=0)
    
    # compute area of both bounding boxes
    A_predict = ((b_predict[..., 2] - b_predict[..., 0]) * (b_predict[..., 3] - b_predict[..., 1])).abs()
    A_truth = ((b_truth[..., 2] - b_truth[..., 0]) * (b_truth[..., 3] - b_truth[..., 1])).abs()
    
    # compute intersection over union
    IoU = A_inter / (A_predict + A_truth - A_inter + 1e-6)
    
    if reduction == "mean":
        IoU = torch.mean(IoU)
        
    elif reduction == "sum":
        IoU = torch.sum(IoU)
    elif reduction == "none":
        pass
    else:
        raise ValueError("reduction must be 'mean', 'sum' or 'none'")
    
    return IoU

intersection_over_union.__name__ = "intersection_over_union"

def generalized_intersection_over_union(b_predict: torch.Tensor, b_truth: torch.Tensor, format: str="cxcywh", reduction: str="sum", check_true: bool=False):
    """
    Receives bounding box predictions and ground truth
    Args:
    b_predict: torch.Tensor: Bounding box predictions.
        format: "xyxy" [x_min, y_min, x_max, y_max]
        format: "cxcywh" [x_center, y_center, width, height]
    
    reduction: str: Reduction method. "mean", "sum" or "none".
        
    b_truth: torch.Tensor: Bounding box ground truth.
    
    format: str: Format of bounding box coordinates. "xyxy" or "cxcywh".
    
    check_true: bool: Check for non-zero bounding boxes to speed up computation.
    
    Returns:
    GIoU: torch.Tensor: Intersection over Union.
    
    """
    # get bounding box where x, y are center of bounding box
    # print(b_truth.shape), print(b_predict.shape)
    if check_true:
        zero_idx = torch.where(torch.any(b_truth[..., :4] != 0, dim=-1))
        b_predict, b_truth = b_predict[zero_idx], b_truth[zero_idx]
    
    if format == "cxcywh":
        b_predict = torch.cat(
            (
                b_predict[..., :2] - b_predict[..., 2:]/2,
                b_predict[..., :2] + b_predict[..., 2:]/2), dim=-1
        )
        
        # print(b_truth.shape)
        
        b_truth = torch.cat(
            (
                b_truth[..., :2] - b_truth[..., 2:]/2,
                b_truth[..., :2] + b_truth[..., 2:]/2), dim=-1
        )
        
        # print(b_truth.shape)
        
        
        
    # # check if bounding boxes are valid
    # if torch.any(b_predict[..., 2] <= b_predict[..., 0]) or torch.any(b_predict[..., 3] <= b_predict[..., 1]):
    #     wrong_idx_width = torch.where(b_predict[..., 2] <= b_predict[..., 0])
    #     wrong_idx_height = torch.where(b_predict[..., 3] <= b_predict[..., 1])

    #     b_predict[wrong_idx_width] = torch.zeros_like(b_predict[wrong_idx_width])
    #     b_predict[wrong_idx_height] = torch.zeros_like(b_predict[wrong_idx_height])
        
    # copute coordinates of intersection rectangle
    x_left = torch.max(b_predict[..., 0], b_truth[..., 0])
    y_top = torch.max(b_predict[..., 1], b_truth[..., 1])
    x_right = torch.min(b_predict[..., 2], b_truth[..., 2])
    y_bottom = torch.min(b_predict[..., 3], b_truth[..., 3])
    
    # print(x_left.shape, y_top.shape, x_right.shape, y_bottom.shape)
    
    # compute area of intersection rectangle
    A_inter = torch.clamp(x_right - x_left, min=0) * torch.clamp(y_bottom - y_top, min=0)
    
    # print(f"A_inter: {A_inter.shape}")
    
    # compute area of both bounding boxes
    A_predict = ((b_predict[..., 2] - b_predict[..., 0]) * (b_predict[..., 3] - b_predict[..., 1])).abs()
    A_truth = ((b_truth[..., 2] - b_truth[..., 0]) * (b_truth[..., 3] - b_truth[..., 1])).abs()
    
    # print(f"A_predict: {A_predict.shape}, A_truth: {A_truth.shape}")
    
    # compute intersection over union
    Union = (A_predict + A_truth - A_inter + 1e-6).abs()
    IoU = A_inter / Union
    
    # print(f"Union: {Union.shape}, IoU: {IoU.shape}")
    
    # compute coordinates of smallest rectangle that encloses both bounding boxes
    x_left = torch.min(b_predict[..., 0], b_truth[..., 0])
    y_top = torch.min(b_predict[..., 1], b_truth[..., 1])
    x_right = torch.max(b_predict[..., 2], b_truth[..., 2])
    y_bottom = torch.max(b_predict[..., 3], b_truth[..., 3])
    
    # print(f"x_left: {x_left.shape}, y_top: {y_top.shape}, x_right: {x_right.shape}, y_bottom: {y_bottom.shape}")
    
    # compute area of smallest rectangle that encloses both bounding boxes
    A_enclose = torch.abs((x_right - x_left) * (y_bottom - y_top) + 1e-6)
    
    # print(f"A_enclose: {A_enclose.shape}")
    
    # compute generalized intersection over union
    GIoU = IoU - ((A_enclose - Union) / A_enclose)
    
    # print(f"GIoU: {GIoU.shape}")
    
    if reduction == "mean":
        GIoU = torch.mean(GIoU)
    elif reduction == "sum":
        GIoU = torch.sum(GIoU)
    elif reduction == "none":
        pass
    else:
        raise ValueError("reduction must be 'mean', 'sum' or 'none'")
    
    return GIoU

generalized_intersection_over_union.__name__ = "generalized_intersection_over_union"

# %% Example
if __name__ == '__main__':
    y_real = torch.rand(size=(3, 7, 7, 4))
    y_real[..., 3] = torch.randint(0, 2, size=(3, 7, 7))

    # y_pred = torch.rand(size=(3, 7, 7, 4))
    y_pred = y_real.clone()

    iou = intersection_over_union(
        y_pred, y_real, 
        format="cxcywh", reduction="none", check_true=True
    )
    print(iou)
    
    giou = generalized_intersection_over_union(
        y_pred, y_real,
        format="cxcywh", reduction="none", check_true=False
    )
    print(giou)
# %%
