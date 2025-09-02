import torch
import numpy as np
from sklearn.metrics import average_precision_score

def dice_coef(groundtruth_masks: torch.Tensor, pred_masks: torch.Tensor) -> float:
    """
    Computes the Dice coefficient for a batch of predictions and targets.
    
    Args:
        groundtruth_mask (torch.Tensor): Ground-truth labels (binary), shape (batch_size, height, width)
        pred_mask (torch.Tensor): Model predictions (binary), shape (batch_size, height, width)
           
    Returns:
        float: The Dice coefficient for the batch.
    """
    dice = 0.0
    for i in range(groundtruth_masks.shape[0]):
        groundtruth_mask = groundtruth_masks[i]
        pred_mask = pred_masks[i]
        if groundtruth_mask.sum() == 0 and pred_mask.sum() == 0:
            dice = 1.0  # perfect match of empty mask
        else:
            tp = torch.sum(pred_mask * groundtruth_mask)
            fp = torch.sum(pred_mask * (1 - groundtruth_mask))
            fn = torch.sum((1 - pred_mask) * groundtruth_mask)
            dice += (2 * tp) / (2 * tp + fp + fn)
    dice /= groundtruth_masks.shape[0]
    
    return dice

def calculate_iou(groundtruth_masks: torch.Tensor, pred_masks: torch.Tensor) -> float:
    """
    Computes the Intersection over Union (IoU) for a batch of predictions and targets.
    
    Args:
        groundtruth_mask (torch.Tensor): Ground-truth labels (binary), shape (batch_size, height, width)
        pred_mask (torch.Tensor): Model predictions (binary), shape (batch_size, height, width)
           
    Returns:
        float: The IoU for the batch.
    """
    iou = 0.0
    for i in range(groundtruth_masks.shape[0]):
        groundtruth_mask = groundtruth_masks[i]
        pred_mask = pred_masks[i]
        if groundtruth_mask.sum() == 0 and pred_mask.sum() == 0:
            iou = 1.0  # perfect match of empty mask
        else:
            tp = torch.sum(pred_mask * groundtruth_mask)
            fp = torch.sum(pred_mask * (1 - groundtruth_mask))
            fn = torch.sum((1 - pred_mask) * groundtruth_mask)
            iou += tp / (tp + fp + fn)
    iou /= groundtruth_masks.shape[0]

    return iou

def calculate_average_precision(predictions: torch.Tensor, targets: torch.Tensor, threshold=0.5) -> float:
    """
    Computes the average precision (AP) for a batch of predictions and targets.
    
    Args:
        predictions (torch.Tensor): Model predictions (logits or probabilities), shape (batch_size, )
        targets (torch.Tensor): Ground-truth labels (binary), shape (batch_size, )
    
    Returns:
        float: The average precision score for the batch.
    """

    aps = []
    for i in range(predictions.shape[0]):
        prediction = predictions[i].detach().cpu().numpy().flatten()
        target = targets[i].detach().cpu().numpy().flatten()
        target = (target >= threshold).astype(int)
        aps.append(average_precision_score(target, prediction))

    ap = np.mean(aps)

    return ap