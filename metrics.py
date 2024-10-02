import torch
import torch.nn as nn
from torch.nn import functional as F

weights = torch.tensor(
    [
        0.011328939365293219,
        0.4646857304592822,
        0.13541345968711663,
        0.3885718704883079
    ],
    dtype=torch.float32,
    device='cuda:0'
)


def iou_score_m(y_pred, y_true, smooth=1e-5):
    """
    y_pred: Tensor of shape (batch_size, num_classes, height, width)
    y_true: Tensor of shape (batch_size, num_classes, height, width)

    Returns:
    - IoU for each class and mean IoU across classes.
    """

    # Apply softmax for multi-class prediction
    y_pred = torch.softmax(y_pred, dim=1)  # (batch_size, num_classes, height, width)

    # Flatten the tensors to compute per-class IoU
    y_pred = y_pred.view(-1, y_pred.shape[1])  # (N, num_classes)
    y_true = y_true.view(-1, y_true.shape[1])  # (N, num_classes)

    # Compute intersection and union per class
    intersection = (y_pred * y_true).sum(dim=0)  # Sum over all pixels (N, num_classes)
    union = (y_pred + y_true).sum(dim=0) - intersection  # Sum over union (N, num_classes)

    iou = (intersection + smooth) / (union + smooth)  # Per class IoU

    mean_iou = iou.mean()  # Mean IoU across classes

    return iou, mean_iou


def iou_score(y_pred, y_true):
    y_pred = torch.sigmoid(y_pred)

    y_pred = y_pred.data.cpu().numpy()
    y_true = y_true.data.cpu().numpy()

    y_pred = y_pred > 0.5
    y_true = y_true > 0.5
    intersection = (y_pred & y_true).sum()
    union = (y_pred | y_true).sum()

    if union != 0:
        return intersection / union
    else:
        return 0


def dice_score_multiclass(y_pred, y_true, smooth=1e-5, class_weights=weights):
    """
    计算多分类的 Dice 系数。
    y_pred: 模型的预测输出，形状为 (batch_size, num_classes, height, width)。
    y_true: 真实标签，形状为 (batch_size, num_classes, height, width)。
    class_weights: 可选，类别的权重列表，用于加权平均 Dice。
    """
    # 使用 softmax 激活函数并取类别索引
    y_pred = torch.softmax(y_pred, dim=1)

    # 将 y_pred 转为 one-hot 编码（每个像素的预测属于一个类别）
    y_pred_onehot = torch.argmax(y_pred, dim=1)
    y_pred_onehot = F.one_hot(y_pred_onehot, num_classes=y_pred.shape[1]).permute(0, 3, 1, 2).float()

    # 初始化总 Dice 系数
    dice_total = 0.0
    num_classes = y_pred.shape[1]

    # 计算每个类别的 Dice 系数
    for c in range(num_classes):
        y_pred_flat = y_pred_onehot[:, c].contiguous().view(-1)
        y_true_flat = y_true[:, c].contiguous().view(-1)

        intersection = (y_pred_flat * y_true_flat).sum()
        dice_class = (2. * intersection + smooth) / (y_pred_flat.sum() + y_true_flat.sum() + smooth)

        # 考虑类别权重
        if class_weights is not None:
            dice_total += dice_class * class_weights[c]
        else:
            dice_total += dice_class

    # 返回加权平均或普通平均的 Dice 系数
    if class_weights is not None:
        return dice_total / sum(class_weights)
    else:
        return dice_total / num_classes


def dice_score(y_pred, y_true, smooth=1e-5):
    y_pred = torch.sigmoid(y_pred)

    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    intersection = (y_pred * y_true).sum()

    return (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        global weights
        dice = dice_score(y_pred, y_true, smooth=1e-3)
        dice_loss = 1 - dice
        criterion = nn.CrossEntropyLoss(weight=weights)
        ce_loss = criterion(y_pred, y_true)

        return dice_loss + ce_loss
