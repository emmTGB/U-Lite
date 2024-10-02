import torch
import os

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch

from dataset import myData
from metrics import iou_score, dice_score, dice_score_multiclass, iou_score_m
from models.ULite import ULite

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
model = ULite()


class Segmentor(pl.LightningModule):
    def __init__(self, model=model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def test_step(self, batch, batch_idx):
        image, y_true = batch
        y_pred = self.model(image)
        dice = dice_score(y_pred, y_true)
        iou = iou_score(y_pred, y_true)
        iou_list, miou = iou_score_m(y_pred, y_true)
        dice2 = dice_score_multiclass(y_pred, y_true)
        metrics = {"Test Dice": dice, "Test Iou": iou,
                   "1 iou": iou_list[0],
                   "2 iou": iou_list[1],
                   "3 iou": iou_list[2],
                   "4 iou": iou_list[3],
                   "miou": miou,
                   "dice2": dice2,
                   }
        self.log_dict(metrics, prog_bar=True)
        return metrics


# For visualization
def visualize_prediction(model, dataset, nums):
    plt.figure(figsize=(6, 2 * nums), layout='compressed')
    for idx in range(nums):
        x, y = dataset[811 + idx]
        y_pred = model(x.unsqueeze(dim=0)).data.squeeze()

        # 将预测结果转换为类别索引
        y_pred = torch.argmax(y_pred, dim=0)

        # convert torch to numpy
        x = x.permute(1, 2, 0).numpy()
        y = y.numpy()  # 真实标签
        y_pred = y_pred.numpy()  # 预测结果

        # 确保 y 和 y_pred 都是二维数组
        if y.ndim > 2:
            y = y[0]
        if y_pred.ndim > 2:
            y_pred = y_pred[0]

        # 可视化
        plt.subplot(nums, 3, 3 * idx + 1)
        plt.title(f"Image {idx + 1}")
        plt.imshow(x, cmap='gray')
        plt.axis('off')

        plt.subplot(nums, 3, 3 * idx + 2)
        plt.title(f"Ground truth {idx + 1}")
        plt.imshow(y, cmap='gray')
        plt.axis('off')

        plt.subplot(nums, 3, 3 * idx + 3)
        plt.title(f"Prediction {idx + 1}")
        plt.imshow(y_pred, cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# For visualization
# def visualize_prediction(model, dataset, nums):
#     plt.figure(figsize=(6, 2 * nums), layout='compressed')
#     for idx in range(nums):
#         x, y = dataset[680 + idx]
#         y_pred = model(x.unsqueeze(dim=0)).data.squeeze()
#         print(y_pred.shape)
#
#         # 将预测结果转换为类别索引
#         y_pred = torch.argmax(y_pred, dim=0)  # 找到每个像素的最大类别索引
#         # y_pred = y_pred.numpy()  # 预测结果
#
#
#         # 确保 y 和 y_pred 都是二维数组
#         y_pred = y_pred.squeeze().numpy()  # 转为numpy数组
#         y_pred_mapped = np.zeros_like(y_pred)
#
#         # 映射 0, 1, 2, 3 到适当的灰度级别
#         y_pred_mapped[y_pred == 0] = 0  # 黑色
#         y_pred_mapped[y_pred == 1] = 85  # 灰色
#         y_pred_mapped[y_pred == 2] = 170  # 更亮的灰色
#         y_pred_mapped[y_pred == 3] = 255  # 白色
#
#
#         # convert torch to numpy
#         x = x.permute(1, 2, 0).numpy()  # 输入图像
#         y = y.squeeze().numpy()  # 真实标签
#         # 确保 y 和 y_pred 都是二维数组
#         if y.ndim > 2:
#             y = y[0]  # 选择第一个通道
#         if y_pred.ndim > 2:
#             y_pred = y_pred[0]  # 选择第一个通道
#
#         # visualization
#         plt.subplot(nums, 3, 3 * idx + 1)
#         plt.title(f"Image {idx + 1}")
#         plt.imshow(x, cmap='gray')
#         plt.axis('off')
#
#         plt.subplot(nums, 3, 3 * idx + 2)
#         plt.title(f"Ground truth {idx + 1}")
#         plt.imshow(y, cmap='gray')
#         plt.axis('off')
#
#         plt.subplot(nums, 3, 3 * idx + 3)
#         plt.title(f"Prediction {idx + 1}")
#         plt.imshow(y_pred_mapped, cmap='gray')  # 显示类别索引图
#         plt.axis('off')
#
#     plt.tight_layout()
#     plt.show()


if __name__ == '__main__':
    DATA_PATH = './data.npz'
    CHECKPOINT_PATH = './content/weights/ckpt0.9483.ckpt'

    # Dataset & Data Loader
    test_dataset = myData(type='test', data_path=DATA_PATH, transform=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, num_workers=2, shuffle=False)

    # Prediction
    trainer = pl.Trainer()
    segmentor = Segmentor.load_from_checkpoint(CHECKPOINT_PATH)
    trainer.test(segmentor, test_loader)

    # Visualization
    visualize_prediction(model=model, dataset=test_dataset, nums=5)
