class SEBlock(nn.Module):
    def __init__(self, in_c, r=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(in_c, in_c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_c // r, in_c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # 对每个通道求平均值
        y = self.fc(y).view(b, c, 1, 1)  # 通过全连接层生成权重
        return x * y  # 输入特征乘以通道权重
