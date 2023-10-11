# train.py
import torch
from torch.utils.data import DataLoader
from dataset import ISICDataset  # 确保这一行正确地导入了您的数据集类
from modules import MaskRCNNModule

# 初始化模型
mask_rcnn_model = MaskRCNNModule(pretrained=False)
model = mask_rcnn_model.get_model()
model.train()

# 创建数据加载器
train_dataset = ISICDataset(path="E:/comp3710/ISIC2018", type="Training")
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# 定义优化器和损失函数（如果需要的话）
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 训练循环
for epoch in range(10):  # 假设我们训练10个epoch
    for images, targets in train_loader:
        # 模型预测
        loss_dict = model(images, targets)

        # 计算总损失
        losses = sum(loss for loss in loss_dict.values())

        # 清除之前的梯度
        optimizer.zero_grad()

        # 反向传播
        losses.backward()

        # 更新权重
        optimizer.step()

        print(f"Loss: {losses.item()}")

    print(f"Epoch {epoch + 1} completed")

# 保存模型
torch.save(model.state_dict(), "mask_rcnn_model.pth")
