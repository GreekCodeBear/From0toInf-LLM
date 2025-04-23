import torch
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler


# 定义模型
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


# 定义数据和优化器
data = torch.randn(32, 10).cuda()
target = torch.randn(32, 1).cuda()
model = MyModel().cuda()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 定义自动混合精度
scaler = GradScaler()

# 训练模型
for epoch in range(10):
    optimizer.zero_grad()
    with autocast():
        output = model(data)
        loss = nn.functional.mse_loss(output, target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    print("Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, 10, loss.item()))
