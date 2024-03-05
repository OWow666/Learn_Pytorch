import torch
import torchvision
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64)

class Mymodule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(3, 6, 3, 1, 0)
        # 彩色图像有3通道，in_channels就是3

    def forward(self, x):
        x = self.conv1(x)  # x处理完要赋值回去，要不然相当于什么都没做
        return x

mymodule = Mymodule()
print(mymodule)

writer = SummaryWriter("../logs_Conv2d")
step = 0

for data in dataloader:
    imgs, targets = data
    output = mymodule(imgs)
    # print(imgs.shape)
    # print(output.shape)

    # torch.Size([64, 3, 32, 32])
    writer.add_images("input", imgs, step)

    # torch.Size([64, 6, 30, 30]) -> [xxx, 3, 30, 30]
    output = torch.reshape(output, (-1, 3, 30, 30))  # 不清楚第一个数是什么时候用-1
    writer.add_images("output", output, step)

    step = step + 1

writer.close()
