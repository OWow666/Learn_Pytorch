import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Preparing dataset
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False, num_workers=0, drop_last=True)

# Test the first img in the dataset
img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("logs_p6")

for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        # print(imgs.shape)
        # print(targets.shape)
        writer.add_images("Epoch: {}".format(epoch), imgs, step)
        step += 1

writer.close()
