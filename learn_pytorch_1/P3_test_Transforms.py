from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import cv2

# transform.ToTensor
# 1、transforms该如何使用
# 2、为什么要Tensor数据类型

# 绝对路径: C:\Users\EricW\Desktop\learn_pytorch\hymenoptera_data\train\ants_images\0013035.jpg
# 相对路径: hymenoptera_data\train\ants_images\0013035.jpg

img_path = "learn_pytorch_1/hymenoptera_data/train/ants_images/0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")


# 1、transforms该如何使用
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer.add_image("Tensor_img", tensor_img)

writer.close()
