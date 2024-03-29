from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
image_path = "learn_pytorch_1/hymenoptera_data/train/ants/0013035.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
# print(type(img_array))

writer.add_image("test", img_array, 1, dataformats="HWC")

# y=x
for i in range(100):
    writer.add_scalar("y=x", i, i)

writer.close()