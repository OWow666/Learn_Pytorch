from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

'''
1、关注输入输出类型（tensor, PIL, ...）
2、看官方文档
3、关注算法所需要的参数（ctrl+p, ...）
4、不知道函数返回值的时候
    - print(...)
    - print(type(...))
5、多用Tensorboard
'''


writer = SummaryWriter("logs_p4")
img = Image.open("learn_pytorch_1/hymenoptera_data/train/ants_images/0013035.jpg")
print(img)

# ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("Totensor", img_tensor)

# Normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([1, 3, 5], [9, 3, 5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm,1)

# Resize - 1st
print(img.size)
trans_resize = transforms.Resize((512,512))
# img_resize -> resize -> img_resize (PIL)
img_resize = trans_resize(img)
# img_resize (PIL) -> totensor -> img_resize (tensor)
img_resize = trans_totensor(img_resize)
writer.add_image("Resize", img_resize, 0)

#  Compose - Resize - 2nd
trans_resize_2 = transforms.Resize(256)
# PIL -> PIL -> tensor
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2, 1)

# RandomCrop
trans_random = transforms.RandomCrop((256,300))
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCropHW", img_crop, i)


writer.close()
