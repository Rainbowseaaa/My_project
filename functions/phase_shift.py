import torch
from PIL import Image
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


if __name__ == "__main__":
    images = []
    t = transforms.ToTensor()
    for i in range(1,5):
        data_name = "C:/Users/25670/Desktop/"+f"Image00{i}.bmp"
        img = Image.open(data_name)
        img = t(img)
        images.append(img[1,:,:].squeeze())
    phi = torch.atan2(images[3]-images[1],images[0]-images[2])
    plt.figure()
    plt.imshow(phi)
    plt.show()


