from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

fmnist_data = FashionMNIST(root="./data",
                           train=True, download=True,
                           transform=transforms.ToTensor())
fmnist_classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                  "Sandale", "Shirt", "Sneaker", "Bag", "Ankle boot"]
print("データの数:", len(fmnist_data))

n_image = 25 # 表示する画像の数
fmnist_loader = DataLoader(fmnist_data,
                           batch_size=n_image, shuffle=True)
for batch_idx, (images, labels) in enumerate(fmnist_loader):
    print(f"Batch{batch_idx + 1}:")
    print("Image shape:", images.shape)
    print("labels shape:", labels.shape)

img_size = 28
plt.figure(figsize=(10, 10)) # 画像のサイズ
for i in range(n_image):
    ax = plt.subplot(5, 5, i+1)
    ax.imshow(images[i].view(img_size, img_size),
              cmap="Greys_r")
    label = fmnist_classes[labels[i]]
    ax.set_title(label)
    ax.get_xaxis().set_visible(False) # 軸を非表示に
    ax.get_yaxis().set_visible(False)

plt.show()
