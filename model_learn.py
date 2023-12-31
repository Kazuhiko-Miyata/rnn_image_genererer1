import torch
from torch.utils.data import DataLoader, TensorDataset

n_time = 14  # 時刻の数
n_in = img_size  # 入力層のニューロン数
n_mid = 256  # 中間層のニューロン数
n_out = img_size  # 出力層のニューロン数
n_sample_in_img = img_size - n_time  # 1枚の画像中のサンプル数

dataloader = DataLoader(fmnist_data, batch_size=len(fmnist_data), shuffle=True)
dataiter = iter(dataloader)

for train_imgs, labels in dataiter:
    train_imgs = train_imgs.view(-1, img_size, img_size)

n_sample = len(train_imgs) * n_sample_in_img # サンプル数

input_data = torch.zeros((n_sample, n_time, n_in)) # 入力
correct_data = torch.zeros((n_sample, n_out)) # 正解
for i in range(len(train_imgs)):
    for j in range(n_sample_in_img):
       sample_id = i*n_sample_in_img + j
       input_data[sample_id] = train_imgs[i, j:j+n_time]
       correct_data[sample_id] = train_imgs[i, j+n_time]

dataset = TensorDataset(input_data, correct_data) # データセットの作成
train_loader = DataLoader(dataset, batch_size=128, shuffle=True) # DataLoaderの設定

#テスト用のデータ
n_disp = 10 # 生成し表示する画像の数

disp_data = FashionMNIST(root="./data",
                           train=False, download=True,
                           transform=transforms.ToTensor())
disp_loader = DataLoader(disp_data, batch_size=n_disp, shuffle=False)
for batch_idx, (disp_imgs, labels) in enumerate(disp_loader):
    print(f"Batch{batch_idx + 1}:")
    print("Image shape:", images.shape)
    print("labels shape:", labels.shape)
disp_imgs = disp_imgs.view(-1, img_size, img_size)

#モデルの構築
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM( # LSTM層
                     input_size=n_in, # 入力サイズ
                     hidden_size=n_mid, # ニューロン数
                     batch_first=True,
# 入力を(バッチサイズ、時刻の数、入力の数)にする
        )
        self.fc = nn.Linear(n_mid, n_out) # 全結合層

    def forward(self, x):
       # y_rnn:全時刻の出力 h:中間層の最終時刻の値 c:記憶セル
       y_rnn, (h, c) = self.rnn(x, None)
       y = self.fc(y_rnn[:, -1, :]) # yは最後の時刻の出力
       return y

net = Net()
net.cuda()  # GPU対応
print(net)

#画像生成用の関数
def generate_images():
    # オリジナルの画像
    print("original:")
    plt.figure(figsize=(20, 2))
    for i in range(n_disp):
        ax = plt.subplot(1, n_disp, i+1)
        ax.imshow(disp_imgs[i], cmap="Greys_r", vmin=0.0, vmax=1.0)
        ax.get_xaxis().set_visible(False) # 軸を非表示に
        ax.get_yaxis().set_visible(False)
    plt.show()

    # 下半分をRNNにより生成した画像
    print("Generated:")
    net.eval() # 評価モード
    gen_imgs = disp_imgs.clone()
    plt.figure(figsize=(20, 2))
    for i in range(n_disp):
       for j in range(n_sample_in_img):
           x = gen_imgs[i, j:j+n_time].view(1,
           n_time, img_size)
           x = x.cuda() # GPU対応
           gen_imgs[i, j+n_time] = net(x)[0]
       ax = plt.subplot(1, n_disp, i+1)
       ax.imshow(gen_imgs[i].detach(),
       cmap="Greys_r", vmin=0.0, vmax=1.0)
       ax.get_xaxis().set_visible(False) # 軸を非表示に
       ax.get_yaxis().set_visible(False)
    plt.show()

#学習のコード
from torch import optim

# 平均二乗誤差
loss_fnc = nn.MSELoss()

# 最適化アルゴリズム
optimizer = optim.Adam(net.parameters())

# 損失のログ
record_loss_train = []

#学習
epochs = 30 # エポック数
for i in range(epochs):
    net.train() # 訓練モード
    loss_train = 0
    for j, (x, t) in enumerate(train_loader):
    # ミニバッチを取り出す
       x, t = x.cuda(), t.cuda() # GPU対応
       y = net(x)
       loss = loss_fnc(y, t)
       loss_train += loss.item()
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
    loss_train /= j+1
    record_loss_train.append(loss_train)

    if i%5==0 or i==epochs-1:
       print("Epoch:", i, "Loss_Train:", loss_train)
       generate_images()
