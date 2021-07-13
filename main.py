from PIL import Image
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

import torch.nn as nn
import torch.optim as optim
import numpy as np

from sklearn.metrics import confusion_matrix, accuracy_score
from tqdm import tqdm

MODEL_PATH = ""

PREDICTING = True
TRAINING = True

N_EPOCH = 1000

N_FRAME = 1000
H = 32
W = 32
COLOR_CH = 3
PATCH_H = 8
PATCH_W = 8

DIM = 180

N_TRANSFORMER = 6
N_ATTENTION = 12
N_CLASS = 10


transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.5, saturation=0.5, hue=0.0),
    transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform_test = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.5, saturation=0.5, hue=0.0),
    transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=N_FRAME,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=N_FRAME,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 事前モデル
        self.mobilenet = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=False)

        # 学習モデル
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.layer1 = nn.Linear(1000, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, N_CLASS)

    def forward(self, images):
        x = self.mobilenet(images)
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        return self.softmax(self.layer3(x))

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.WQ = nn.Linear(DIM, int(DIM / N_ATTENTION))
        self.WK = nn.Linear(DIM, int(DIM / N_ATTENTION))
        self.WV = nn.Linear(DIM, int(DIM / N_ATTENTION))
        
    def forward(self, X):
        Q, K, V = X, X, X
        D = torch.tensor(DIM)
        Q_DASH = self.WQ(Q)
        K_DASH = self.WK(K)
        V_DASH = self.WV(V)
        return torch.matmul(
                torch.matmul(Q_DASH, K_DASH.permute([0, 2, 1])).softmax(dim=-1) / torch.sqrt(D),
                V_DASH)

class MHA(nn.Module):
    def __init__(self):
        super().__init__()
        self.attentions = nn.ModuleList([
            Attention()
            for _ in range(N_ATTENTION)
        ])
        
    def forward(self, X):
        return torch.cat([
            f(X)
            for f in self.attentions
        ], dim=2)
    
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(DIM)
        self.mha = MHA()

        self.dropout = nn.Dropout(p=0.1)
        self.activation = nn.ReLU()
        self.layer = nn.Linear(DIM, DIM)
        
    def forward(self, X):
        norm_X = self.norm(X)
        mha_X = self.mha(norm_X)
        X = self.norm(mha_X + X)
        X = self.activation(self.layer(X))
        return X
    

class MyViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_embedding = nn.Linear(COLOR_CH * PATCH_H * PATCH_W, DIM)

        self.position_embedding = nn.Parameter(
            self.positional_encoding((H / PATCH_H) * (W / PATCH_W), DIM))
       

        self.transformers = nn.ModuleList([
            Transformer()
            for f in range(N_TRANSFORMER)
        ])
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.layer1 = nn.Linear(DIM * int(W / PATCH_W) * int(H / PATCH_H), DIM)
        self.layer2 = nn.Linear(DIM, DIM)
        self.layer3 = nn.Linear(DIM, N_CLASS)

    def forward(self, images):
        patches = self.image2patches(images)
        X = self.to_embedding(patches)
        X = self.add_position_embedding(X)
            
        for f in self.transformers:
            X = f(X)
        X = self.activation(self.layer1(X.reshape(N_FRAME, -1)))
        X = self.activation(self.layer2(X))
        return self.softmax(self.layer3(X))
    
    def add_position_embedding(self, patches):
        _, position_embedding = torch.broadcast_tensors(patches, self.position_embedding)
        patches = patches + position_embedding
        return patches
            
    def image2patches(self, images):
        patches = images.unfold(2, PATCH_H, PATCH_H).unfold(3, PATCH_W, PATCH_W)
        patches = patches.permute([0, 2, 3, 1, 4, 5]).reshape(N_FRAME, -1, COLOR_CH * PATCH_H * PATCH_W)
        return patches
    
    # Code from https://www.tensorflow.org/tutorials/text/transformer
    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / torch.pow(10000, (2 * (i//2)) / d_model)
        return pos * angle_rates

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(torch.arange(position).unsqueeze(1),
                              torch.arange(d_model).unsqueeze(0),
                              d_model)

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads.unsqueeze(0)
        return pos_encoding

model = MyViT()
model.cuda()

if MODEL_PATH:
    model.load_state_dict(torch.load(MODEL_PATH)['model'])

optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

loss_sum=0
total=0


for epoch in range(N_EPOCH):
    if PREDICTING:
        model.train()
        with tqdm(total=len(trainloader), unit="batch") as pbar:

            actuals_all = []
            predicts_all = []
            for images, labels in trainloader:
                labels = labels.cuda()
                images = images.cuda()

                predicts = model(images)
                loss = criterion(predicts, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_sum += loss.item() * images.size(0) 
                total += images.size(0)
                running_loss = loss_sum / total

                pbar.set_postfix({"loss":running_loss})
                pbar.update(1)
                actuals_all += labels.cpu().tolist()
                predicts_all += np.argmax(predicts.cpu().detach().numpy(), axis=1).tolist()

    
    if TRAINING and epoch % 10 == 0:
        print(confusion_matrix(
            actuals_all,
            predicts_all
        ))
        print(accuracy_score(
            actuals_all,
            predicts_all
        ))
        torch.save({'model': model.state_dict(),
            'optimizer': optimizer.state_dict()},
            f'./model_{epoch:02}.pt')
        print("epoch:", epoch)
        model.eval()
        with torch.no_grad():
            actuals_all = []
            predicts_all = []
            
            for images, labels in tqdm(testloader):
                labels = labels.cuda()
                images = images.cuda()

                # 損失計算
                predicts = model(images)                    
                actuals_all += labels.cpu().tolist()
                predicts_all += np.argmax(predicts.cpu().detach().numpy(), axis=1).tolist()
            print(confusion_matrix(
                actuals_all,
                predicts_all
            ))
            print(accuracy_score(
                actuals_all,
                predicts_all
            ))
