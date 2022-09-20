import torch
import torch.nn as nn
from models import DnCNN
import cv2
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from dataset import prepare_data, Dataset, Dataset_new
from torch.utils.data import DataLoader
class DnCNN_AT(DnCNN):
    def forward(self, x):
        x = self.relu(self.input(x))

        g0 = self.layer1(x)
        g1 = self.layer2(g0)
        g2 = self.layer3(g1)
        g3 = self.layer4(g2)
        g4 = self.layer4(g3)
        g5 = self.layer4(g4)
        g6 = self.layer4(g5)
        g7 = self.layer4(g6)
        g8 = self.layer4(g7)
        g9 = self.layer4(g8)
        g10 = self.layer4(g9)

        return [g.pow(2).mean(1) for g in (g0, g1, g2, g3, g4, g5, g6, g7, g8, g9, g10)]


tr_center_crop = T.Compose([
        T.ToPILImage(),
        T.Resize(256),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# im = cv2.imread("data/Set68/test011.png",0)
# print(im.shape)
dataset_val = Dataset_new(train=False)
loader_val = DataLoader(dataset=dataset_val, num_workers=4, batch_size=1, shuffle=True)

model = DnCNN_AT(1).cuda()
checkpoint = torch.load("logs/net.pth")
model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint.items()})

model.eval()
with torch.no_grad():
    # x = tr_center_crop(im)
    for i, data in enumerate(loader_val, 0):
      img_val, _ = data
      noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=25 / 255.)
      imgn_val = img_val + noise
      img_val, imgn_val = Variable(img_val.cuda(), volatile=True), Variable(imgn_val.cuda(), volatile=True)
      # out_val = torch.clamp(imgn_val - model(imgn_val), 0., 1.)
      gs = model(imgn_val)

for i, g in enumerate(gs):
    plt.imshow(g[0].data.cpu(), interpolation='bicubic')
    plt.title(f'g{i}')
    plt.show()