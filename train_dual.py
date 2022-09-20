import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models.model import DnCNN,PriorBoostLayer, NNEncLayer, NonGrayMaskLayer, Color_model
from dataset import prepare_data, Dataset, Dataset_new
from utils import *
from colorizers import *
import torch.utils.model_zoo as model_zoo
from torchvision.utils import save_image

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=16, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=3e-5, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs/dual", help='path of log files')
parser.add_argument("--mode", type=str, default="S", help='with known noise level (S) or blind training (B)')
parser.add_argument("--noiseL", type=float, default=25, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')
opt = parser.parse_args()

def normalize_r(data):
    return data*255.
    
def main():
    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset_new(train=True)
    dataset_val = Dataset_new(train=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    loader_val = DataLoader(dataset=dataset_val, num_workers=4, batch_size=1, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model

    net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    # net.apply(weights_init_kaiming)
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load('/content/drive/MyDrive/CGIR/logs/DnCNN-S-15/net.pth'))
    # net.apply(weights_init_kaiming)

    colorizer_siggraph17 = SIGGRAPHGenerator()
  #  colorizer_siggraph17.load_state_dict(model_zoo.load_url('https://colorizers.s3.us-east-2.amazonaws.com/siggraph17-df00044c.pth', map_location='cpu',check_hash=True))
    colorizer_siggraph17.cuda()
    # colorizer_siggraph17 = nn.DataParallel(Color_model()).cuda()

    # encode_layer = NNEncLayer()
    # boost_layer = PriorBoostLayer()
    # nongray_mask = NonGrayMaskLayer()

    # Move to GPU
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    criterion = nn.MSELoss(size_average=False)
    criterion.cuda()
    # Optimizer
    optimizer_n = optim.Adam(model.parameters(), lr=opt.lr/ 10.)
    optimizer_c = optim.Adam(colorizer_siggraph17.parameters(), lr=opt.lr/ 10.)
    # criterion = nn.CrossEntropyLoss(reduce=False).cuda()
    # visu_param = list(colorizer_siggraph17.parameters())
    # optimizer_c = torch.optim.Adam(
    #         [{'params': visu_param, 'lr': opt.lr/10.},], 
    #         lr=opt.lr, 
    #         betas=(0.9, 0.99),
    #         weight_decay=0.001)
    # training
    writer = SummaryWriter(opt.outf)


    step = 0
    noiseL_B = [0, 55]  # ingnored when opt.mode=='S'
    for epoch in range(opt.epochs):
        
        # train
        for i, data in enumerate(loader_train, 0):
            # training step
            model.train()
            model.zero_grad()
            colorizer_siggraph17.train()
            colorizer_siggraph17.zero_grad()
            optimizer_n.zero_grad()
            optimizer_c.zero_grad()

            img_train, color_real = data
            if opt.mode == 'S':
                noise = torch.FloatTensor(img_train.size()).normal_(mean=0, std=opt.noiseL / 255.)
            if opt.mode == 'B':
                noise = torch.zeros(img_train.size())
                stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size()[0])
                for n in range(noise.size()[0]):
                    sizeN = noise[0, :, :, :].size()
                    noise[n, :, :, :] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n] / 255.)
            imgn_train = img_train + noise
            img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda())
            noise = Variable(noise.cuda())
            color_real = Variable(color_real.cuda(), requires_grad=False)
            # images = img_train.unsqueeze(1).float().cuda()
            # img_ab = img_ab.float()  # [bs, 2, 56, 56]
            # ## Preprocess data
            # encode, max_encode = encode_layer.forward(img_ab)  # Paper Eq(2) Z空间ground-truth的计算
            # targets = torch.Tensor(max_encode).long().cuda()
            # boost = torch.Tensor(boost_layer.forward(encode)).float().cuda()  # Paper Eq(3)-(4), [bs, 1, 56, 56], 每个空间位置的ab概率
            # mask = torch.Tensor(nongray_mask.forward(img_ab)).float().cuda()  # ab通道数值和小于5的空间位置不计算loss, [bs, 1, 1, 1]
            # boost_nongray = boost * mask
            
            # out_img_siggraph17 = model(images)
            
            # # compute loss
            # print(out_img_siggraph17.shape)
            # print(targets.shape)
            # loss_c = (criterion(out_img_siggraph17,targets)*(boost_nongray.squeeze(1))).mean()

            # colorizer_siggraph17.zero_grad()
            # loss_c.backward(retain_graph=True)
            # optimizer_c.step()

            out_train = model(imgn_train)
            out = torch.clamp(imgn_train - out_train, 0., 1.)  # (64，1，50，50)
            out = out.cpu().detach().numpy().squeeze(1)  # (64,50,50)
            out = np.tile(out[:, :, :, None], 3)  # (64,50,50,3)
            out = normalize_r(out).astype(int)

            # img_train = img_train.squeeze(1)
            # img_train = np.expand_dims(img_train.cpu().detach().numpy(),axis=3).repeat(3,axis=3)
            
            tens_l_rs = preprocess_img(out, HW=(256, 256))  # (64,1,256,256)
            tens_l_rs = tens_l_rs.cuda()
            out_img_siggraph17 = postprocess_tens(tens_l_rs, colorizer_siggraph17(tens_l_rs))  # (64,50,50,3)
            
            out_img_siggraph17 = out_img_siggraph17.permute(0, 3, 1, 2).cuda() # (64,3,50,50)
            
            out_img_siggraph17 = Variable(out_img_siggraph17.cuda())
            # if i==1:
            #   save_image(out_img_siggraph17[0].cpu().data,'fake.png')
            #   save_image(targets[0].cpu().data,'ori.png')
            #   save_image(color_real[0].cpu().data,'real.png')
            # train denosing network
            loss_n = criterion(out_train, noise) / (imgn_train.size()[0]*2) 
            loss_c = (criterion(out_img_siggraph17, color_real) / (imgn_train.size()[0] * 2)).requires_grad_() 
            loss = loss_n + loss_c
            loss.requires_grad_(True)
            loss.backward()
            optimizer_n.step()

            # train colorization network 
            loss_c = loss_c.requires_grad_()        
            optimizer_c.step()

            print("[epoch %d][%d/%d] loss: %.4f" % (epoch + 1, i + 1, len(loader_train), loss_c.item()))
            
            # results
            model.eval()
            out_train = torch.clamp(imgn_train - model(imgn_train), 0., 1.)
            psnr_train = batch_PSNR(out_train, img_train, 1.)

            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" % (epoch + 1, i + 1, len(loader_train), loss_n.item(), psnr_train))

            if step % 10 == 0:
                writer.add_scalar('loss', loss_c.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1

        ## the end of each epoch
        model.eval()
        # validate
        psnr_val = 0

        for i, data in enumerate(loader_val, 0):
            img_val, _ = data
            noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=opt.val_noiseL / 255.)
            imgn_val = img_val + noise
            img_val, imgn_val = Variable(img_val.cuda(), volatile=True), Variable(imgn_val.cuda(), volatile=True)
            out_val = torch.clamp(imgn_val - model(imgn_val), 0., 1.)
            psnr_val += batch_PSNR(out_val, img_val, 1.)
        psnr_val /= len(dataset_val)

        print("\n[epoch %d] PSNR_val: %.4f" % (epoch + 1, psnr_val))
        writer.add_scalar('PSNR on validation data', psnr_val, epoch)

        # # log the images
        out_train = torch.clamp(imgn_train - model(imgn_train), 0., 1.)
        Img = utils.make_grid(img_train.data, nrow=8, normalize=True, scale_each=True)
        Imgn = utils.make_grid(imgn_train.data, nrow=8, normalize=True, scale_each=True)
        Irecon = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
        Irecon = utils.make_grid(out_img_siggraph17, nrow=8, normalize=True, scale_each=True)
        writer.add_image('clean image', Img, epoch)
        writer.add_image('noisy image', Imgn, epoch)
        writer.add_image('reconstructed image', Irecon, epoch)

        img_siggraph17 = utils.make_grid(out_img_siggraph17.data, nrow=8, normalize=True, scale_each=True)
        img_real = utils.make_grid(color_real.data, nrow=8, normalize=True, scale_each=True)
        writer.add_image('out', img_siggraph17, epoch)
        writer.add_image('real', img_real, epoch)

        # save model
        torch.save(model.state_dict(), os.path.join(opt.outf, 'netD_dual_{}.pth'.format(epoch)))
        torch.save(colorizer_siggraph17.state_dict(), os.path.join(opt.outf, 'netC_dual_{}.pth'.format(epoch)))


if __name__ == "__main__":
    # if opt.preprocess:
    #     if opt.mode == 'S':
    #         prepare_data(data_path='data', patch_size=50, stride=10, aug_times=1)
    #     if opt.mode == 'B':
    #         prepare_data(data_path='data', patch_size=50, stride=10, aug_times=2)
    main()

