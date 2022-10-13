import os
import math
import argparse
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from dataloader import DL
from loss.Loss import LossWithGAN_STE
from models.sa_gan import STRnet2
from pdb import set_trace as stx
from image_utils import *
from warmup_scheduler import GradualWarmupScheduler
from PIL import Image

# torch.set_num_threads(5)

# os.environ["CUDA_VISIBLE_DEVICES"] = ""    ### set the gpu as No....

#忽略警告
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--numOfWorkers', type=int, default=0,
                    help='workers for dataloader')
parser.add_argument('--modelsSavePath', type=str, default='./checkpoint',
                    help='path for saving models')
parser.add_argument('--logPath', type=str,
                    default='./log')
parser.add_argument('--batchSize', type=int, default=16)
parser.add_argument('--loadSize', type=int, default=512,
                    help='image loading size')
parser.add_argument('--dataRoot', type=str,
                    default='../datasets')
parser.add_argument('--pretrained',type=str, default='', help='pretrained models for finetuning')
parser.add_argument('--num_epochs', type=int, default=150, help='epochs')
parser.add_argument('--RESUME', type=bool, default=False)
parser.add_argument('--LR_MIN', type=float, default=1e-6)
parser.add_argument('--VAL_AFTER_EVERY', type=int, default=1)

args = parser.parse_args()


def visual(image):
    im = image.transpose(1,2).transpose(2,3).detach().cpu().numpy()
    Image.fromarray(im[0].astype(np.uint8)).show()


cuda = torch.cuda.is_available()
if cuda:
    cudnn.enable = True
    cudnn.benchmark = True

batchSize = args.batchSize
loadSize = (args.loadSize, args.loadSize)

if not os.path.exists(args.modelsSavePath):
    os.makedirs(args.modelsSavePath)

dataRoot = args.dataRoot

# import pdb;pdb.set_trace()
train_dataset = DL(args.dataRoot,mode='train')
train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True, num_workers=4, drop_last=False, pin_memory=True)

val_dataset = DL(args.dataRoot,mode='val')
val_loader = DataLoader(dataset=val_dataset, batch_size=4, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)

netG = STRnet2(3)

if args.pretrained != '':
    print('loaded ')
    netG.load_state_dict(torch.load(args.pretrained)['state_dict'])

# numOfGPUs = torch.cuda.device_count()

if cuda:
    netG = netG.cuda()
#     if numOfGPUs > 1:
#         netG = nn.DataParallel(netG, device_ids=range(numOfGPUs))

count = 1
start_epoch=1
G_optimizer = optim.Adam(netG.parameters(), lr=0.0001, betas=(0.5, 0.9))

######### Scheduler ###########
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(G_optimizer, args.num_epochs-warmup_epochs, eta_min=args.LR_MIN)
scheduler = GradualWarmupScheduler(G_optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()
######### Resume ###########
if args.RESUME:
    path_chk_rest = get_last_path(args.modelsSavePath, '_best.pth')
    load_checkpoint(netG,path_chk_rest)
    start_epoch = load_start_epoch(path_chk_rest) + 1
    load_optim(G_optimizer, path_chk_rest)
    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print("==> Resuming Training with learning rate:", new_lr)

criterion = LossWithGAN_STE(args.logPath)

if cuda:
    criterion = criterion.cuda()

num_epochs = args.num_epochs
best_psnr=-1
for epoch in range(start_epoch, num_epochs + 1):
    netG.train()

    for k,(imgs, gt) in enumerate(train_loader):
        if cuda:
            imgs = imgs.cuda()
            gt = gt.cuda()
          
#         gt=torch.flatten(gt,0,1)
#         imgs=torch.flatten(imgs,0,1)

        netG.zero_grad()

        final,x_o3,x_o2,x_o1 = netG(imgs)
     
        G_loss = criterion(final,x_o3,x_o2,x_o1,gt)
        G_loss = G_loss.sum()
        G_optimizer.zero_grad()

        G_loss.backward()
        G_optimizer.step()       
        if(k%50==0):
            print('[{}/{}] Loss: {}'.format(k,len(train_loader),G_loss.item()))

        count += 1
    

        if k>1 and k % (len(train_loader)-1) == 0:
            netG.eval()
            psnr_val_rgb = []
            mask_loss=0
            
            for ii, (imgs, gt, filename) in enumerate((val_loader), 0):
#                 print("%d/%d"%(ii,len(val_loader)))
                target = gt.cuda()
                imgs = imgs.cuda()
            
#                 gt=torch.flatten(gt,0,1)
#                 imgs=torch.flatten(imgs,0,1)

                with torch.no_grad():
                    final,x_o3,x_o2,x_o1 = netG(imgs)
                    
                for res,tar,imgs in zip(final,target,imgs):
                    psnr_val_rgb.append(torchPSNR(res, tar))

            psnr_val_rgb  = torch.stack(psnr_val_rgb).mean().item()
         
            if psnr_val_rgb > best_psnr:
                best_psnr = psnr_val_rgb
                best_epoch = epoch
                torch.save({'epoch': epoch, 
                            'state_dict': netG.state_dict(),
                            'optimizer' : G_optimizer.state_dict()
                            }, os.path.join(args.modelsSavePath,"model_best.pth"))

#             print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (epoch, psnr_val_rgb, best_epoch, best_psnr))

            torch.save({'epoch': epoch, 
                        'state_dict': netG.state_dict(),
                        'optimizer' : G_optimizer.state_dict()
                        }, os.path.join(args.modelsSavePath,f"model_psnr_{psnr_val_rgb}_epoch_{epoch}.pth")) 

            scheduler.step()




