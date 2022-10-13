import os
import math
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from PIL import Image
import numpy as np
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from models.sa_gan import STRnet2
from dataloader import *
from skimage import img_as_ubyte
from image_utils import *
from pdb import set_trace as stx

parser = argparse.ArgumentParser()
parser.add_argument('--numOfWorkers', type=int, default=0,
                    help='workers for dataloader')
parser.add_argument('--modelsSavePath', type=str, default='./checkpoints',
                    help='path for saving models')
parser.add_argument('--logPath', type=str,
                    default='./log')
parser.add_argument('--batchSize', type=int, default=1)
parser.add_argument('--loadSize', type=int, default=512,
                    help='image loading size')
parser.add_argument('--dataRoot', type=str,
                    default='./datasets')
parser.add_argument('--pretrained',type=str, default='./checkpoint/l1_best.pth', help='pretrained models for finetuning')
parser.add_argument('--savePath', type=str, default='./results/')
args = parser.parse_args()

cuda = torch.cuda.is_available()
if cuda:
    print('Cuda is available!')
    cudnn.benchmark = True


def visual(image):
    im =(image).transpose(1,2).transpose(2,3).detach().cpu().numpy()
    Image.fromarray(im[0].astype(np.uint8)).show()

batchSize = args.batchSize
loadSize = (args.loadSize, args.loadSize)
dataRoot = args.dataRoot
savePath = args.savePath

if not os.path.exists(savePath):
    os.makedirs(savePath)
    os.makedirs(result_with_mask)
    os.makedirs(result_straight)

dataRoot = args.dataRoot

val_dataset = DL(args.dataRoot,mode='test')
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)



netG = STRnet2(3)

load_checkpoint(netG,args.pretrained)

#
if cuda:
    netG = netG.cuda()

for param in netG.parameters():
    param.requires_grad = False


import time
start = time.time()
netG.eval()

for inp_img,H,W,p_h,p_w,filename in (val_loader):
    print(filename)
   
    inp_img = inp_img.cuda()
    inp_img=torch.flatten(inp_img,0,1)
    
    with torch.no_grad():
        final,x_o_unet,x_o2,x_o1= netG(inp_img)
    final = final.data.cpu()
    final=np.transpose(final,(0,2,3,1))
    final=reset_result(final,H,W,p_h,p_w)
    final=np.clip(final,0,1)
    final=img_as_ubyte(final)
    cv2.imwrite(savePath+'/'+filename[0],final)






