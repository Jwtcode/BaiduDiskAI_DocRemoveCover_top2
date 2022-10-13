import torch
import numpy as np
import cv2
from pdb import set_trace as stx
from natsort import natsorted
from glob import glob
import os
def torchPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps

def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def numpyPSNR(tar_img, prd_img):
    imdff = np.float32(prd_img) - np.float32(tar_img)
    rmse = np.sqrt(np.mean(imdff**2))
    ps = 20*np.log10(255/rmse)
    return ps
    
def reset(mask,h,w):
# 	stx()  
	cur_c=mask.shape[1]
	cur_h=mask.shape[2]
	cur_w=mask.shape[3]

	new_mask=torch.zeros([cur_c,h,w],dtype=torch.float, requires_grad=False)
	i=0
	for r in range(2):
		for c in range(2):
			new_mask[:,r * cur_h:(r + 1) * cur_h,c * cur_w:(c + 1) * cur_w]=mask[i]
			i=i+1
	return new_mask


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
def get_last_path(path, session):
	x = natsorted(glob(os.path.join(path,'*%s'%session)))[-1]
	return x

def load_start_epoch(weights):
    checkpoint = torch.load(weights)
    epoch = checkpoint["epoch"]
    return epoch
def load_optim(optimizer, weights):
    checkpoint = torch.load(weights)
    optimizer.load_state_dict(checkpoint['optimizer'])
    # for p in optimizer.param_groups: lr = p['lr']
    # return lr