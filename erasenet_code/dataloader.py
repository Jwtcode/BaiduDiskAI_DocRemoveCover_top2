import os
from torch.utils.data import Dataset
import cv2
from torchvision import transforms 
from pdb import set_trace as stx
import random
from random import shuffle
random.seed(1234)


def random_horizontal_flip(imgs):
    if random.random() < 0.3:
        number=random.choice([-1,0,1])
        for i in range(len(imgs)):
            imgs[i]=cv2.flip(imgs[i],number)
    return imgs

def random_rotate(imgs):
    if random.random() < 0.3:
        max_angle = 10
        angle = random.random() * 2 * max_angle - max_angle
        for i in range(len(imgs)):
            w, h = imgs[i].shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
            imgs[i] = cv2.warpAffine(imgs[i], rotation_matrix, (h, w))
    return imgs

class DL(Dataset):
    def __init__(self, rgb_dir,mode='val'):
        """Initializes image paths and preprocessing module."""
	
        self.mode = mode
        self.train_inp_filenames=[]
        self.val_inp_filenames=[]
        self.all_inp_filenames=[]
        self.all_tar_filenames=[]
        self.train_tar_filenames=[]
        self.val_tar_filenames=[]
        self.transform=transforms.Compose([transforms.ToTensor()])
        Tar=os.path.join(rgb_dir,"TAR")
        Ima=os.path.join(rgb_dir,"IMG")
    
        img_path_filenames=os.listdir(Ima)
        shuffle(self.all_inp_filenames)
        for i in range(len(img_path_filenames)):
            filename=img_path_filenames[i].split('.jpg')[0]
            self.all_inp_filenames.append(os.path.join(Ima,img_path_filenames[i]))
            self.all_tar_filenames.append(os.path.join(Tar,filename+'_tar.jpg'))
        numfiles=len(self.all_inp_filenames)
        numtest=numfiles//6
        numtrain=numfiles-numtest
        self.train_inp_filenames=self.all_inp_filenames[:numtrain]
        self.train_tar_filenames= self.all_tar_filenames[:numtrain]
        self.val_inp_filenames=self.all_inp_filenames[numtrain:]
        self.val_tar_filenames=self.all_tar_filenames[numtrain:]

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        if self.mode == 'train':
            image_path = self.train_inp_filenames[index]
            #print(image_path)
            tar_path=self.train_tar_filenames[index]
            inp_img = cv2.imread(image_path,1)
            tar_img=cv2.imread(tar_path)

            all_input = [inp_img,tar_img]
            all_input = random_horizontal_flip(all_input)   
            all_input = random_rotate(all_input)
            inp_img = all_input[0]
            tar_img = all_input[1]
#             cv2.imwrite("inp.jpg",inp_img)
#             cv2.imwrite("tar.jpg",tar_img)
#             stx()
           
            inp_img=self.transform(inp_img)
            tar_img=self.transform(tar_img)
            return inp_img, tar_img
        elif self.mode== 'val':
            image_path = self.val_inp_filenames[index]
            #print(image_path)
            tar_path=self.val_tar_filenames[index]
            inp_img = cv2.imread(image_path,1)
            tar_img=cv2.imread(tar_path)

            inp_img=self.transform(inp_img)
            tar_img=self.transform(tar_img)
            filename = os.path.split(image_path)[-1]

            return inp_img, tar_img,filename




    def __len__(self):
        """Returns the total number of font files."""
        if self.mode=='train':
            return len(self.train_inp_filenames)
        elif self.mode=='val':
            return len(self.val_inp_filenames)
        # else :
        # 	return len(self.all_inp_filenames)
