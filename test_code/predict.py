import os
import sys
import glob
import cv2
import numpy as np
from DataProcess import *
import onnxruntime
# import pdb
from pdb import set_trace as stx
sess1 = onnxruntime.InferenceSession(path_or_bytes='./my_yolox_s.onnx', providers=['CUDAExecutionProvider'])
sess2 = onnxruntime.InferenceSession(path_or_bytes='./model.onnx', providers=['CUDAExecutionProvider'])
input_name1 = sess1.get_inputs()[0].name
input_name2 = sess2.get_inputs()[0].name




def process(src_image_dir, save_dir):
   
    image_paths = glob.glob(os.path.join(src_image_dir, "*.jpg"))
   
    for image_path in image_paths:
        save_path = os.path.join(save_dir, os.path.basename(image_path))
        inp1,inp2,raw_img,ratio=partition(image_path)
       
        ort_inputs = {sess1.get_inputs()[0].name: inp1[None, :, :, :]}
        output1 = sess1.run(None, ort_inputs)
        predictions = demo_postprocess(output1[0], (640,640), False)[0]
        data = get_boxes(predictions,ratio,inp2)
        if(len(data)==0):
            cv2.imwrite(save_path,raw_img)
            continue
        inp1,bboxes,bbox_expand=data
        output2 = sess2.run(None,{input_name2:inp1})
       
        output2=np.transpose(output2[0],(0,2,3,1))
        
        output2=cover(bboxes,bbox_expand,output2,raw_img)
        cv2.imwrite(save_path,output2)
        

    
if __name__ == "__main__":
    assert len(sys.argv) == 3
  
    src_image_dir = sys.argv[1]
    save_dir = sys.argv[2]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # src_image_dir='./img2'
    # save_dir='./res2'
    process(src_image_dir,save_dir)
  