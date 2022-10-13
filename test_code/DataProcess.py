import os
from matplotlib import test
import numpy as np
import cv2
from PIL import Image
from pdb import set_trace as stx
import copy
import functools

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 椭圆结构
def demo_postprocess(outputs, img_size, p6=False):

    grids = []
    expanded_strides = []

    if not p6:
        strides = [8, 16, 32]
    else:
        strides = [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs

def preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r

def partition(image_path):
	#读取图片
	raw_img = cv2.imread(image_path,1)
	inp1_img,ratio=preproc(raw_img,(640,640))
	inp2_img = raw_img.astype(np.float32) / 255.0
	inp2_img = np.transpose(inp2_img, (2, 0, 1)).astype(np.float32)
	return inp1_img,inp2_img,raw_img,ratio


########################YOLO###############################

def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep

def multiclass_nms_class_aware(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-aware version."""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)

def multiclass_nms_class_agnostic(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-agnostic version."""
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

    valid_score_mask = cls_scores > score_thr
    if valid_score_mask.sum() == 0:
        return None
    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]
    keep = nms(valid_boxes, valid_scores, nms_thr)
    if keep:
        dets = np.concatenate(
            [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
        )
    return dets

def multiclass_nms(boxes, scores, nms_thr, score_thr, class_agnostic=True):
    """Multiclass NMS implemented in Numpy"""
    if class_agnostic:
        nms_method = multiclass_nms_class_agnostic
    else:
        nms_method = multiclass_nms_class_aware
    return nms_method(boxes, scores, nms_thr, score_thr)

def compare_area(a,b):
	
	if((a[3]-a[1])*(a[2]-a[0])>(b[3]-b[1])*(b[2]-b[0])) : return -1
	else : return 1



def get_boxes(predictions, ratio,inp2):
	#inp1 CHW
	H,W=inp2.shape[1:]
	boxes = predictions[:, :4]
	scores = predictions[:, 4:5] * predictions[:, 5:]
	boxes_xyxy = np.ones_like(boxes)
	boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
	boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
	boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
	boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
	boxes_xyxy /= ratio
	dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.4)
	image_list=[]
	bbox_expand=[]
	if dets is not None:
		BASE=1280
		sortbb=[]
		for bb in dets[:, :4]:
			sortbb.append(bb)
		sortbb.sort(key=functools.cmp_to_key(compare_area))
		maxx1=W;maxy1=H;maxx2=0;maxy2=0;
		LAST_W=0
		LAST_H=0
		# test_img=np.transpose(inp2,(1,2,0))
		# test_img=(test_img*255).astype("uint8")
		sortbb2=[]
		for box in sortbb:
	
			x1,y1,x2,y2=box
			x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
			x1=max(0,x1)
			x2=min(W,x2)
			y1=max(0,y1)
			y2=min(H,y2)
			w=x2-x1
			h=y2-y1
			area=w*h
			if(area<11000): continue
			sortbb2.append(np.array([x1,y1,x2,y2]))

			#################################################
			
		# 	cv2.rectangle(test_img, (x1,y1), (x2,y2), (255, 0, 255), 2)
		# cv2.imwrite("./{}.jpg".format(str(x1+y1+x2*y2)),test_img)
		# return []
			# stx()
			# test_img=inp2[:,y1:y2,x1:x2]
			# test_img=np.transpose(test_img,(1,2,0))
			# cv2.imwrite("test1.jpg",test_img*255)
			###############################################
			if(x1>=maxx1 and y1>=maxy1 and x2<=maxx2 and y2<=maxy2): continue
			if LAST_H!=0 :
				x_mid=(x1+x2)//2
				y_mid=(y1+y2)//2
				w_add_half=LAST_W//2
				h_add_half=LAST_H//2
				if(x_mid-w_add_half<=0):
					x1=0
					x2=LAST_W
				elif(x_mid+w_add_half>W):
					x2=W
					x1=W-LAST_W
				else:
					x1=x_mid-w_add_half
					x2=x_mid+(LAST_W-w_add_half)
				if(y_mid-h_add_half<=0):
					y1=0
					y2=LAST_H
				elif(y_mid+h_add_half>H):
					y2=H
					y1=H-LAST_H
				else:
					y1=y_mid-h_add_half
					y2=y_mid+(LAST_H-h_add_half)
			else:
				w_add_half=BASE//2
				h_add_half=BASE//2
				x_mid=(x1+x2)//2
				y_mid=(y1+y2)//2

				if(x_mid-w_add_half<=0):
					x1=0
					if(x2<BASE):x2=min(BASE,W)
					else:
						pad=64-((x2-x1)%64)
						x2=x2+pad
				elif(x_mid+w_add_half>W):
					x2=W
					if(x1>W-BASE): x1=max(0,W-BASE)
					else:
						pad=64-((x2-x1)%64)
						x1=x1-pad
				else:
					if(w<BASE):
						x1=x_mid-w_add_half
						x2=x_mid+w_add_half
					else:
						pad=64-(w%64)
						if(x1-pad>=0):
							x1=x1-pad
						else:
							x2=x2+pad


				if(y_mid-h_add_half<=0):
					y1=0
					if(y2<BASE):y2=min(BASE,H)
					else:
						pad=64-((y2-y1)%64)
						y2=y2+pad
				elif(y_mid+h_add_half>H):
					y2=H
					if(y1>H-BASE): y1=max(0,H-BASE)
					else:
						pad=64-((y2-y1)%64)
						y1=y1-pad
				else:
					
					if(h<BASE):
						y1=y_mid-h_add_half
						y2=y_mid+h_add_half
					else:
						pad=64-(h%64)
						if(y1-pad>=0):
							y1=y1-pad
						else:
							y2=y2+pad

			#######################################
			# stx()
			# test_img=inp2[:,y1:y2,x1:x2]
			# test_img=np.transpose(test_img,(1,2,0))
			# cv2.imwrite("test2.jpg",test_img*255)
			###################################
			image_list.append(inp2[:,y1:y2,x1:x2])
			bbox_expand.append(np.array([x1,y1,x2,y2]))
			maxx1=x1;maxy1=y1;maxx2=x2;maxy2=y2;
			LAST_W=x2-x1
			LAST_H=y2-y1
		inp=np.stack(image_list,0)
		bbox_expand=np.stack(bbox_expand,0)
		return [inp,sortbb2,bbox_expand]
	else:
		return []
			



def FillHole(imgPath):

	im_th = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE);
	# Copy the thresholded image.
	im_floodfill = im_th.copy()

	# Mask used to flood filling.
	# Notice the size needs to be 2 pixels than the image.
	h, w = im_th.shape[:2]
	mask = np.zeros((h+2, w+2), np.uint8)

	# Floodfill from point (0, 0)
	cv2.floodFill(im_floodfill, mask, (0,0), 255);

	# Invert floodfilled image
	im_floodfill_inv = cv2.bitwise_not(im_floodfill)

	# Combine the two images to get the foreground.
	im_out = im_th | im_floodfill_inv

	return im_out


def cover(boxes,bbox_expand,image,raw_img):
	image=np.clip(image,0,1)
	h,w=raw_img.shape[:2]
	image=(image*255).astype('uint8')
	
	# cv2.imwrite("result.jpg",image[0])
	
	parent=copy.deepcopy(raw_img)
	son=copy.deepcopy(raw_img)
	for i in range(len(bbox_expand)):
		x1,y1,x2,y2=bbox_expand[i]
		raw_img[y1:y2,x1:x2,:]=image[i]
	for box in (boxes):
		left, top, right,bottom=box
		top = top - 50
		left = left - 50
		bottom = bottom + 50
		right = right + 50
		top = max(0, int(top))
		left = max(0, int(left))
		bottom = min(h, int(bottom))
		right = min(w, int(right))
		parent[top:bottom,left:right,:]=raw_img[top:bottom,left:right,:]
		divmap=np.abs(np.mean(parent[top:bottom,left:right,:],-1)-np.mean(son[top:bottom,left:right,:],-1))
		divmap[np.where(divmap>=2)]=255
		divmap[np.where(divmap<2)]=0
		divmap = cv2.morphologyEx(divmap,cv2.MORPH_OPEN,kernel,iterations=1)
		cv2.imwrite("divmap.jpg",divmap)
		divmap=FillHole("./divmap.jpg")
		##############################
		# name=str(top+left+bottom)
		# cv2.imwrite("mask/{}.jpg".format(name),divmap)
		#########################################
		valid_index=np.where(divmap>0)
		son[top:bottom,left:right,:][valid_index]=raw_img[top:bottom,left:right,:][valid_index]
		# cv2.imwrite("son.jpg",son.astype('uint8'))
	return son

if __name__=="__main__":


	mask=cv2.imread("/home/jiaowt/Documents/MPRNet-main/53.jpg",1)
	# img,_=DA.mprin("/home/jiaowt/Documents/MPRNet-main/53.jpg",mask)
	inp_img,H,W,p_w,p_h=partition("/home/jiaowt/Documents/MPRNet-main/53.jpg")

