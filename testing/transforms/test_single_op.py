import os, sys
sys.path.insert(0, os.getcwd())
import numpy as np
import torch
import cv2
from horizonms.transforms.base import TargetStructure
from horizonms import transforms as T


if __name__ == "__main__":

    to_tensor_fp = T.ToTensor('float')

    oppt = T.RandomMaskCrop(300, 1, mask_type='masks', obj_labels=[0,1])
    opcv = T.CVRandomMaskCrop(300, 1, mask_type='masks', obj_labels=[0,1])
    # oppt = T.RandomCrop(prob=1, crop_ratio=0.3)
    # opcv = T.CVRandomCrop(prob=1, crop_ratio=0.3)

    img = cv2.imread('009745.jpg')  # hwc
    box1 = [347, 184, 431, 229]
    box2 = [98, 192, 214, 267]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # pt
    imgpt = to_tensor_fp(img)
    # gtpt_box = dict(labels=TargetStructure(type='bboxes', value=torch.tensor([box1, box2])))  # bbox
    mask = torch.zeros(2, img.shape[0], img.shape[1])  # mask
    mask[0, 184:229 + 1, 347:431 + 1] = 1.0
    mask[1, 192:267 + 1, 98:214 + 1] = 1.0
    gtpt_mask = dict(labels=TargetStructure(type='masks', value=mask))

    # cv
    # gtcv_box = dict(labels=TargetStructure(type='bboxes', value=np.vstack([box1, box2])))  # bbox
    mask = np.zeros((img.shape[0], img.shape[1], 2))
    mask[184:229+1, 347:431+1, 0] = 1.0
    mask[192:267+1, 98:214+1, 1] = 1.0
    gtcv_mask = dict(labels=TargetStructure(type='masks', value=mask))

    # pt
    respt = oppt(imgpt, gtpt_mask)
    outimgpt = (respt[0].permute(1,2,0).contiguous().numpy() * 255).astype(np.uint8)  # chw
    outboxpt = respt[1]['labels'].value.numpy()

    # cv
    rescv = opcv((img/255.0).astype(np.float32), gtcv_mask)
    outimgcv = (rescv[0] * 255).astype(np.uint8)  # chw
    outboxcv = rescv[1]['labels'].value

    # plot
    # bbox
    for i in range(len(outboxpt)):
        outimgpt = cv2.rectangle(outimgpt, (outboxpt[i][0], outboxpt[i][1]), (outboxpt[i][2], outboxpt[i][3]), (255,0,0))
    # mask
    # outimgpt[outboxpt.sum(0) == 1] = (255, 0, 0)
    cv2.imwrite("1.jpg", cv2.cvtColor(outimgpt, cv2.COLOR_RGB2BGR))

    for i in range(len(outboxcv)):
        outimgcv = cv2.rectangle(outimgcv, (outboxcv[i][0], outboxcv[i][1]), (outboxcv[i][2], outboxcv[i][3]), (255,0,0))
    # mask
    # outimgcv[outboxcv.sum(-1) == 1] = (255, 0, 0)
    cv2.imwrite("2.jpg", cv2.cvtColor(outimgcv, cv2.COLOR_RGB2BGR))