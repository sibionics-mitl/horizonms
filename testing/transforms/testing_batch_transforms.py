import os, sys
sys.path.insert(0, os.getcwd())

import torch
import warnings
warnings.filterwarnings("ignore")
import copy
import cv2
import numpy as np
# from utils import urandom_int
from horizonms import transforms as T
from horizonms .models import BatchImage
from testing_augment import random_target_generator


# T.ToOnehotLabels (√)
# T.Mixup (√)
# T.SoftmaxLabelSmoothing
# T.SigmoidLabelSmoothing

h, w, c = 300, 400, 3

img1 = np.random.rand(300, 400, 3).astype(np.float32)
img2 = np.random.rand(350, 500, 3).astype(np.float32)
img1_tensor = torch.from_numpy(img1.transpose((2, 0, 1))).contiguous()
img2_tensor = torch.from_numpy(img2.transpose((2, 0, 1))).contiguous()
img_list = [img1_tensor, img2_tensor]

gt_list_single = [dict(labels=T.TargetStructure(type='labels', value=torch.tensor(0))),
                   dict(labels=T.TargetStructure(type='labels', value=torch.tensor(2)))]
gt_list_multi = [dict(labels=T.TargetStructure(type='labels', value=[torch.tensor(0), torch.tensor(1)])),
                  dict(labels=T.TargetStructure(type='labels', value=[torch.tensor(2), torch.tensor(0)]))]

batch_image_process = BatchImage()
img_batch1, gt_batch_single = batch_image_process(img_list, gt_list_single)
img_batch2, gt_batch_multi = batch_image_process(img_list, gt_list_multi)


if __name__ == "__main__":

    # --- one hot ---
    print('---single head: ---')
    onehot_label = T.ToOnehotLabels(3)
    _, result = onehot_label(img_batch1, copy.deepcopy(gt_batch_single))
    print('before onehot:\n', gt_batch_single['labels'].value, '\nafter onehot:\n', result['labels'].value)

    print("\n--- multi heads: --- ")
    onehot_label = T.ToOnehotLabels(3, index=list(range(len(gt_batch_multi['labels'].value))))
    _, result = onehot_label(img_batch2, copy.deepcopy(gt_batch_multi))
    for i in range(len(gt_batch_multi['labels'].value)):
        print(f'head{i}, before onehot:\n', gt_batch_multi['labels'].value[i], '\nafter result:\n', result['labels'].value[i])
    # del result
    # --- one hot ---

    # === mix up ===
    mix_up = T.Mixup()
    _, result = mix_up(img_batch1, copy.deepcopy(gt_batch_single))
    print(gt_batch_single['labels'].value, result['labels'].value)
    try:
        mix_up = T.Mixup(index=[0, 1])
        _, result = mix_up(img_batch2, gt_batch_multi, )
    except Exception as e:
        print(f"error MIXUP: {e}")
    # del result
    # === mix up ===

    # *** SoftmaxLabelSmoothing ***
    # del result
    onehot_label = T.ToOnehotLabels(3)
    _, result = onehot_label(img_batch1, copy.deepcopy(gt_batch_single))
    label_smooth = T.SoftmaxLabelSmoothing(smoothing=0.1)
    _, result2 = label_smooth(img_batch1, copy.deepcopy(result))
    print("single head ", result['labels'].value, '\n', result2['labels'].value)

    onehot_label = T.ToOnehotLabels(3, index=[0, 1])
    _, result = onehot_label(img_batch2, copy.deepcopy(gt_batch_multi))
    label_smooth = T.SoftmaxLabelSmoothing(smoothing=0.1, index=[0, 1])
    # label_smooth = T.SoftmaxLabelSmoothing(smoothing=[0.1, 0.5], index=[0, 1])
    _, result2 = label_smooth(img_batch1, copy.deepcopy(result))
    for i in range(2):
        print(f'head{i}', result['labels'].value[i], '\n', result2['labels'].value[i])

    label_smooth = T.SoftmaxLabelSmoothing(smoothing=[0.1, 0.5], index=[0, 1])
    _, result2 = label_smooth(img_batch1, copy.deepcopy(result))
    for i in range(2):
        print(f'head{i}', result['labels'].value[i], '\n', result2['labels'].value[i])
    # *** SoftmaxLabelSmoothing ***

    # +++ SigmoidLabelSmoothing +++
    label_smooth = T.SigmoidLabelSmoothing(smoothing=0.1)
    _, result = label_smooth(img_batch1, copy.deepcopy(gt_batch_single))
    print(gt_batch_single['labels'].value, result['labels'].value)

    label_smooth = T.SigmoidLabelSmoothing(smoothing=0.1, index=[0, 1])
    _, result = label_smooth(img_batch1, copy.deepcopy(gt_batch_multi))
    for i in range(2):
        print(f'head{i}', gt_batch_multi['labels'].value[i], result['labels'].value[i])
    # +++ SigmoidLabelSmoothing +++