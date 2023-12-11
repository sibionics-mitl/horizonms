import os
import cv2
import torch
import random
import numpy as np
import pandas as pd
from torch import Tensor
import copy
from horizonms.transforms.base import TargetStructure


def urandom_choice(elements):
    ## 真随机数随机选取
    ## 产生真随机数,不受seed影响
    random_data = os.urandom(1)
    random_data = int.from_bytes(random_data, 'big')
    index = random_data % len(elements)
    return elements[index]


def urandom_int(low=0, high=10):
    ## 真随机数
    ## 产生真随机数,不受seed影响
    assert high > low
    random_data = os.urandom(1)
    random_data = int.from_bytes(random_data, 'big')
    random_data = random_data % 100
    random_data = random_data / 100
    random_data = low + (high - low) * random_data
    return int(random_data)


def urandom_uniform(low=0, high=1):
    ## 真随机数均衡分布
    assert high > low
    ## 产生真随机数,不受seed影响
    random_data = os.urandom(1)
    random_data = int.from_bytes(random_data, 'big')
    random_data = random_data % 100
    random_data = random_data / 100
    random_data = low + (high - low) * random_data
    return random_data


def urandom_uniform_n(low=0, high=1, n=None):
    assert n > 2
    result = [urandom_uniform(low, high) for _ in range(n)]
    return result


def random_target_generator(img, target_type):
    target_type = urandom_choice(target_type)
    value_pt = None
    value_cv = None
    if target_type == 'bboxes':
        bboxes = []
        for i in range(urandom_int(1, 3)):
            (h, w, _) = img.shape  # 0-axis为rows,所以是h,1-axis是cols,所以是w
            # 随机生成边界框的左上角坐标
            x1 = urandom_int(0, w - 1)
            y1 = urandom_int(0, h - 1)
            # 随机生成边界框的右下角坐标
            x2 = urandom_int(x1 + 1, w)
            y2 = urandom_int(y1 + 1, h)
            bboxes.append([x1, y1, x2, y2])
        # print(target_type,bboxes)
        value_pt = torch.tensor(bboxes).float()
        value_cv = np.vstack(bboxes)
    elif target_type == 'points':
        points = []
        for i in range(urandom_int(1, 3)):
            (h, w, _) = img.shape
            x = urandom_int(0, w - 1)
            y = urandom_int(0, h - 1)
            points.append([x, y])
        value_pt = torch.tensor(points).float()
        value_cv = np.vstack(points)
    elif target_type == 'masks':
        masks_pt = []
        masks_cv = []
        for i in range(urandom_int(1, 3)):
            (h, w, _) = img.shape
            x1 = urandom_int(0, w - 1)
            y1 = urandom_int(0, h - 1)
            x2 = urandom_int(x1 + 1, w)
            y2 = urandom_int(y1 + 1, h)
            mask_pt = torch.zeros((h, w)).float()
            mask_pt[y1:y2, x1:x2] = torch.ones((y2 - y1, x2 - x1)).float()
            mask_cv = torch.zeros((h, w)).float()
            mask_cv[y1:y2, x1:x2] = torch.ones((y2 - y1, x2 - x1)).float()
            masks_pt.append(mask_pt)
            masks_cv.append(mask_cv[..., None])
        # print(target_type,masks[0].shape,len(masks))
        value_pt = torch.stack(masks_pt, 0)
        value_cv = np.concatenate(masks_cv, axis=-1)
    elif target_type == 'labels':
        labels_pt = []
        labels_cv = []
        for i in range(urandom_int(1, 3)):
            label = urandom_int(0, 10)
            labels_pt.append(torch.tensor(label))
            labels_cv.append(label)
        value_pt = labels_pt
        value_cv = labels_cv
    target_pt = dict(labels=TargetStructure(type=target_type, value=value_pt))
    target_cv = dict(labels=TargetStructure(type=target_type, value=value_cv))
    return target_pt, target_cv


def array2tensor(img, dtype):
    img = Tensor(np.transpose(img, (2, 0, 1)))
    if dtype == 'float':
        dtype = torch.float32
    elif dtype == 'uint8':
        dtype = torch.uint8
    img = img.to(dtype)

    return img


def tensor2array(tensor):
    if isinstance(tensor, tuple):
        return np.transpose(tensor[0].numpy(), (1, 2, 0)), tensor[1]
    else:
        return np.transpose(tensor.numpy(), (1, 2, 0))


def random_color(c):
    return [random.randint(100, 255) for i in range(c)]


def save_img(img, img_pt, img_cv, diff_img, sub_dir, save_dir, op_name, img_name, target_cv, target_pt, target):
    img = cv2.putText(img.copy(), 'src_img', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
    img_pt = cv2.putText(img_pt.copy(), 'op_img', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
    img_cv = cv2.putText(img_cv.copy(), 'cv_img', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
    diff_img = cv2.putText(diff_img.copy(), 'diff_img', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1,
                           cv2.LINE_AA)

    if target_cv['labels'].type == 'masks':
        assert len(target_pt['labels'].value.shape) == 3, len(target_cv['labels'].value.shape) == 3
        for i in range(target_cv['labels'].value.shape[-1]):
            c = random_color(img_pt.shape[-1])
            img_pt = img_pt + np.concatenate([target_pt['labels'].value.numpy()[i, ...][..., None]] * img_pt.shape[-1],
                                             axis=-1) * c
            img_cv = img_cv + np.concatenate([target_cv['labels'].value[..., i][..., None]] * img_cv.shape[-1],
                                             axis=-1) * c

            img = img + np.concatenate([target['labels'].value[..., i][..., None]] * img.shape[-1],
                                       axis=-1) * c
    if target_cv['labels'].type == 'bboxes':
        for i in range(len(target_cv['labels'].value)):
            c = random_color(img_pt.shape[-1])
            img_pt = cv2.rectangle(img_pt,
                                   (int(target_pt['labels'].value.numpy()[i][0]),
                                    int(target_pt['labels'].value.numpy()[i][1])),
                                   (int(target_pt['labels'].value.numpy()[i][2]),
                                    int(target_pt['labels'].value.numpy()[i][3])),
                                   c, 8)
            img_cv = cv2.rectangle(img_cv, (int(target_cv['labels'].value[i][0]), int(target_cv['labels'].value[i][1])),
                                   (int(target_cv['labels'].value[i][2]), int(target_cv['labels'].value[i][3])), c, 8)

        for i in range(len(target['labels'].value)):
            c = random_color(img_pt.shape[-1])
            img = cv2.rectangle(img, (int(target['labels'].value[i][0]), int(target['labels'].value[i][1])),
                                (int(target['labels'].value[i][2]), int(target['labels'].value[i][3])), c, 8)
    if target_cv['labels'].type == 'points':
        for i in range(len(target_cv['labels'].value)):
            c = random_color(img_pt.shape[-1])
            cv2.circle(img_pt,
                       (int(target_pt['labels'].value.numpy()[i][0]), int(target_pt['labels'].value.numpy()[i][1])), 8,
                       c, -1)
            cv2.circle(img_cv, (int(target_cv['labels'].value[i][0]), int(target_cv['labels'].value[i][1])), 8, c, -1)

        for i in range( target['labels'].value.shape[0]):
            c = random_color(img_pt.shape[-1])
            cv2.circle(img, (int(target['labels'].value[i][0]), int(target['labels'].value[i][1])), 8, c, -1)

    if diff_img.shape < img.shape:
        diff_img_resize = np.zeros(img.shape)
        diff_img_resize[:diff_img.shape[0], :diff_img.shape[1], ...] = diff_img
        diff_img = diff_img_resize
        op_img_resize = np.zeros(img.shape)
        op_img_resize[:img_pt.shape[0], :img_pt.shape[1], ...] = img_pt
        img_pt = op_img_resize
        cv_img_resize = np.zeros(img.shape)
        cv_img_resize[:img_cv.shape[0], :img_cv.shape[1], ...] = img_cv
        img_cv = cv_img_resize
    if diff_img.shape > img.shape:
        img_resize = np.zeros(diff_img.shape)
        img_resize[:img.shape[0], :img.shape[1], ...] = img
        img = img_resize

    if img.shape[2] != img_pt.shape[2]:
        img_pt = np.concatenate([img_pt, img_pt, img_pt])
        img_cv = np.concatenate([img_cv, img_cv, img_cv])
        diff_img = np.concatenate([diff_img, diff_img, diff_img])

    h_img1 = np.concatenate([img_pt, img_cv], axis=1)
    h_img2 = np.concatenate([img, diff_img], axis=1)
    save_img = np.concatenate([h_img1, h_img2], axis=0)

    save_path = os.path.join(save_dir, op_name, sub_dir)
    os.makedirs(save_path, exist_ok=True)
    cv2.imwrite(os.path.join(save_path, img_name[:-4] + '.jpg'), save_img[..., ::-1])


def error_calculation(operators, img_pd, img_path, save_dir, data_type):
    msg_dict = {'op_name': [], 'all img distance avg': [], 'all img different num': [], 'all img allclose num': []}
    MARGIN = 1e-4
    for op_name in operators.keys():
        print(op_name, operators[op_name]['params'])
        torch_op = operators[op_name]['torch']
        cv_op = operators[op_name]['cv']
        random.seed(0)
        torch_op = torch_op(**operators[op_name]['params'])
        random.seed(0)
        cv_op = cv_op(**operators[op_name]['params'])
        gray_supported = operators[op_name]['gray_supported']
        distance_list = []
        different_num_list = []
        allclose_num_list = []
        for img_name in img_pd['file_name']:
            # print(img_name)
            img_raw = cv2.imread(os.path.join(img_path, img_name))[..., ::-1].copy()

            img = img_raw

            ## 随机测试灰度图
            if urandom_int() > 5 and gray_supported:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[..., None]

            if data_type == 'float':
                img = img.astype('float32') / 255.
            target_pt, target_cv = random_target_generator(img, operators[op_name]['target_type'])
            target = copy.deepcopy(target_cv)
            np.random.seed(0)
            torch.manual_seed(0)
            random.seed(0)

            img_pt, target_pt = tensor2array(torch_op(array2tensor(img.copy(), data_type), target_pt))

            np.random.seed(0)
            torch.manual_seed(0)
            random.seed(0)
            img_cv, target_cv = cv_op(img.copy(), target_cv)

            diff_img = np.abs(img_pt - img_cv.astype(np.float32))  # 防止uint8数值溢出
            distance = np.sum(diff_img)
            distance_list.append(distance)
            different_num_list.append(np.sum(diff_img > MARGIN))

            if np.allclose(img_pt, img_cv):
                sub_dir = 'allclose_true'
                allclose_num_list.append(1)
            else:
                sub_dir = 'allclose_false'
                allclose_num_list.append(0)

            if data_type == 'float':
                img = img * 255
                img_pt = img_pt * 255.
                img_cv = img_cv * 255.
                diff_img = diff_img * 255.

            save_img(img, img_pt, img_cv, diff_img, sub_dir, save_dir, op_name, img_name, target_cv, target_pt, target)

        msg_dict['op_name'].append(op_name)
        msg_dict['all img distance avg'].append(int(np.mean(distance_list).round(0)))
        msg_dict['all img different num'].append(int(np.sum(different_num_list).round(0)))
        msg_dict['all img allclose num'].append(np.sum(allclose_num_list))

    msg_pd = pd.DataFrame(msg_dict)
    return msg_pd
