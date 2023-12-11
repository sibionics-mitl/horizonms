import os
import numpy as np
import pandas as pd
import tqdm
import cv2
from .eye_region import get_eye_region
from .base import BaseDataset
from typing import Callable, Optional


__all__ = ("EyePACSClassification", "EyePACSClassificationPng", "eyepacs_preprocessing")
    

class EyePACSClassification(BaseDataset):
    r"""Eye PACS dataset for diabetic retinopathy stage classification.

    Args:
        root (str): root directory of the ImageNet dataset.
        mode (str): dataset split, it has to be `'train'`, `'valid'`, or `'test'`.
        use_valid (bool): if True, splits the original testing subset into testing and validation subsets.
        valid_seed (int): seed for testing-validation subsets split.
        valid_ratio (float): ratio of samples in validation subset during subset split.
        resize (bool): if True, resizes the image such that the lowest dimension is 1200.
        transforms_pt (callable, optional): a data augmentation object that is implemented by PyTorch.
        transforms_cv (callable, optional): a data augmentation object that is implemented by OpenCV.
        to_tensor (bool): if True, converts the samples into PyTorch Tensor.
    """
    def __init__(self, root: str, mode: str, 
                 use_valid: bool = True, 
                 valid_seed: int = 0,
                 valid_ratio: float = 0.2, # value used in official kaggle competition
                 resize: bool = False, 
                 transforms_pt: Optional[Callable] = None,
                 transforms_cv: Optional[Callable] = None,
                 to_tensor: bool = True):
        super(EyePACSClassification, self).__init__(transforms_pt, transforms_cv, to_tensor)
        assert mode in ["train", "valid", "test"]
        self.root = root
        self.mode = mode
        self.use_valid = use_valid
        self.valid_seed = valid_seed
        self.valid_ratio = valid_ratio
        self.resize = resize

        self.suffix = f"seed={self.valid_seed}"
        if use_valid:
            if not (os.path.exists(os.path.join(root,'csv', f'test_test_{self.suffix}.csv'))
                | os.path.exists(os.path.join(root,'csv', f'test_valid_{self.suffix}.csv'))):
                self.test_valid_split()
            self.csvFile = f"test_{self.mode}_{self.suffix}.csv"
        if mode == 'train':
            self.csvFile = "trainLabels.csv"
        else:
            self.csvFile = f'test_{mode}_{self.suffix}.csv'
        print(f"csv file: {self.csvFile}")
        self.folder = mode
        if self.mode == 'valid':
            self.folder = "test"
        self.csv = pd.read_csv(os.path.join(root, 'csv', self.csvFile))
        self.images = self.csv['image']
        # self.labels = self.csv['level']
        self.num_classes = 5
        self.sampling_labels = self.csv['level']
        self.sampling_classes = 5
        
        print("samples in classes: ", [(self.csv['level']==k).sum() for k in range(5)])

    def test_valid_split(self):
        r"""splits the original testing subset into testing and validation subsets.
        """
        np.random.seed(self.valid_seed)
        data = pd.read_csv(os.path.join(self.root, 'csv', 'retinopathy_solution.csv'))
        data['case_id'] = data['image'].apply(lambda x: x.split('_')[0])
        case_id = np.unique(data['case_id'])
        n_case = int(len(case_id)*self.valid_ratio)
        ind_rand = np.random.permutation(len(case_id))
        case_valid = case_id[ind_rand[:n_case]]
        print(f'{n_case}/{len(case_id)} cases are selected for validation')

        flag_valid = np.in1d(data['case_id'],case_valid)

        data_valid = data[flag_valid].reset_index(drop=True)
        data_test = data[flag_valid==False].reset_index(drop=True)
        print(f'{len(data_valid)} images for validation, {len(data_test)} images for testing')

        data_valid.to_csv(os.path.join(self.root, 'csv', f'test_valid_{self.suffix}.csv'), index=False)
        data_test.to_csv(os.path.join(self.root, 'csv', f'test_test_{self.suffix}.csv'), index=False)

    def getitem(self, index):
        r"""gets image and target for a single sample.

        Args:
            index (int): index of sample in the dataset.
        Returns:
            tuple: Tuple (image, target).
        """
        image_name = self.csv['image'].iloc[index]
        label = np.eye(self.num_classes)[self.csv['level'].iloc[index]]
        img = cv2.imread(os.path.join(self.root, self.folder, image_name+'.jpeg'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        eye_mask, crop_loc = get_eye_region(img)
        img = img[crop_loc[0]:crop_loc[1], crop_loc[2]:crop_loc[3]]
        eye_masks = eye_mask[crop_loc[0]:crop_loc[1], crop_loc[2]:crop_loc[3]]
        if self.resize:
            org_size = np.array(img.shape[:2])
            if org_size.min() > 1200:
                new_size = tuple((org_size / org_size.min() * 1200).astype(int))
                img = cv2.resize(img.astype(np.float64), (new_size[1], new_size[0]))
                eye_masks = cv2.resize(eye_masks, (new_size[1], new_size[0]))
        eye_masks = (eye_masks > 255/2.0).astype(np.bool)
        target = self.get_target_single_item("labels", label, type="labels")
        target.update(self.get_target_single_item("image_id", index, type=None))
        target.update(self.get_target_single_item("eye_masks", eye_masks[:,:,None], type="masks"))
        target.update(self.get_target_single_item("image_name", image_name, type=None))
        return img, target

    def get_images(self):
        r"""gets image names in the dataset.
        """
        return self.images
       

def image_processing(img_id, csv, img_dir, save_dir):
    r"""image preprocessing for a single image in Eye PACS dataset.
    """
    image_name = csv['image'].iloc[img_id]
    fn_save = os.path.join(save_dir, f'{image_name}.png')
    if os.path.exists(fn_save):
        return

    image = cv2.imread(os.path.join(img_dir, image_name+'.jpeg'))
    eye_mask, crop_loc = get_eye_region(image, 16)
    B,G,R = cv2.split(image)
    image = cv2.merge((B, G, R, eye_mask))
    image = image[crop_loc[0]:crop_loc[1], crop_loc[2]:crop_loc[3]]

    org_size = np.array(image.shape[:2])
    if org_size.min() > 1200:
        new_size = tuple((org_size / org_size.min() * 1200).astype(int))
        image = cv2.resize(image, (new_size[1], new_size[0]))
    
    cv2.imwrite(fn_save, image)


def eyepacs_preprocessing(root, mode, num_workers=12):
    r"""image preprocessing for Eye PACS dataset.
    """
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial
    
    img_dir = os.path.join(root, mode)
    save_dir = os.path.join(root, f'{mode}_preprocessing')
    os.makedirs(save_dir, exist_ok=True)

    if mode == 'train':
        csv = pd.read_csv(os.path.join(root, 'csv','trainLabels.csv'))
    elif mode == 'test':
        csv = pd.read_csv(os.path.join(root, 'csv','retinopathy_solution.csv'))
    print(f"#images = {len(csv)}")

    idx_image_processing = partial(image_processing, csv=csv, img_dir=img_dir, save_dir=save_dir)

    pool = ProcessPoolExecutor(max_workers=num_workers)
    requests = [pool.submit(idx_image_processing, img_id) for img_id in range(len(csv))]
    for result in tqdm.tqdm(requests):
        result.result()  


class EyePACSClassificationPng(EyePACSClassification):
    r"""Eye PACS dataset for diabetic retinopathy stage classification. The preprocessed png images are used as employed.

    Args:
        root (str): root directory of the ImageNet dataset.
        mode (str): dataset split, it has to be `'train'`, `'valid'`, or `'test'`.
        use_valid (bool): if True, splits the original testing subset into testing and validation subsets.
        valid_seed (int): seed for testing-validation subsets split.
        valid_ratio (float): ratio of samples in validation subset during subset split.
        resize (bool): if True, resizes the image such that the lowest dimension is 1200.
        transforms_pt (callable, optional): a data augmentation object that is implemented by PyTorch.
        transforms_cv (callable, optional): a data augmentation object that is implemented by OpenCV.
        to_tensor (bool): if True, converts the samples into PyTorch Tensor.
    """
    def __init__(self, root: str, mode: str, 
                 use_valid: bool = True, 
                 valid_seed: int = 0,
                 valid_ratio: float = 0.2, # value used in official kaggle competition
                 resize: bool = False, 
                 transforms_pt: Optional[Callable] = None,
                 transforms_cv: Optional[Callable] = None,
                 to_tensor: bool = True):
        super(EyePACSClassificationPng, self).__init__(root, mode, use_valid,
                            valid_seed, valid_ratio, resize,
                            transforms_pt, transforms_cv, to_tensor)
        if mode in ['test', 'valid']:
            self.folder = f"test_preprocessing"
        else:
            self.folder = f"train_preprocessing"

    def getitem(self, index):
        r"""gets image and target for a single sample.
        
        Args:
            index (int): index of sample in the dataset.
        Returns:
            tuple: Tuple (image, target).
        """
        image_name = self.csv['image'].iloc[index]
        label = np.eye(self.num_classes)[self.csv['level'].iloc[index]]
        if not os.path.exists(os.path.join(self.root, self.folder, image_name+'.png')):
            print(f"{os.path.join(self.root, self.folder, image_name+'.png')} does not exist!")
        img = cv2.imread(os.path.join(self.root, self.folder, image_name+'.png'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        eye_masks = (img[:, :, -1] > 255/2.0).astype(np.bool)
        img = img[:, :, :3]
    
        target = self.get_target_single_item("labels", label, type="labels")
        target.update(self.get_target_single_item("image_id", index, type=None))
        target.update(self.get_target_single_item("eye_masks", eye_masks[:,:,None], type="masks"))
        target.update(self.get_target_single_item("image_name", image_name, type=None))
        return img, target