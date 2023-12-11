import collections
import os
import cv2
from xml.etree.ElementTree import Element as ET_Element
try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse
from typing import Any, Callable, Dict, List, Optional, Tuple
from PIL import Image
from .utils import download_and_extract_archive, verify_str_arg
from .base import BaseDataset
import numpy as np


__all__ = ("VOCBase", "VOCSegmentation", "VOCDetection")


DATASET_YEAR_DICT = {
    "2012": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
        "filename": "VOCtrainval_11-May-2012.tar",
        "md5": "6cd6e144f989b92b3379bac3b3de84fd",
        "base_dir": os.path.join("VOCdevkit", "VOC2012"),
    },
    "2011": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar",
        "filename": "VOCtrainval_25-May-2011.tar",
        "md5": "6c3384ef61512963050cb5d687e5bf1e",
        "base_dir": os.path.join("TrainVal", "VOCdevkit", "VOC2011"),
    },
    "2010": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar",
        "filename": "VOCtrainval_03-May-2010.tar",
        "md5": "da459979d0c395079b5c75ee67908abb",
        "base_dir": os.path.join("VOCdevkit", "VOC2010"),
    },
    "2009": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2009/VOCtrainval_11-May-2009.tar",
        "filename": "VOCtrainval_11-May-2009.tar",
        "md5": "a3e00b113cfcfebf17e343f59da3caa1",
        "base_dir": os.path.join("VOCdevkit", "VOC2009"),
    },
    "2008": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar",
        "filename": "VOCtrainval_11-May-2012.tar",
        "md5": "2629fa636546599198acfcfbfcf1904a",
        "base_dir": os.path.join("VOCdevkit", "VOC2008"),
    },
    "2007": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
        "filename": "VOCtrainval_06-Nov-2007.tar",
        "md5": "c52e279531787c972589f7e41ab4ae64",
        "base_dir": os.path.join("VOCdevkit", "VOC2007"),
    },
    "2007-test": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",
        "filename": "VOCtest_06-Nov-2007.tar",
        "md5": "b6e924de25625d8de591ea690078ad9f",
        "base_dir": os.path.join("VOCdevkit", "VOC2007"),
    },
}


class VOCBase(BaseDataset):
    r"""Base class for VOC dataset.

    Args:
        root (str): root directory where images are downloaded to.
        years (List[str]): the dataset year.
        image_sets (str): subset of dataset, it has to be `'train'`, `'trainval'`, or `'val'`.
        download (bool): whether to download dataset.
        keep_difficult (bool): whether to keep the difficult object as ground truth.
        transforms_pt (callable, optional): a data augmentation object that is implemented by PyTorch.
        transforms_cv (callable, optional): a data augmentation object that is implemented by OpenCV.
        to_tensor (bool): whether to convert the samples into PyTorch Tensor.
    """
    _SPLITS_DIR: str
    _TARGET_DIR: str
    _TARGET_FILE_EXT: str

    def __init__(
        self,
        root: str,
        years: List[str] = ["2007", "2012"],
        image_sets: str = "train",
        download: bool = False,
        keep_difficult: bool = False,
        transforms_pt: Optional[Callable] = None,
        transforms_cv: Optional[Callable] = None,
        to_tensor: bool = False,
    ):
        super(VOCBase, self).__init__(transforms_pt, transforms_cv, to_tensor)
        self.root = root
        self.years = years
        self.keep_difficult = keep_difficult

        self.image_sets = []
        self.images, self.annotations = [], []
        for year, image_set in zip(self.years, image_sets):
            valid_image_sets = ["train", "trainval", "val"]
            if year == "2007":
                valid_image_sets.append("test")
            if year == "2012":
                valid_image_sets.append("all")
                images = os.listdir(os.path.join(self.root, 'VOCdevkit/VOC2012/JPEGImages'))
                images = [os.path.splitext(img)[0] for img in images]
                rm_images = ['2008_000763', '2008_004172', '2008_004562', '2008_005145',
                            '2008_005262', '2008_005953', '2008_007355', '2008_008051']
                images = list(set(images).difference(rm_images))
                with open(os.path.join(root, 'VOCdevkit/VOC2012/ImageSets/Main/all.txt'), 'w') as f:
                    for line in images:
                        f.write(f"{line}\n")
            image_set = verify_str_arg(image_set, "image_set", valid_image_sets)
            self.image_sets.append(image_set)

            key = "2007-test" if year == "2007" and image_set == "test" else year
            dataset_year_dict = DATASET_YEAR_DICT[key]

            url = dataset_year_dict["url"]
            filename = dataset_year_dict["filename"]
            md5 = dataset_year_dict["md5"]

            base_dir = dataset_year_dict["base_dir"]
            voc_root = os.path.join(self.root, base_dir)

            if download:
                os.makedirs(voc_root, exist_ok=True)
                download_and_extract_archive(url, self.root, filename=filename, md5=md5)

            if not os.path.isdir(voc_root):
                raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

            splits_dir = os.path.join(voc_root, "ImageSets", self._SPLITS_DIR)
            split_f = os.path.join(splits_dir, image_set.rstrip("\n") + ".txt")
            with open(os.path.join(split_f)) as f:
                file_names = [x.strip() for x in f.readlines()]

            image_dir = os.path.join(voc_root, "JPEGImages")
            images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
            self.images += images

            annotation_dir = os.path.join(voc_root, self._TARGET_DIR)
            annotations = [os.path.join(annotation_dir, x + self._TARGET_FILE_EXT) for x in file_names]
            self.annotations += annotations

        assert len(self.images) == len(self.annotations)
        self.voc_classes = get_voc_classes()
        self.category2label = dict(zip(self.voc_classes, range(len(self.voc_classes))))
    
    def get_images(self):
        r"""gets image names in the dataset.
        """
        return self.images


class VOCSegmentation(VOCBase):
    r"""VOC dataset for segmentation.

    Args:
        root (str): root directory where images are downloaded to.
        years (List[str]): the dataset year.
        image_sets (str): subset of dataset, it has to be `'train'`, `'trainval'`, or `'val'`.
        download (bool): whether to download dataset.
        keep_difficult (bool): whether to keep the difficult object as ground truth.
        transforms_pt (callable, optional): a data augmentation object that is implemented by PyTorch.
        transforms_cv (callable, optional): a data augmentation object that is implemented by OpenCV.
        to_tensor (bool): whether to convert the samples into PyTorch Tensor.
    """
    _SPLITS_DIR = "Segmentation"
    _TARGET_DIR = "SegmentationClass"
    _TARGET_FILE_EXT = ".png"

    def getitem(self, index: int) -> Tuple[Any, Any]:
        r"""gets image and target for a single sample.
        Args:
            index (int): index of sample in the dataset.
        Returns:
            tuple: Tuple (image, target).
        """
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = np.array(Image.open(self.annotations[index]), copy=True) / 255.0
        target = self.get_target_single_item("masks", value=masks, type="masks")
        target.update(self.get_target_single_item("image_id", index, type=None))
        return image, target


class VOCDetection(VOCBase):
    r"""VOC dataset for object detection.

    Args:
        root (str): root directory where images are downloaded to.
        years (List[str]): the dataset year.
        image_sets (str): subset of dataset, it has to be `'train'`, `'trainval'`, or `'val'`.
        download (bool): whether to download dataset.
        keep_difficult (bool): whether to keep the difficult object as ground truth.
        transforms_pt (callable, optional): a data augmentation object that is implemented by PyTorch.
        transforms_cv (callable, optional): a data augmentation object that is implemented by OpenCV.
        to_tensor (bool): whether to convert the samples into PyTorch Tensor.
    """
    _SPLITS_DIR = "Main"
    _TARGET_DIR = "Annotations"
    _TARGET_FILE_EXT = ".xml"

    def getitem(self, index: int) -> Tuple[Any, Any]:
        r"""gets image and target for a single sample.
        Args:
            index (int): index of sample in the dataset.
        Returns:
            tuple: Tuple (image, target).
        """
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        target = ET_parse(self.annotations[index]).getroot()
        target = self.parse_voc_xml(target)
        bboxes, difficults = self.get_bbox_target(target['annotation'], self.keep_difficult)
        target = self.get_target_single_item("bboxes", value=bboxes, type="bboxes")
        target.update(self.get_target_single_item("difficults", difficults, type=None))
        target.update(self.get_target_single_item("image_id", index, type=None))
        return image, target

    @staticmethod
    def parse_voc_xml(node: ET_Element) -> Dict[str, Any]:
        voc_dict: Dict[str, Any] = {}
        children = list(node)
        if children:
            def_dic: Dict[str, Any] = collections.defaultdict(list)
            for dc in map(VOCDetection.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == "annotation":
                def_dic["object"] = [def_dic["object"]]
            voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict

    def get_bbox_target(self, target, keep_difficult):
        bbox_list = []
        labels_list = []
        difficult_list = []
        for obj in target['object']:
            difficult = int(obj['difficult']) == 1
            difficult_list.append(int(obj['difficult']))
            if not keep_difficult and difficult:
                continue
            name = obj['name'].lower().strip()
            bbox = obj['bndbox']
            bbox = [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']]
            bbox = [float(eval(b)) for b in bbox]
            bbox.append(self.category2label[name])
            bbox_list.append(bbox)
        bbox_list = np.vstack(bbox_list)
        difficult_list = np.array(difficult_list)
        return bbox_list, difficult_list


def get_voc_classes():
    voc_classes = (
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
            'train', 'tvmonitor'
    )
    return voc_classes