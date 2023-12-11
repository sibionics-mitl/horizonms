from contextlib import contextmanager
import os
import shutil
import tempfile
import torch
from .utils import check_integrity, extract_archive
import pandas as pd
import cv2
from .base import BaseDataset
from typing import Any, Dict, List, Iterator, Optional, Tuple, Callable


__all__ = ("ImageNetClassification")


ARCHIVE_META = {
    'train': ('ILSVRC2012_img_train.tar', '1d675b47d978889d74fa0da5fadfb00e'),
    'val': ('ILSVRC2012_img_val.tar', '29b22e2961454d5413ddabcf34fc5622'),
    'devkit': ('ILSVRC2012_devkit_t12.tar.gz', 'fa75699e90414af021442c21a62c3abf')
}


META_FILE = "meta.bin"


class ImageNetClassification(BaseDataset): 
    r"""`ImageNet <http://image-net.org/>`_ 2012 Classification Dataset.
    Link: https://image-net.org/challenges/LSVRC/2012/2012-downloads.php

    Args:
        root (str): root directory of the ImageNet dataset.
        mode (str): dataset split, it has to be `'train'` or `'val'`.
        transforms_pt (callable, optional): a data augmentation object that is implemented by PyTorch.
        transforms_cv (callable, optional): a data augmentation object that is implemented by OpenCV.
        to_tensor (bool): if True, converts the samples into PyTorch Tensor.

    Attributes:
        classes (list): List of the class name tuples.
        class_to_idx (dict): Dict with items (class_name, class_index).
        wnids (list): List of the WordNet IDs.
        wnid_to_idx (dict): Dict with items (wordnet_id, class_index).
    """
    def __init__(self, root: str, mode: str = 'train', 
                 transforms_pt: Optional[Callable] = None,
                 transforms_cv: Optional[Callable] = None,
                 to_tensor: bool = True):
        super(ImageNetClassification, self).__init__(transforms_pt, transforms_cv, to_tensor)
        assert mode in ['train', 'val'], "split has to be 'train' or 'val'"
        self.root = root
        self.mode = mode

        # self.parse_archives()
        self.wnid_to_classes = load_meta_file(self.root)[0]

        self.wnids = [d.name for d in os.scandir(self.data_folder) if d.is_dir()]
        self.wnids.sort()
        assert len(self.wnids) == 1000
        self.num_classes = len(self.wnids)
        self.wnid_to_idx = {wnid_name: i for i, wnid_name in enumerate(self.wnids)}
        self.classes = [self.wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx
                             for idx, clss in enumerate(self.classes)
                             for cls in clss}
        # print(self.classes)
        # print(self.class_to_idx)
        if not os.path.exists(f"{self.data_folder}.csv"):
            print("buiding info for the dataset ...")
            self.build_dataset_info()
        self.info = pd.read_csv(f"{self.data_folder}.csv")
        self.images = self.info['image']

    def build_dataset_info(self):
        info = []
        for wnid in self.wnids:
            sub_folder = os.path.join(self.data_folder, wnid)
            class_name = ", ".join(self.wnid_to_classes[wnid])
            label = self.wnid_to_idx[wnid]
            files = os.listdir(sub_folder)
            info += [[sub_folder, f, wnid, class_name, label] for f in files]
        info = pd.DataFrame(data=info, columns=['directory', 'image', 'wnid', 'class_name', 'label'])
        info.to_csv(f"{self.data_folder}.csv", index=False)       

    def parse_archives(self) -> None:
        if not check_integrity(os.path.join(self.root, META_FILE)):
            print("parse devkit ...")
            parse_devkit_archive(self.root)

        if not os.path.isdir(self.data_folder):
            if self.mode == 'train':
                print("parse training set ...")
                parse_train_archive(self.root)
            elif self.mode == 'val':
                print("parse validation set ...")
                parse_val_archive(self.root)

    @property
    def data_folder(self) -> str:
        return os.path.join(self.root, self.mode)

    def get_images(self):
        r"""gets image names in the dataset.
        """
        return self.image_names

    def getitem(self, index):
        r"""gets image and target for a single sample.
        
        Args:
            index (int): index of sample in the dataset.
        Returns:
            tuple: Tuple (image, target).
        """
        image_name = os.path.join(self.info['directory'].iloc[index], self.info['image'].iloc[index])
        label = self.info['label'].iloc[index]
        img = cv2.imread(image_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = self.get_target_single_item("labels", label, type="labels")
        target.update(self.get_target_single_item("image_id", index, type=None))
        target.update(self.get_target_single_item("image_name", image_name, type=None))
        return img, target


def load_meta_file(root: str, file: Optional[str] = None) -> Tuple[Dict[str, str], List[str]]:
    if file is None:
        file = META_FILE
    file = os.path.join(root, file)

    if check_integrity(file):
        return torch.load(file)
    else:
        msg = ("The meta file {} is not present in the root directory or is corrupted. "
               "This file is automatically created by the ImageNet dataset.")
        raise RuntimeError(msg.format(file, root))


def _verify_archive(root: str, file: str, md5: str) -> None:
    if not check_integrity(os.path.join(root, file), md5):
        msg = ("The archive {} is not present in the root directory or is corrupted. "
               "You need to download it externally and place it in {}.")
        raise RuntimeError(msg.format(file, root))


def parse_devkit_archive(root: str, file: Optional[str] = None) -> None:
    """Parse the devkit archive of the ImageNet2012 classification dataset and save
    the meta information in a binary file.

    Args:
        root (str): Root directory containing the devkit archive
        file (str, optional): Name of devkit archive. Defaults to
            'ILSVRC2012_devkit_t12.tar.gz'
    """
    import scipy.io as sio

    def parse_meta_mat(devkit_root: str) -> Tuple[Dict[int, str], Dict[str, str]]:
        metafile = os.path.join(devkit_root, "data", "meta.mat")
        meta = sio.loadmat(metafile, squeeze_me=True)['synsets']
        nums_children = list(zip(*meta))[4]
        meta = [meta[idx] for idx, num_children in enumerate(nums_children)
                if num_children == 0]
        idcs, wnids, classes = list(zip(*meta))[:3]
        classes = [tuple(clss.split(', ')) for clss in classes]
        idx_to_wnid = {idx: wnid for idx, wnid in zip(idcs, wnids)}
        wnid_to_classes = {wnid: clss for wnid, clss in zip(wnids, classes)}
        return idx_to_wnid, wnid_to_classes

    def parse_val_groundtruth_txt(devkit_root: str) -> List[int]:
        file = os.path.join(devkit_root, "data",
                            "ILSVRC2012_validation_ground_truth.txt")
        with open(file, 'r') as txtfh:
            val_idcs = txtfh.readlines()
        return [int(val_idx) for val_idx in val_idcs]

    @contextmanager
    def get_tmp_dir() -> Iterator[str]:
        tmp_dir = tempfile.mkdtemp()
        try:
            yield tmp_dir
        finally:
            shutil.rmtree(tmp_dir)

    archive_meta = ARCHIVE_META["devkit"]
    if file is None:
        file = archive_meta[0]
    md5 = archive_meta[1]

    _verify_archive(root, file, md5)

    with get_tmp_dir() as tmp_dir:
        extract_archive(os.path.join(root, file), tmp_dir)

        devkit_root = os.path.join(tmp_dir, "ILSVRC2012_devkit_t12")
        idx_to_wnid, wnid_to_classes = parse_meta_mat(devkit_root)
        val_idcs = parse_val_groundtruth_txt(devkit_root)
        val_wnids = [idx_to_wnid[idx] for idx in val_idcs]

        torch.save((wnid_to_classes, val_wnids), os.path.join(root, META_FILE))


def parse_train_archive(root: str, file: Optional[str] = None, folder: str = "train") -> None:
    """Parse the train images archive of the ImageNet2012 classification dataset and
    prepare it for usage with the ImageNet dataset.

    Args:
        root (str): Root directory containing the train images archive
        file (str, optional): Name of train images archive. Defaults to
            'ILSVRC2012_img_train.tar'
        folder (str, optional): Optional name for train images folder. Defaults to
            'train'
    """
    archive_meta = ARCHIVE_META["train"]
    if file is None:
        file = archive_meta[0]
    md5 = archive_meta[1]

    _verify_archive(root, file, md5)

    train_root = os.path.join(root, folder)
    print(f"train_root", train_root)
    extract_archive(os.path.join(root, file), train_root)

    archives = [os.path.join(train_root, archive) for archive in os.listdir(train_root)]
    for archive in archives:
        print(archive)
        extract_archive(archive, os.path.splitext(archive)[0], remove_finished=True)


def parse_val_archive(
    root: str, file: Optional[str] = None, wnids: Optional[List[str]] = None, folder: str = "val"
) -> None:
    """Parse the validation images archive of the ImageNet2012 classification dataset
    and prepare it for usage with the ImageNet dataset.

    Args:
        root (str): Root directory containing the validation images archive
        file (str, optional): Name of validation images archive. Defaults to
            'ILSVRC2012_img_val.tar'
        wnids (list, optional): List of WordNet IDs of the validation images. If None
            is given, the IDs are loaded from the meta file in the root directory
        folder (str, optional): Optional name for validation images folder. Defaults to
            'val'
    """
    archive_meta = ARCHIVE_META["val"]
    if file is None:
        file = archive_meta[0]
    md5 = archive_meta[1]
    if wnids is None:
        wnids = load_meta_file(root)[1]

    _verify_archive(root, file, md5)

    val_root = os.path.join(root, folder)
    extract_archive(os.path.join(root, file), val_root)

    images = sorted([os.path.join(val_root, image) for image in os.listdir(val_root)])

    for wnid in set(wnids):
        os.mkdir(os.path.join(val_root, wnid))

    for wnid, img_file in zip(wnids, images):
        shutil.move(img_file, os.path.join(val_root, wnid, os.path.basename(img_file)))
