from PIL import Image
import os
import io
from typing import Any, Callable, Optional, Tuple
from skimage import measure
import numpy as np
from .base import BaseDataset


__all__ = ("PromiseSegmentation")


class PromiseSegmentation(BaseDataset):
    r"""Promise dataset.

    Args:
        root (str): root directory where images are downloaded to.
        image_folder (str): image folder in root.
        gt_folder (str): ground truth folder in root.
        margin (int): number of margin is added to the bounding boxes of the object.
        random (bool): if True, the margin is fixed; otherwise, it is randomly generated.
        transforms_pt (callable, optional): a data augmentation object that is implemented by PyTorch.
        transforms_cv (callable, optional): a data augmentation object that is implemented by OpenCV.
        to_tensor (bool): if True, converts the samples into PyTorch Tensor.
    """
    def __init__(
            self,
            root: str,
            image_folder: str,
            gt_folder: str,
            margin: int = 0,
            random: bool = False,
            transforms_pt: Optional[Callable] = None,
            transforms_cv: Optional[Callable] = None,
            to_tensor: bool = True
    ) -> None:
        super(PromiseSegmentation, self).__init__(transforms_pt, transforms_cv, to_tensor)
        self.categories = [{'name': 'lesion', 'id': 0}]         
        image_folder = os.path.join(root, image_folder)
        gt_folder = os.path.join(root, gt_folder)
        self.image_names = os.listdir(image_folder)
        assert len(self.image_names) == len(os.listdir(gt_folder))
        self.ids = list(range(len(self.image_names)))
        self.margin = margin
        self.random = random
        
        self.images = self.load_images(image_folder, in_memory=True)
        self.gt = self.load_images(gt_folder, in_memory=True)
        if self.random:
            np.random.seed(12345)
            self.random_margin = np.random.randint(0, self.margin+1, size=(len(self.images), 4))
        
    def load_images(self, folder: str, in_memory: bool, quiet: bool = False):
        r"""load images from folder.

        Args:
            folder (folder): name of the image folder.
            in_memory (bool): if True, all images are read to memory.
            quiet (bool): if True, do not print log info.
        Returns:
            List[str]: list of file names.
        """
        def load(folder, filename):
            p = os.path.join(folder, filename)
            if in_memory:
                with open(p, 'rb') as data:
                    res = io.BytesIO(data.read())
                return res
            return p
        if in_memory and not quiet:
            print("> Loading the data in memory...")
        files = [load(folder, im) for im in self.image_names] 
        return files

    def getitem(self, index: int) -> Tuple[Any, Any]:
        r"""gets image and target for a single sample.
        
        Args:
            index (int): index of sample in the dataset.
        Returns:
            tuple: Tuple (image, target).
        """
        img = np.array(Image.open(self.images[index]), copy=True)
        gt = np.array(Image.open(self.gt[index]), copy=True)
        img_id = self.ids[index]
        _, boxes, errors = self._binary2boxcoords(gt, index)
        if len(boxes) > 0:
            boxes = np.vstack(boxes)
            errors = np.vstack(errors)
        else:
            boxes = np.empty((0, 4))
            errors = np.empty((0, 4))
        labels = np.array([0]*boxes.shape[0])
        
        target = self.get_target_single_item("masks", gt, type="masks")
        # target.update(self.get_target_single_item("iscrowd", np.array([img_id]), type=None))
        target.update(self.get_target_single_item("labels", labels, type="labels"))
        target.update(self.get_target_single_item("image_id", np.array([img_id]), type=None))
        target.update(self.get_target_single_item("errors", errors, type=None))
        target.update(self.get_target_single_item("bboxes", boxes, type="bboxes"))
        return img, target

    def _binary2boxcoords(self, seg, index):
        assert set(np.unique(seg)).issubset([0, 1])
        assert len(seg.shape) == 2  # ensure the 2d shape

        blobs, n_blob = measure.label(seg, background=0, return_num=True)            
        assert set(np.unique(blobs)) <= set(range(0, n_blob + 1)), np.unique(blobs)

        obj_coords = []
        obj_seg = []
        errs = []
        for b in range(1, n_blob + 1):
            blob_mask = blobs == b
            obj_seg.append(blob_mask)

            assert blob_mask.dtype == np.bool, blob_mask.dtype
            coords = np.argwhere(blob_mask)
            x1, y1 = coords.min(axis=0)
            x2, y2 = coords.max(axis=0)
            xo_1, xo_2, yo_1, yo_2 = x1, x2, y1, y2
            if self.margin > 0:
                if self.random:
                    y1 -= self.random_margin[index, 0]
                    x1 -= self.random_margin[index, 1]
                    y2 += self.random_margin[index, 2]
                    x2 += self.random_margin[index, 3]
                else:
                    y1 -= self.margin
                    x1 -= self.margin
                    y2 += self.margin
                    x2 += self.margin
                y1 = max(0, y1)
                x1 = max(0, x1)
                y2 = min(y2, seg.shape[1] - 1)
                x2 = min(x2, seg.shape[0] - 1)
            obj_coords.append([y1, x1, y2, x2])
            diff_x, diff_y = xo_2 - xo_1 + 1, yo_2 - yo_1 + 1
            err = [(xo_1-x1)/diff_x, (x2-xo_2)/diff_x, (yo_1-y1)/diff_y, (y2-yo_2)/diff_y]
            errs.append(err)
        assert len(obj_coords) == n_blob
        return obj_seg, obj_coords, errs

    def get_images(self):
        r"""gets image names in the dataset.
        """
        return self.image_names
