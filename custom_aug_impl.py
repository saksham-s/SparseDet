import numpy as np
import sys
from typing import Tuple
import torch
from PIL import Image
import random
from detectron2.data.transforms import Augmentation
import numpy as np
import math
import numbers
from fvcore.transforms.transform import NoOpTransform
from custom_transforms import Custom_Erase, Shear
__all__ = [
	"BBox_Erase",
	"RandomShear",
]
import random
import numpy as np
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
torch.backends.cudnn.benchmark = False
class RandomShear(Augmentation):
    def __init__(self, shear_factor=0.2):
        super().__init__()
        assert -1 <= shear_factor <= 1
        self.shear_factor = random.uniform(-shear_factor, shear_factor)

    def get_transform(self, image):
        img_center = np.array(image.shape[:2])[::-1]/2
        img_center = np.hstack((img_center, img_center))
        M = np.array([[1, abs(self.shear_factor), 0],[0,1,0]])
        nW =  image.shape[1] + abs(self.shear_factor*image.shape[0])
        return Shear(shear_factor=self.shear_factor, img_center=img_center, M=M, nW=nW, w=image.shape[1])


class BBox_Erase(Augmentation):
    """
    This method returns a copy of this image, with erasing applied to a random bounding box.
    """

    def __init__(self, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
        """
        Args:
            n_max : Number of bounding boxes to apply the transform to.
        """
        super().__init__()
        # self.n_max = n_max
        assert isinstance(scale, tuple) and isinstance(ratio, tuple)
        # TODO:Put a check on values
        self.scale = scale
        self.ratio = ratio
        self.value = 0
        self.inplace = False
        # self._init(locals())

    def get_params(self, img, bbox):
        """Get parameters for ``erase`` for a random erasing.
        Args:
            img (Tensor): Tensor image of size (C, H, W) to be erased.
            scale: range of proportion of erased area against input image.
            ratio: range of aspect ratio of erased area.
        Returns:
            tuple: params (i, j, h, w, v) to be passed to ``erase`` for random erasing.
        """
        img_c, img_h, img_w = img.shape[0], int(bbox[3]-bbox[1]), int(bbox[2]-bbox[0])
        area = img_h * img_w

        for attempt in range(10):
            erase_area = random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))
            if h < img_h and w < img_w:
                i = random.randint(int(bbox[1]), int(bbox[1]) + img_h - h)
                j = random.randint(int(bbox[0]), int(bbox[0]) + img_w - w)
                if isinstance(self.value, numbers.Number):
                    v = self.value
                elif isinstance(self.value, torch._six.string_classes):
                    v = torch.empty([img_c, h, w], dtype=torch.float32).normal_()
                elif isinstance(self.value, (list, tuple)):
                    v = torch.tensor(self.value, dtype=torch.float32).view(-1, 1, 1).expand(-1, h, w)
                return i, j, h, w, v

        # Return original image
        return 0, 0, img_h, img_w, img

    def get_transform(self, image, boxes):
        if len(boxes) == 0:
            return NoOpTransform()
        ind = np.random.randint(0, len(boxes))
        x, y, h, w, v = self.get_params(image, boxes[ind])
        return Custom_Erase(x, y, h, w, v, self.inplace)

