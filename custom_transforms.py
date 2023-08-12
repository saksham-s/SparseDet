import numpy as np
import torch
import torch.nn.functional as F
from fvcore.transforms.transform import (
	CropTransform,
	HFlipTransform,
	NoOpTransform,
	Transform,
	TransformList,
)
from PIL import Image
from torchvision.transforms import functional as F
import cv2

try:
	import cv2  # noqa
except ImportError:
	# OpenCV is an optional dependency at the moment
	pass

__all__ = [
	"Custom_Erase",
	"Shear",
	]


class Custom_Erase(Transform):
    """
    This method returns a copy of this image, rotated the given
    number of degrees counter clockwise around its center.
    """

    def __init__(self, x, y, h, w, v, inplace):
        """
        Args:
            h, w (int): original image size
            angle (float): degrees for rotation
            expand (bool): choose if the image should be resized to fit the whole
                rotated image (default), or simply cropped
            center (tuple (width, height)): coordinates of the rotation center
                if left to None, the center will be fit to the center of each image
                center has no effect if expand=True because it only affects shifting
            interp: cv2 interpolation method, default cv2.INTER_LINEAR
        """
        super().__init__()
        self.x = x
        self.y = y
        self.h = h
        self.w = w
        self.v = 0
        self.inplace = inplace
        # self._set_attributes(locals())
        # self.rm_coords = self.create_rotation_matrix()
        # Needed because of this problem https://github.com/opencv/opencv/issues/11784
        # self.rm_image = self.create_rotation_matrix(offset=-0.5)

    def apply_image(self, img, interp=None):
        copy_im = torch.tensor(img.copy().transpose(2,0,1))
        return F.erase(copy_im, self.x, self.y, self.h, self.w, self.v, self.inplace).numpy().transpose(1,2,0)
    def apply_coords(self, coords):
        return coords



class Shear(Transform):
    """
    This method returns a copy of this image, rotated the given
    number of degrees counter clockwise around its center.
    """

    def __init__(self, shear_factor=0.1, img_center=None, M=None, nW=None, w=None):
        """
        Args:
            h, w (int): original image size
            angle (float): degrees for rotation
            expand (bool): choose if the image should be resized to fit the whole
                rotated image (default), or simply cropped
            center (tuple (width, height)): coordinates of the rotation center
                if left to None, the center will be fit to the center of each image
                center has no effect if expand=True because it only affects shifting
            interp: cv2 interpolation method, default cv2.INTER_LINEAR
        """
        super().__init__()
        self.shear_factor = shear_factor
        self.img_center = img_center
        self.M = M
        self.nW = nW
        self.w = w
        self.new_img_center = 0
        # self._set_attributes(locals())
        # self.rm_coords = self.create_rotation_matrix()
        # Needed because of this problem https://github.com/opencv/opencv/issues/11784
        # self.rm_image = self.create_rotation_matrix(offset=-0.5)

    def apply_image(self, img, interp=None):
        w,h = img.shape[1], img.shape[0]
        if self.shear_factor < 0:
            img = img[:, ::-1, :]
        img = cv2.warpAffine(img, self.M, (int(self.nW), img.shape[0]))
        new_img_center = np.array(img.shape[:2])[::-1]/2
        self.new_img_center = np.hstack((new_img_center, new_img_center))
        if self.shear_factor < 0:
            img = img[:, ::-1, :]
        img = cv2.resize(img, (w,h))
        return img

    def apply_coords(self, coords):
        if coords.shape[0] == 0:
            return coords
        bboxes = []
        for i in range(0, len(coords), 4):
            bboxes.append([coords[i][0], coords[i][1], coords[i+3][0], coords[i+3][1]])
        bboxes = np.array(bboxes).astype(float)
        if(self.shear_factor<0):
            bboxes[:, [0, 2]] += 2*(self.img_center[[0, 2]] - bboxes[:, [0, 2]])
            box_w = abs(bboxes[:, 0] - bboxes[:, 2])
            bboxes[:, 0] -= box_w
            bboxes[:, 2] += box_w
        try:
            bboxes[:,[0,2]] += ((bboxes[:,[1,3]]) * abs(self.shear_factor)).astype(int)
        except:
            print(bboxes.shape, bboxes)
        if(self.shear_factor<0):
            bboxes[:, [0, 2]] += 2*(self.new_img_center[[0, 2]] - bboxes[:, [0, 2]])
            box_w = abs(bboxes[:, 0] - bboxes[:, 2])
            bboxes[:, 0] -= box_w
            bboxes[:, 2] += box_w
        scale_factor_x = self.nW / self.w
        bboxes[:,:4] /= [scale_factor_x, 1, scale_factor_x, 1]
        new_coords = []
        for box in bboxes:
            new_coords.append([box[0], box[1]])
            new_coords.append([box[2], box[1]])
            new_coords.append([box[0], box[3]])
            new_coords.append([box[2], box[3]])
        new_coords = np.array(new_coords)
        return new_coords
