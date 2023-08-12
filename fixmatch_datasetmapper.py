import copy
import logging
import numpy as np
from typing import List, Optional, Union
import torch
import pdb

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from custom_aug_impl import BBox_Erase, RandomShear



from typing import Union, Optional, List, Tuple, Text, BinaryIO
from PIL import Image, ImageDraw, ImageFont, ImageColor
import numpy as np
from torchvision.utils import save_image

import random
import numpy as np
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
torch.backends.cudnn.benchmark = False

def draw_bounding_boxes(
	image: torch.Tensor,
	boxes: torch.Tensor,
	labels: Optional[List[str]] = None,
	colors: Optional[List[Union[str, Tuple[int, int, int]]]] = None,
	fill: Optional[bool] = False,
	width: int = 4,
	font: Optional[str] = None,
	font_size: int = 10
) -> torch.Tensor:

	"""
	Draws bounding boxes on given image.
	The values of the input image should be uint8 between 0 and 255.
	If filled, Resulting Tensor should be saved as PNG image.

	Args:
		image (Tensor): Tensor of shape (C x H x W)
		boxes (Tensor): Tensor of size (N, 4) containing bounding boxes in (xmin, ymin, xmax, ymax) format. Note that
			the boxes are absolute coordinates with respect to the image. In other words: `0 <= xmin < xmax < W` and
			`0 <= ymin < ymax < H`.
		labels (List[str]): List containing the labels of bounding boxes.
		colors (List[Union[str, Tuple[int, int, int]]]): List containing the colors of bounding boxes. The colors can
			be represented as `str` or `Tuple[int, int, int]`.
		fill (bool): If `True` fills the bounding box with specified color.
		width (int): Width of bounding box.
		font (str): A filename containing a TrueType font. If the file is not found in this filename, the loader may
			also search in other directories, such as the `fonts/` directory on Windows or `/Library/Fonts/`,
			`/System/Library/Fonts/` and `~/Library/Fonts/` on macOS.
		font_size (int): The requested font size in points.
	"""

	if not isinstance(image, torch.Tensor):
		raise TypeError(f"Tensor expected, got {type(image)}")
	elif image.dtype != torch.uint8:
		raise ValueError(f"Tensor uint8 expected, got {image.dtype}")
	elif image.dim() != 3:
		raise ValueError("Pass individual images, not batches")

	ndarr = image.permute(1, 2, 0).numpy()
	img_to_draw = Image.fromarray(ndarr)

	img_boxes = boxes.to(torch.int64).tolist()

	if fill:
		draw = ImageDraw.Draw(img_to_draw, "RGBA")

	else:
		draw = ImageDraw.Draw(img_to_draw)

	txt_font = ImageFont.load_default() if font is None else ImageFont.truetype(font=font, size=font_size)

	for i, bbox in enumerate(img_boxes):
		if colors is None:
			color = None
		else:
			color = colors[i]

		if fill:
			if color is None:
				fill_color = (255, 255, 255, 100)
			elif isinstance(color, str):
				# This will automatically raise Error if rgb cannot be parsed.
				fill_color = ImageColor.getrgb(color) + (100,)
			elif isinstance(color, tuple):
				fill_color = color + (100,)
			draw.rectangle(bbox, width=width, outline=color, fill=fill_color)
		else:
			draw.rectangle(bbox, width=width, outline='red')

		if labels is not None:
			draw.text((bbox[0], bbox[1]), labels[i], fill='red', font=txt_font)

	return torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1)



class FixMatchDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the callable to be used to map your dataset dict into fixmatch  training data.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        teacher_augmentations: List[Union[T.Augmentation, T.Transform]],
        student_augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        use_keypoint: bool = False,
        instance_mask_format: str = "polygon",
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        precomputed_proposal_topk: Optional[int] = None,
        recompute_teacher_boxes: bool = False,
        recompute_student_boxes: bool = False,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.
            keypoint_hflip_indices: see :func:`detection_utils.create_keypoint_hflip_indices`
            precomputed_proposal_topk: if given, will load pre-computed
                proposals from dataset_dict and keep the top k proposals for each image.
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
        """
        if recompute_teacher_boxes or recompute_student_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"
        # fmt: off
        self.is_train               = is_train
        self.teacher_augmentations  = T.AugmentationList(teacher_augmentations)
        self.student_augmentations  = T.AugmentationList(student_augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.instance_mask_format   = instance_mask_format
        self.use_keypoint           = use_keypoint
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.proposal_topk          = precomputed_proposal_topk
        self.recompute_teacher_boxes= recompute_teacher_boxes
        self.recompute_student_boxes= recompute_student_boxes
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode} (teacher): {teacher_augmentations}")
        logger.info(f"[DatasetMapper] Augmentations used in {mode} (student): {student_augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = utils.build_augmentation(cfg, is_train)
        # st_augs = utils.build_augmentation(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train:
            augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            recompute_teacher_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_teacher_boxes = False
        st_augs = []
        if cfg.INPUT.CROP.ENABLED and is_train:
            st_augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            recompute_student_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_student_boxes = False
        # Random Contrast
        st_augs.append(T.RandomContrast(0.5, 1.5))
        # Random Brightness
        st_augs.append(T.RandomBrightness(0.5, 1.5))
        # Random Saturation
        st_augs.append(T.RandomSaturation(0.5, 1.5))
        # Random Lighting
        st_augs.append(T.RandomLighting(1.2))
        #  Randomly erase bounding box part
        if cfg.FIXMATCH_STRONG_AUG:
            st_augs.append(BBox_Erase(cfg.FIXMATCH_BBOX_ERASE_SCALE,
                                      cfg.FIXMATCH_BBOX_ERASE_RATIO))

        ret = {
            "is_train": is_train,
            "teacher_augmentations": augs,
            "student_augmentations": st_augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_teacher_boxes": recompute_teacher_boxes,
            "recompute_student_boxes": recompute_student_boxes,
        }

        if cfg.MODEL.KEYPOINT_ON:
            ret["keypoint_hflip_indices"] = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)

        if cfg.MODEL.LOAD_PROPOSALS:
            ret["precomputed_proposal_topk"] = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        # Student image
        st_image = image.copy()
        utils.check_image_size(dataset_dict, image)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        og_annos = copy.deepcopy(dataset_dict['annotations'])
        annos = [
            utils.transform_instance_annotations(
                obj, [], image.shape[:2], keypoint_hflip_indices=self.keypoint_hflip_indices
            )
            for obj in list(og_annos)
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(
            annos, image.shape[:2], mask_format=self.instance_mask_format
        )


        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.teacher_augmentations(aug_input)

        st_aug_input = T.AugInput(st_image, boxes=instances.get('gt_boxes').tensor, sem_seg=sem_seg_gt)
        # st_transforms = self.student_augmentations(st_aug_input)+T.TransformList([transforms[0]])
        st_transforms = self.student_augmentations(st_aug_input)

        image, sem_seg_gt = aug_input.image, aug_input.sem_seg
        st_image, st_sem_seg_gt = st_aug_input.image, st_aug_input.sem_seg
        # Apply the teacher transforms after all the student transforms are
        # applied to bring both images to the same size
        for transform in transforms:
            params = copy.deepcopy(transform.__dict__)
            if 'w' in params:
                params['w'] = st_image.shape[1]
            if 'h' in params:
                params['h'] = st_image.shape[0]
            st_image = type(transform)(**params).apply_image(st_image)
            st_transforms = st_transforms + T.TransformList([type(transform)(**params)])
        image_shape = image.shape[:2]  # h, w
        st_image_shape = st_image.shape[:2]
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["sa_image"] = torch.as_tensor(np.ascontiguousarray(st_image.transpose(2, 0, 1)))

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))
            dataset_dict["sa_sem_seg"] = torch.as_tensor(st_sem_seg_gt.astype("long"))
        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            # TODO: Do it for student image - I don't think its necessary as we
            # are not using proposals
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            og_annos = copy.deepcopy(dataset_dict['annotations'])
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in list(og_annos)
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )
            og_annos = copy.deepcopy(dataset_dict['annotations'])

            st_annos = [
                utils.transform_instance_annotations(
                    obj, st_transforms, st_image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in list(og_annos)
                if obj.get("iscrowd", 0) == 0
            ]
            st_instances = utils.annotations_to_instances(
                st_annos, st_image_shape, mask_format=self.instance_mask_format
            )
            dataset_dict.pop("annotations")
            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if self.recompute_teacher_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            if self.recompute_student_boxes:
                st_instances.gt_boxes = st_instances.gt_masks.get_bounding_boxes()

            dataset_dict["instances"] = utils.filter_empty_instances(instances,
                                                                    by_box=True)
            dataset_dict["sa_instances"] = utils.filter_empty_instances(st_instances, by_box=True)
        return dataset_dict
