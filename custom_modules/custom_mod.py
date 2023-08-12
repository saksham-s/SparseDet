# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from typing import Optional, Tuple, List
import torch
from torch import nn
import torchvision

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
from detectron2.modeling import GeneralizedRCNN, ProposalNetwork
from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.structures import Boxes, Instances
from detectron2.layers import Linear, ShapeSpec, batched_nms, cat, nonzero_tuple

def merge_gt_teacher(
	pred_classes,
	pred_boxes,
	pred_scores,
	image_shape: Tuple[int, int],
	gt_boxes,
	gt_classes,
	score_thresh: float,
	nms_thresh: float,
	topk_per_image: int,
	use_score_thresh: int,):
	"""
	Single-image inference. Return bounding-box detection results by thresholding
	on scores and applying non-maximum suppression (NMS).

	Args:
		Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
		per image.

	Returns:
		Same as `fast_rcnn_inference`, but for only one image.
	"""

	pred_boxes = Boxes(pred_boxes.reshape(-1, 4))
	pred_boxes.clip(image_shape)
	pred_boxes = pred_boxes.tensor
	if(use_score_thresh):
		pred_boxes = pred_boxes[pred_scores>score_thresh]
		pred_classes = pred_classes[pred_scores>score_thresh]
		pred_scores = pred_scores[pred_scores>score_thresh]

	gt_boxes = Boxes(gt_boxes.reshape(-1, 4))
	gt_boxes.clip(image_shape)
	gt_boxes = gt_boxes.tensor

	boxes = torch.cat((gt_boxes, pred_boxes))
	idxs = torch.cat((gt_classes, pred_classes))
	scores = torch.cat((torch.ones(len(gt_classes)).to(idxs.device), pred_scores))
	keep = torchvision.ops.batched_nms(boxes, scores, idxs, nms_thresh)
	
	if topk_per_image >= 0:
		keep = keep[:topk_per_image]
	boxes, scores = boxes[keep], scores[keep]

	result = Instances(image_shape)
	result.gt_boxes = Boxes(boxes)
	result.gt_classes = idxs[keep]
	return result
