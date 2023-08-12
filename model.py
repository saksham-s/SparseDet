import pdb
import torch
import torch.nn as nn
from typing import Optional, Tuple, List
import torch.nn.functional as F
import detectron2
from detectron2.config import configurable
from detectron2.modeling import GeneralizedRCNN
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import ImageList, Boxes
from detectron2.structures.boxes import pairwise_iou
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from custom_modules.custom_mod import merge_gt_teacher
from detectron2.layers import ShapeSpec
from detectron2.modeling.box_regression import Box2BoxTransform
from torchvision.ops import nms

import random
import numpy as np
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
torch.backends.cudnn.benchmark = False

def apply_deltas_broadcast(box2box_transform, deltas, boxes):
    """
    Apply transform deltas to boxes. Similar to `box2box_transform.apply_deltas`,
    but allow broadcasting boxes when the second dimension of deltas is a multiple
    of box dimension.
    Args:
        box2box_transform (Box2BoxTransform or Box2BoxTransformRotated): the transform to apply
        deltas (Tensor): tensor of shape (N,B) or (N,KxB)
        boxes (Tensor): tensor of shape (N,B)
    Returns:
        Tensor: same shape as deltas.
    """
    assert deltas.dim() == boxes.dim() == 2, f"{deltas.shape}, {boxes.shape}"
    N, B = boxes.shape
    assert (
        deltas.shape[1] % B == 0
    ), f"Second dim of deltas should be a multiple of {B}. Got {deltas.shape}"
    K = deltas.shape[1] // B
    ret = box2box_transform.apply_deltas(
        deltas.view(N * K, B), boxes.unsqueeze(1).expand(N, K, B).reshape(N * K, B)
    )
    return ret.view(N, K * B)

@META_ARCH_REGISTRY.register()
class FixMatchGeneralizedRCNN(GeneralizedRCNN):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """
    # pass the model_initi here too
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        mask_boxes: int = 0,
        mask_thresh: int = 0.9,
        mask_boxes_rpn: bool = False,
        det_thresh: float = 0.0,
        distillation_loss_weight: float = 0.0,
        cosine: bool = False,
        box2box_transform: Optional[Box2BoxTransform] = None,
        consistency_reg=False
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element,
                representing the per-channel mean and std to be used to normalize
                the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
            teacher_loss_on_rpn: Put loss on rpn using teacher's predictions
            self_train: use own predictions to guide training
        """
        super().__init__(backbone=backbone,
        proposal_generator=proposal_generator,
        roi_heads=roi_heads,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
        input_format=input_format,
        vis_period=vis_period,
        )
        self.mask_boxes = mask_boxes
        self.mask_thresh = mask_thresh
        self.mask_boxes_rpn = mask_boxes_rpn
        self.distillation_loss_weight = distillation_loss_weight
        self.box2box_transform = box2box_transform
        self.det_thresh = det_thresh
        self.consistency_reg = consistency_reg

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        output_shape = backbone.output_shape()
        output_shape = {'p2': ShapeSpec(channels=2*output_shape['p2'][0], height=None, width=None, stride=4), \
            'p3': ShapeSpec(channels=2*output_shape['p3'][0], height=None, width=None, stride=8), \
            'p4': ShapeSpec(channels=2*output_shape['p4'][0], height=None, width=None, stride=16), \
            'p5': ShapeSpec(channels=2*output_shape['p5'][0], height=None, width=None, stride=32), \
            'p6': ShapeSpec(channels=2*output_shape['p6'][0], height=None, width=None, stride=64)}
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, output_shape),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "mask_boxes": cfg.MASK_BOXES,
            "mask_thresh": cfg.MASK_BOXES_THRESH,
            "mask_boxes_rpn": cfg.MASK_BOXES_RPN,
            "distillation_loss_weight": cfg.DISTILLATION_LOSS_WEIGHT,
            "det_thresh": cfg.DET_THRESH,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
            "consistency_reg": cfg.CONSISTENCY_REGULARIZATION
        }

    def custom_preprocess_image(self, batched_inputs, prefix=''):
        """
        Normalize, pad and batch the input images.
        """
        images = [x[prefix+"image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def predict_boxes(self, predictions, proposals):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        _, proposal_deltas = predictions
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = [p.proposal_boxes for p in proposals]
        proposal_boxes = proposal_boxes[0].cat(proposal_boxes).tensor
        predict_boxes = apply_deltas_broadcast(
            self.box2box_transform, proposal_deltas, proposal_boxes
        )  # Nx(KxB)
        return predict_boxes.split(num_prop_per_image)


    def get_iou_matrix(self, box_arr1, box_arr2):
        x11, y11, x12, y12 = torch.split(box_arr1, 1, dim=1)
        x21, y21, x22, y22 = torch.split(box_arr2, 1, dim=1)
        xA = torch.maximum(x11, x21.T)
        yA = torch.maximum(y11, y21.T)
        xB = torch.minimum(x12, x22.T)
        yB = torch.minimum(y12, y22.T)
        interArea = torch.clamp((xB - xA + 1e-9), min=0) * torch.clamp((yB - yA + 1e-9), min=0)
        boxAArea = (x12 - x11 + 1e-9) * (y12 - y11 + 1e-9)
        boxBArea = (x22 - x21 + 1e-9) * (y22 - y21 + 1e-9)
        iou = interArea / (boxAArea + boxBArea.T - interArea)
        return iou

    def get_proposal_mask(self, wa_proposals, wa_gt_instances, mask_thresh, iou_mask_=True):
        new_wa_prop = []
        masked_prop = []
        for i in range(len(wa_proposals)):
            if(iou_mask_):
                iou_mat = self.get_iou_matrix(wa_proposals[i].get('proposal_boxes').tensor, wa_gt_instances[i].get('gt_boxes').tensor)
                score_mask = torch.sigmoid(wa_proposals[i].get('objectness_logits')) > mask_thresh
                iou_mat[iou_mat>0.2] = 10
                iou_mat[iou_mat<=0.2] = 1
                iou_mat[iou_mat==10] = 0
                iou_mask = torch.all(iou_mat.detach(), axis=1)
                final_mask = score_mask & iou_mask
                masked_prop.append(wa_proposals[i][final_mask])
                final_mask = ~final_mask
                new_wa_prop.append(wa_proposals[i][final_mask])
            else:
                score_mask = torch.sigmoid(wa_proposals[i].get('objectness_logits')) > mask_thresh
                masked_prop.append(wa_proposals[i][score_mask])
                final_mask = ~score_mask
                new_wa_prop.append(wa_proposals[i][final_mask])
        return new_wa_prop, masked_prop

    def get_roi_predictions_masked(self, features, proposals):
        if isinstance(self.roi_heads,
                      detectron2.modeling.roi_heads.roi_heads.Res5ROIHeads):
            proposal_boxes = [x.proposal_boxes for x in proposals]
            box_features = self.roi_heads._shared_roi_transform([features[f]
                                                                 for f in
                                                                 self.roi_heads.in_features],
                                                                proposal_boxes)
            predictions = self.roi_heads.box_predictor(box_features.mean(dim=[2,3]))
            box_features = box_features.mean(dim=[2,3])

        elif isinstance(self.roi_heads,
                        detectron2.modeling.roi_heads.roi_heads.StandardROIHeads):
            features = [features[f] for f in self.roi_heads.in_features]
            box_features = self.roi_heads.box_pooler(features, [x.proposal_boxes for x in proposals])
            box_features = self.roi_heads.box_head(box_features)
            predictions = self.roi_heads.box_predictor(box_features)
        return box_features

    def get_roi_predictions(self, features, proposals, targets):
        if self.training:
            proposals = self.roi_heads.label_and_sample_proposals(proposals, targets)
        if isinstance(self.roi_heads,
                      detectron2.modeling.roi_heads.roi_heads.Res5ROIHeads):
            proposal_boxes = [x.proposal_boxes for x in proposals]
            box_features = self.roi_heads._shared_roi_transform([features[f]
                                                                 for f in
                                                                 self.roi_heads.in_features],
                                                                proposal_boxes)
            predictions = self.roi_heads.box_predictor(box_features.mean(dim=[2,3]))

        elif isinstance(self.roi_heads,
                        detectron2.modeling.roi_heads.roi_heads.StandardROIHeads):
                        
            features = [features[f] for f in self.roi_heads.in_features]
            box_features = self.roi_heads.box_pooler(features, [x.proposal_boxes for x in proposals])
            box_features = self.roi_heads.box_head(box_features)
            predictions = self.roi_heads.box_predictor(box_features)

        pred_instances, _ = self.roi_heads.box_predictor.inference(predictions,
                                                         proposals)
        if self.training:
            losses = self.roi_heads.box_predictor.losses(predictions, proposals)
            return losses, pred_instances, proposals, box_features
        pred_instances = self.roi_heads.forward_with_given_boxes(features,
                                                       pred_instances)
        return {}, pred_instances, proposals, box_features

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                See :meth:`postprocess` for details.
        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)
        # Weakly augmented images
        wa_images = self.custom_preprocess_image(batched_inputs)
        # Strongly augmented images
        sa_images = self.custom_preprocess_image(batched_inputs, prefix='sa_')
        wa_gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        sa_gt_instances = [x["sa_instances"].to(self.device) for x in
                            batched_inputs]
        # Get a mask for all images which don't have gt instances
        present_ids = [i for i, l in enumerate(wa_gt_instances) if l.get('gt_boxes').tensor.shape[0] > 0]
        absent_ids = [i for i, l in enumerate(wa_gt_instances) if
                      l.get('gt_boxes').tensor.shape[0] <= 0]
        # Get features for weakly augmented images
        
        wa_features = self.backbone(wa_images.tensor)
        # For strongly augmented images, get backbone features
        sa_features = self.backbone(sa_images.tensor)

        concat_features = {}
        for key in wa_features.keys():
            concat_features[key] = torch.cat((wa_features[key], sa_features[key]), axis=1)

        # split the features, images and gt_instances into images containing gt
        # and without gt
        if present_ids:
            w_gt_wa_images = ImageList.from_tensors([wa_images[i] for i in
                                        present_ids],
                                                    self.backbone.size_divisibility)
            w_gt_wa_features_concat = {k: v[torch.Tensor(present_ids).long()] for k, v in
                                concat_features.items()}
            w_gt_wa_features = {k: v[torch.Tensor(present_ids).long()] for k, v in
                                wa_features.items()}
            w_gt_wa_gt_instances = [wa_gt_instances[i] for i in present_ids]

            # doing a single forward pass with concatenated features
            w_gt_wa_proposals, wa_proposal_losses = self.proposal_generator(w_gt_wa_images, w_gt_wa_features_concat, w_gt_wa_gt_instances)

            if(batched_inputs[0]['iter']>self.mask_boxes and self.mask_boxes_rpn):
                w_gt_wa_proposals, masked_proposals = self.get_proposal_mask(w_gt_wa_proposals, w_gt_wa_gt_instances, self.mask_thresh)
                wa_masked_counts = []
                for p in range(len(masked_proposals)):
                    masked_boxes = masked_proposals[p].proposal_boxes.tensor
                    masked_boxes_score = torch.sigmoid(masked_proposals[p].objectness_logits)
                    if(len(masked_boxes_score)>1):
                        masked_boxes_nms_ind = nms(boxes=masked_boxes, scores=masked_boxes_score, iou_threshold=0.3)
                        masked_proposals[p] = masked_proposals[p][masked_boxes_nms_ind]
                    wa_masked_counts.append(len(masked_proposals[p]))
                if(self.distillation_loss_weight):
                    wa_boxfeat_masked = self.get_roi_predictions_masked(w_gt_wa_features, masked_proposals)

            wa_detector_losses, w_gt_wa_pred_instances, w_gt_wa_nms_proposals, w_gt_wa_box_features= self.get_roi_predictions(w_gt_wa_features,
                                                                   w_gt_wa_proposals, w_gt_wa_gt_instances)
            w_gt_wa_box_features = torch.split(w_gt_wa_box_features, [len(l)
                                                                      for l in
                                                                      w_gt_wa_nms_proposals],
                                               dim=0)
        else:
            wa_proposal_losses = {'loss_rpn_cls': None,
                                  'loss_rpn_loc': None}
            wa_detector_losses = {'loss_cls': None,
                                  'loss_box_reg':None}


        if absent_ids:
            wo_gt_wa_images = ImageList.from_tensors([wa_images[i] for i in absent_ids],
                                                     self.backbone.size_divisibility).to(self.device)
            wo_gt_wa_features_concat = {k: v[torch.Tensor(absent_ids).long()] for k, v in
                                concat_features.items()}
            wo_gt_wa_features = {k: v[torch.Tensor(absent_ids).long()] for k, v in
                                wa_features.items()}
            wo_gt_wa_gt_instances = [wa_gt_instances[i] for i in absent_ids]
            wo_gt_wa_proposals, _ =\
                    self.proposal_generator(wo_gt_wa_images, wo_gt_wa_features_concat, wo_gt_wa_gt_instances)

            if(batched_inputs[0]['iter']>self.mask_boxes and self.mask_boxes_rpn):
                wo_gt_wa_proposals, masked_proposals_abs = self.get_proposal_mask(wo_gt_wa_proposals, wo_gt_wa_gt_instances, self.mask_thresh, iou_mask_=False)
                wa_masked_counts_abs = []
                for p in range(len(masked_proposals_abs)):
                    masked_boxes = masked_proposals_abs[p].proposal_boxes.tensor
                    masked_boxes_score = torch.sigmoid(masked_proposals_abs[p].objectness_logits)
                    if(len(masked_boxes_score)>1):
                        masked_boxes_nms_ind = nms(boxes=masked_boxes, scores=masked_boxes_score, iou_threshold=0.3)
                        masked_proposals_abs[p] = masked_proposals_abs[p][masked_boxes_nms_ind]
                    wa_masked_counts_abs.append(len(masked_proposals_abs[p]))
                if(self.distillation_loss_weight):
                    wa_boxfeat_masked_abs = self.get_roi_predictions_masked(wo_gt_wa_features, masked_proposals_abs)

            # Get the detection head output
            _, wo_gt_wa_pred_instances, wo_gt_wa_nms_proposals, wo_gt_wa_box_features =\
                    self.get_roi_predictions(wo_gt_wa_features, wo_gt_wa_proposals,
                                             wo_gt_wa_gt_instances)
            wo_gt_wa_box_features = torch.split(wo_gt_wa_box_features, [len(l)
                                                                        for l
                                                                        in
                                                                        wo_gt_wa_nms_proposals],
                                                dim=0)
        else:
            wo_gt_wa_images, wo_gt_wa_features, wo_gt_gt_instances = [], {}, []

        # combine the outputs
        wa_proposals = []
        wa_pred_instances = []
        masked_proposals_merged = []
        wa_boxfeat_masked_ = []
        p_count=0
        a_count=0
        wa_nms_proposals = []
        wa_box_features = []
        # pdb.set_trace()
        for i in range(len(batched_inputs)):
            if i in present_ids:
                req_idx = present_ids.index(i)
                wa_proposals.append(w_gt_wa_proposals[req_idx])
                wa_pred_instances.append(w_gt_wa_pred_instances[req_idx])
                if(batched_inputs[0]['iter']>self.mask_boxes and self.mask_boxes_rpn and self.distillation_loss_weight):
                    masked_proposals_merged.append(masked_proposals[p_count])
                    lower=0
                    if(p_count>0):
                        lower = np.sum(wa_masked_counts[:p_count])
                    upper = np.sum(wa_masked_counts[:p_count+1])
                    wa_boxfeat_masked_.append(wa_boxfeat_masked[lower:upper])
                    p_count+=1
                if self.consistency_reg:
                    wa_nms_proposals.append(w_gt_wa_nms_proposals[req_idx])
                    wa_box_features.append(w_gt_wa_box_features[req_idx])
            elif i in absent_ids:
                req_idx = absent_ids.index(i)
                wa_proposals.append(wo_gt_wa_proposals[req_idx])
                wa_pred_instances.append(wo_gt_wa_pred_instances[req_idx])
                if(batched_inputs[0]['iter']>self.mask_boxes and self.mask_boxes_rpn and self.distillation_loss_weight):
                    masked_proposals_merged.append(masked_proposals_abs[a_count])
                    lower=0
                    if(a_count>0):
                        lower= np.sum(wa_masked_counts_abs[:a_count])
                    upper = np.sum(wa_masked_counts_abs[:a_count+1])
                    a_count+=1
                    wa_boxfeat_masked_.append(wa_boxfeat_masked_abs[lower:upper])
                if self.consistency_reg:
                    wa_nms_proposals.append(wo_gt_wa_nms_proposals[req_idx])
                    wa_box_features.append(wo_gt_wa_box_features[req_idx])
            else:
                print('getting stuck here?')
                pdb.set_trace()
        if self.consistency_reg:
            wa_box_features = torch.cat(wa_box_features, dim=0)
        sa_appended_gt = []
        for i in range(len(wa_pred_instances)):
            sa_appended_gt.append(merge_gt_teacher(wa_pred_instances[i].pred_classes,
                                                   wa_pred_instances[i].pred_boxes.tensor,
                                                   wa_pred_instances[i].scores,
                                                   wa_pred_instances[i].image_size,
                                                   sa_gt_instances[i].gt_boxes.tensor,
                                                   sa_gt_instances[i].gt_classes,
                                                   0.9, 0.5, -1,
                                                   use_score_thresh=True))

        # Only compute losses for instances which have gt boxes
        if present_ids and batched_inputs[0]['iter']>self.mask_boxes and self.mask_boxes_rpn and self.distillation_loss_weight:
            wa_boxfeat_masked = torch.cat(wa_boxfeat_masked_)
            sa_boxfeat_masked = self.get_roi_predictions_masked(sa_features, masked_proposals_merged)


        keep_ids = [i for i in range(len(sa_appended_gt)) if
                    sa_appended_gt[i].get('gt_boxes').tensor.shape[0] > 0]
        rem_ids = [i for i in range(len(sa_appended_gt)) if
                sa_appended_gt[i].get('gt_boxes').tensor.shape[0] <= 0]

        if keep_ids:
            sa_k_features = {k: v[torch.Tensor(keep_ids).long()] for k, v in
                                sa_features.items()}
            sa_k_appended_gt = [sa_appended_gt[i] for i in keep_ids]
            sa_k_proposals = [wa_proposals[i] for i in keep_ids]


            sa_detector_losses, sa_pred_instances, sa_nms_proposals, sa_box_features = self.get_roi_predictions(sa_k_features,
                                                                       sa_k_proposals,
                                                                       sa_k_appended_gt)
            if rem_ids:
                sa_box_features = torch.split(sa_box_features, [len(v) for v in
                                                                sa_nms_proposals],
                                              dim=0)
        else:
            sa_r_features = {k: v[torch.Tensor(rem_ids).long()] for k, v in
                                sa_features.items()}
            sa_r_appended_gt = [sa_appended_gt[i] for i in rem_ids]
            sa_r_proposals = [wa_proposals[i] for i in rem_ids]


            sa_detector_losses, sa_pred_instances, sa_nms_proposals,sa_box_features = self.get_roi_predictions(sa_r_features,
                                                                       sa_r_proposals,
                                                                       sa_r_appended_gt)

            sa_detector_losses = {k: 0*v for k, v in
                                  sa_detector_losses.items()}

        # Distillation losses between common boxes between sa_nms_proposals and wa_nms_proposals
        if self.consistency_reg:
            assert self.distillation_loss_weight > 0
            if keep_ids and rem_ids:
                # This is needed as for sa features boxes are computed only if
                # all images have gts or all images don't
                sa_r_features = {k: v[torch.Tensor(rem_ids).long()] for k, v in
                                sa_features.items()}
                sa_r_appended_gt = [sa_appended_gt[i] for i in rem_ids]
                sa_r_proposals = [wa_proposals[i] for i in rem_ids]
                _, _, sa_r_nms_proposals,sa_r_box_features = self.get_roi_predictions(sa_r_features,
                                                                           sa_r_proposals,
                                                                           sa_r_appended_gt)
                sa_r_box_features = torch.split(sa_r_box_features, [len(v) for
                                                                    v in
                                                                    sa_r_nms_proposals],
                                                dim=0)
                new_sa_nms_proposals = []
                new_sa_box_features = []
                for i in range(len(batched_inputs)):
                    if i in keep_ids:
                        req_idx = keep_ids.index(i)
                        new_sa_nms_proposals.append(sa_nms_proposals[req_idx])
                        new_sa_box_features.append(sa_box_features[req_idx])
                    else:
                        req_idx = rem_ids.index(i)
                        new_sa_nms_proposals.append(sa_r_nms_proposals[req_idx])
                        new_sa_box_features.append(sa_r_box_features[req_idx])

                sa_box_features = torch.cat(new_sa_box_features, dim=0)
                sa_nms_proposals = new_sa_nms_proposals

            start = 0
            cur_loss = nn.MSELoss()
            distillation_loss_dict = {}

            present_count = 0
            for i in range(len(batched_inputs)):
                try:
                    req_wa_boxes = wa_nms_proposals[i].get('proposal_boxes')
                    req_sa_boxes = sa_nms_proposals[i].get('proposal_boxes')
                except:
                    pdb.set_trace()
                iou_mat = pairwise_iou(req_wa_boxes, req_sa_boxes)
                match = (iou_mat == 1).nonzero()
                if match.shape[0] > 0:
                    # match will be a Nx2 tensor with first column containing box
                    # index of wa_boxes and second column of sa_boxes
                    # Get wa features
                    match += start
                    req_wa_box_features = wa_box_features[match[:,0]].mean(-1).mean(-1)  # Nx2048
                    req_sa_box_features = sa_box_features[match[:,1]].mean(-1).mean(-1)  # Nx2048
                    distillation_loss = cur_loss(req_sa_box_features,
                                                 req_wa_box_features)
                    present_count += 1
                    # Next image start from previous images end point
                    if 'distillation_loss' not in distillation_loss_dict:
                        distillation_loss_dict['distillation_loss'] = self.distillation_loss_weight * distillation_loss
                    else:
                        distillation_loss_dict['distillation_loss'] += self.distillation_loss_weight * distillation_loss
                    start += len(wa_nms_proposals[i])
                else:
                    mseloss = nn.MSELoss()
                    distillation_loss = 0*mseloss(wa_features['p6'], wa_features['p6'])
                    if 'distillation_loss' not in distillation_loss_dict:
                        distillation_loss_dict['distillation_loss'] = distillation_loss
                    else:
                        distillation_loss_dict['distillation_loss'] += distillation_loss
            if present_count:
                distillation_loss_dict['distillation_loss'] /= present_count
        else:
            mseloss = nn.MSELoss()
            distillation_loss_dict = {}
            distillation_loss = 0*mseloss(wa_features['p6'], wa_features['p6'])
            distillation_loss_dict['distillation_loss'] = distillation_loss


        detector_losses = {}
        proposal_losses = {}
        for k in sa_detector_losses:
            if wa_detector_losses[k]:
                detector_losses[k] = 0.5*(sa_detector_losses[k] +
                                          wa_detector_losses[k])
            else:
                # sa losses will always be present
                detector_losses[k] = sa_detector_losses[k]

        mseloss = nn.MSELoss()
        for k in wa_proposal_losses:
            if wa_proposal_losses[k]:
                proposal_losses[k] = wa_proposal_losses[k]
            else:
                proposal_losses[k] = 0*mseloss(wa_features['p6'], wa_features['p6'])

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        
        if(self.distillation_loss_weight and batched_inputs[0]['iter']>self.mask_boxes and self.mask_boxes_rpn and present_ids): #corrected
            distill_loss = nn.MSELoss()
            if(len(sa_boxfeat_masked)>0 and len(wa_boxfeat_masked)>0):
                distillation_loss = self.distillation_loss_weight*distill_loss(sa_boxfeat_masked,
                                                           wa_boxfeat_masked)
            else:
                distillation_loss = 0*mseloss(wa_features['p6'], wa_features['p6'])
            # Distillation_loss_dict will always exist
            distillation_loss_dict['distillation_loss'] += distillation_loss
            losses.update(distillation_loss_dict)
        else:
            distillation_loss = 0*mseloss(wa_features['p6'], wa_features['p6'])
            distillation_loss_dict['distillation_loss'] += distillation_loss
            losses.update(distillation_loss_dict)

        return losses

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        wa_images = self.custom_preprocess_image(batched_inputs)
        sa_images = self.custom_preprocess_image(batched_inputs, prefix='sa_')
        
        wa_features = self.backbone(wa_images.tensor)
        sa_features = self.backbone(sa_images.tensor)

        concat_features = {}
        for key in wa_features.keys():
            concat_features[key] = torch.cat((wa_features[key], sa_features[key]), axis=1)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(wa_images, concat_features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(wa_images, wa_features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            return GeneralizedRCNN._postprocess(results, batched_inputs, wa_images.image_sizes)
        else:
            return results
