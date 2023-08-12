from detectron2.config import CfgNode as CN

def add_student_teacher_config(cfg):
    cfg.MODEL.USE_SCORE_THRESH = False
    cfg.THRESH_PATTERN = False
    cfg.FIXMATCH = False
    cfg.FIXMATCH_STRONG_AUG = False
    cfg.FIXMATCH_BBOX_ERASE_SCALE = (0.4, 0.7)
    cfg.FIXMATCH_BBOX_ERASE_SCALE_INFERENCE = (0.01, 0.05)
    cfg.FIXMATCH_BBOX_ERASE_RATIO = (0.3, 3.3)
    cfg.MASK_BOXES = 0
    cfg.MASK_BOXES_THRESH = 0.9
    cfg.MASK_BOXES_RPN = False
    cfg.DET_THRESH = 0.8
    cfg.DISTILLATION_LOSS_WEIGHT = 0.0
    cfg.CONSISTENCY_REGULARIZATION = False
