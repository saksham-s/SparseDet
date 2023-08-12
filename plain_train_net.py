#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""
import pdb
import socket
import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
from model import FixMatchGeneralizedRCNN
from fixmatch_datasetmapper_inference import FixMatchDatasetMapper_Inf

from detectron2.modeling import ROI_HEADS_REGISTRY
from student_teacher_config import add_student_teacher_config
from fixmatch_datasetmapper import FixMatchDatasetMapper

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
from detectron2.data.datasets import register_coco_instances
logger = logging.getLogger("detectron2")
import random
import numpy as np
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] = str(1234)

split1_30p = 'splits/split1_30p.json'
split1_50p = 'splits/split1_50p.json'
split1_70p = 'splits/split1_70p.json'
split2_30p = 'splits/split2_30p.json'
split2_50p = 'splits/split2_50p.json'
split2_70p = 'splits/split2_70p.json'
split3_30p = 'splits/split3_30p.json'
split3_50p = 'splits/split3_50p.json'
split3_70p = 'splits/split3_70p.json'

# train_img_path_coco = <YOUR COCO IMAGE DIR>
# train_ann_path_coco = <YOUR COCO TRAIN JSON>
# val_img_path_coco = <YOUR COCO IMAGE DIR>
# val_ann_path_coco = <YOUR COCO VAL JSON>

train_img_path_coco = '/fs/vulcan-datasets/coco/images/train2017'
train_ann_path_coco ='/fs/vulcan-datasets/coco/annotations/instances_train2017.json'
val_img_path_coco = '/fs/vulcan-datasets/coco/images/val2017'
val_ann_path_coco = '/fs/vulcan-datasets/coco/annotations/instances_val2017.json'

register_coco_instances("split1_30p", {},
                        split1_30p, train_img_path_coco)
register_coco_instances("split1_50p", {},
                        split1_50p, train_img_path_coco)
register_coco_instances("split1_70p", {},
                        split1_70p, train_img_path_coco)

register_coco_instances("split2_30p", {},
                        split2_30p, train_img_path_coco)
register_coco_instances("split2_50p", {},
                        split2_50p, train_img_path_coco)
register_coco_instances("split2_70p", {},
                        split2_70p, train_img_path_coco)

register_coco_instances("split3_30p", {},
                        split3_30p, train_img_path_coco)
register_coco_instances("split3_50p", {},
                        split3_50p, train_img_path_coco)
register_coco_instances("split3_70p", {},
                        split3_70p, train_img_path_coco)

register_coco_instances("coco_val", {},
                        val_ann_path_coco, val_img_path_coco)

split4_easy = 'splits/split4_easy.json'
split4_hard = 'splits/split4_hard.json'
split4_extreme = 'splits/split4_extreme.json'
split5_30p = 'splits/split5_30p.json'
split5_40p = 'splits/split5_40p.json'
split5_50p = 'splits/split5_50p.json'

# train_img_path_2007 = <YOUR VOC 2007 TRAIN IMAGE DIR>
# train_img_path = <YOUR VOC 2007 and 2012 IMAGE DIR> #for all voc paths are relative to this
# voc_val_json_path =  <YOUR VOC 2007 VAL JSON PATH>
# voc_val_img_path = <YOUR VOC 2007 VAL IMAGE DIR>

train_img_path_2007 = '/fs/vulcan-datasets/VOCdevkit/VOC2007/JPEGImages'
train_img_path = '/fs/vulcan-datasets'
voc_val_json_path =  '/fs/cfar-projects/grammar_analysis/iccv_final_code/SparseDet/voc_test_2007.json'
voc_val_img_path = '/fs/vulcan-datasets/VOCdevkit/VOC2007/JPEGImages'

register_coco_instances("split4_easy", {},
                        split4_easy, train_img_path)
register_coco_instances("split4_hard", {},
                        split4_hard, train_img_path)
register_coco_instances("split4_extreme", {},
                        split4_extreme, train_img_path)

register_coco_instances("split5_30p", {},
                        split5_30p, train_img_path_2007)
register_coco_instances("split5_40p", {},
                        split5_40p, train_img_path_2007)
register_coco_instances("split5_50p", {},
                        split5_50p, train_img_path_2007)
register_coco_instances("voc_test", {},
                        voc_val_json_path, voc_val_img_path)
                        


def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        assert (
            torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        assert (
            torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesSemSegEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def do_test(cfg, model):
    results = OrderedDict()
    mapper = FixMatchDatasetMapper_Inf(cfg, True) if cfg.FIXMATCH else None
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def do_train(cfg, model, teacher, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )
    mapper = FixMatchDatasetMapper(cfg, True) if cfg.FIXMATCH else None
    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement
    data_loader = build_detection_train_loader(cfg, mapper=mapper)
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            iteration = iteration + 1
            storage.step()
            for d in data:
                d['iter'] = iteration
            if cfg.THRESH_PATTERN:
                thresh = min(1, 1.2655*(1-np.exp(-iteration/210000.)))
            else:
                thresh = 0.8
            if teacher is not None:
                loss_dict = model(data, teacher=teacher, score_thresh=thresh)
            else:
                loss_dict = model(data)

            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict
            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()
            if (
                cfg.TEST.EVAL_PERIOD > 0
                and iteration % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter
            ):
                do_test(cfg, model)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()
            if iteration - start_iter > 5 and (iteration % 20 == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)


def adjust_inline_cfg_format(opts_list):
    for i, (k, v) in enumerate(zip(opts_list[0::2], opts_list[1::2])):
        if k == 'DATASETS.TRAIN' or k == 'DATASETS.TEST':
            opts_list[2*i+1] = (v, )
    return opts_list

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_student_teacher_config(cfg)
    cfg.merge_from_file(args.config_file)
    args.opts = adjust_inline_cfg_format(args.opts)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg


def main(args):
    cfg = setup(args)
    teacher= None

    cfg.defrost()
    if not cfg.FIXMATCH and 'FixMatch' in cfg.MODEL.META_ARCHITECTURE:
        logger.warning('Cannot use {} meta architecture without fixmatch training'.format(cfg.MODEL.META_ARCHITECTURE))
        cfg.MODEL.META_ARCHITECTURE = 'GeneralizedRCNN'
        logger.info('Changed meta architecture to default GeneralizedRCNN')
    if cfg.FIXMATCH and 'FixMatch' not in cfg.MODEL.META_ARCHITECTURE:
        logger.warning('Cannot use {} meta architecture with fixmatch training'.format(cfg.MODEL.META_ARCHITECTURE))
        cfg.MODEL.META_ARCHITECTURE = 'FixMatchGeneralizedRCNN' if 'faster_rcnn' in args.config_file else Exception('Bad architecture')
        logger.info('Changed meta architecture to {}'.format(cfg.MODEL.META_ARCHITECTURE))
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False if cfg.FIXMATCH else True
    cfg.freeze()
    model = build_model(cfg)

    print(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False ,find_unused_parameters=True
        )
        if teacher is not None:
            teacher = DistributedDataParallel(
                teacher, device_ids=[comm.get_local_rank()], broadcast_buffers=False, find_unused_parameters=True
            )
    do_train(cfg, model, teacher, resume=args.resume)
    return do_test(cfg, model)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
