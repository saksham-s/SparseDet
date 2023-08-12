# SparseDet: Improving Sparsely Annotated Object Detection with Pseudo-positive Mining (ICCV 2023)

![plot](./teaser.png)

This is the official repository for the work [SparseDet: Improving Sparsely Annotated Object Detection with Pseudo-positive Mining] accepted to ICCV 2023. It includes scripts to train SparseDet on different splits also provided in this repository. Also see our [Project Webpage](cs.umd.edu/~sakshams/SparseDet/).


## Setup
Tested with Python 3.6.15

### Create Environment
```
python3 -m venv env_sparsedet
```

### Activate Environment and Install Packages
```
source env_sparsedet/bin/activate
pip install --upgrade pip
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
pip install opencv-python==4.6.0.66
pip install setuptools==59.5.0
```



## Splits

Link to Splits - https://drive.google.com/drive/folders/168agXPO7LmpMWItl2bonbsdcEulYD2Cq?usp=sharing. Download them and place them in the splits directory.

## Sample Commands for Training on 4 Gpus

```
#COCO
python plain_train_net.py \
--config-file configs/faster_rcnn_R_101_FPN_3x_mod.yaml \
--dist-url tcp://0.0.0.0:12345 \
--num-gpus 4 \
--resume \
OUTPUT_DIR experiments_coco_fpn/split1_30p \
DATASETS.TRAIN split1_30p \
DATASETS.TEST coco_val \
DATALOADER.NUM_WORKERS 8 \
SOLVER.IMS_PER_BATCH 8 \
FIXMATCH True \
FIXMATCH_STRONG_AUG True \
MASK_BOXES 30000 \
MASK_BOXES_THRESH 0.8 \
MASK_BOXES_RPN True \
DISTILLATION_LOSS_WEIGHT 1.0 \
DET_THRESH 0.0 \
CONSISTENCY_REGULARIZATION False \
MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE 256 \
MODEL.ROI_HEADS.POSITIVE_FRACTION 0.5 \
SOLVER.BASE_LR 0.01 \
SEED 1234
```

```
#VOC
python plain_train_net.py \
--config-file configs/faster_rcnn_R_101_FPN_3x_voc_18k.yaml \
--dist-url tcp://0.0.0.0:12345 \
--num-gpus 4 \
--resume \
OUTPUT_DIR experiments_voc_fpn/split5_50p \
DATASETS.TRAIN split5_50p \
DATASETS.TEST voc_test \
DATALOADER.NUM_WORKERS 8 \
SOLVER.IMS_PER_BATCH 8 \
FIXMATCH True \
FIXMATCH_STRONG_AUG True \
MASK_BOXES 9000 \
MASK_BOXES_THRESH 0.8 \
MASK_BOXES_RPN True \
DISTILLATION_LOSS_WEIGHT 1.0 \
DET_THRESH 0.0 \
CONSISTENCY_REGULARIZATION False \
MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE 256 \
MODEL.ROI_HEADS.POSITIVE_FRACTION 0.5 \
SOLVER.BASE_LR 0.01 \
SEED 1234
```


## License
Distributed under the MIT License.