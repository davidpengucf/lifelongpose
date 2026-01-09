# [AAAI 2026] Lifelong Domain Adaptive 3D Human Pose Estimation

This implementation is based on [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) and [PoseAug](https://github.com/jfzhang95/PoseAug). Experiments on 4 datasets: Human3.6M, 3DHP, 3DPW, and Ski are provided. AdaptPose is aimed to improve accuracy of 3D pose estimators in cross-dataset scenarios and now supports **Lifelong (Continual) Learning** settings.

## Environment

```bash
conda create -n my_env python=3.6.9
conda activate my_env

```


```bash
pip install -r requirements.txt

```

## Lifelong Learning

We now support lifelong domain adaptation settings where the model adapts to a sequence of target domains (e.g., H36M 3DHP 3DPW) while mitigating catastrophic forgetting.

### 1. Lifelong Training

To train on a sequence of domains (configured in `lifelong_targets` inside the script):

```bash
python3 run_train.py --note lifelong_exp --posenet_name 'videopose' --lr_p 1e-4 --checkpoint './checkpoint/lifelong' --keypoints gt --pad 13

```

### 2. Lifelong Evaluation

To evaluate a checkpoint against all seen domains (Past + Present + Future):

```bash
python3 run_evaluate.py --posenet_name 'videopose' --keypoints gt --evaluate 'checkpoint/lifelong/final_stage.pth.tar' --pad 13

```

## Standard Experiments

### 1. Cross-dataset Evaluation of Pretrained Model on 3DHP dataset

Source: Human3.6M -> Target: 3DHP, 3DPW

```bash
python3 run_evaluate.py --posenet_name 'videopose' --keypoints gt --evaluate 'checkpoint/adaptpose/videopose/gt/3dhp/ckpt_best_dhp_p1.pth.tar' --dataset_target 3dhp --keypoints_target 'gt' --pad 13 --pretrain_path 'checkpoint/pretrain_baseline/videopose/gt/3dhp/ckpt_best.pth.tar'

```

### 2. Cross-dataset Training of Pretrained Model on 3DHP dataset

Source: Human3.6M / Target: 3DHP, 3DPW

```bash
python3 run_train.py --note poseaug --posenet_name 'videopose' --lr_p 1e-4 --checkpoint './checkpoint/adaptpose' --keypoints gt --keypoints_target gt --dataset_target '3dhp' --pretrain_path './checkpoint/pretrain_baseline/videopose/gt/3dhp/ckpt_best.pth.tar' --pad 13

```

