<h1 align="center">
SAGA: Stochastic Whole-Body Grasping with Contact
</h1>

> [**SAGA: Stochastic Whole-Body Grasping with Contact**](https://jiahaoplus.github.io/SAGA/saga.html)  
> **ECCV 2022**  
> Yan Wu*, Jiahao Wang*, Yan Zhang, Siwei Zhang, Otmar Hilliges, Fisher Yu, Siyu Tang

![alt text](https://github.com/JiahaoPlus/SAGA/blob/main/images/teaser.png)
This repository is the official implementation for the ECCV 2022 paper: [SAGA: Stochastic Whole-Body Grasping with Contact](https://jiahaoplus.github.io/SAGA/saga.html).

\[[Project Page](https://jiahaoplus.github.io/SAGA/saga.html) | [Paper](https://arxiv.org/abs/2112.10103)\]

## Introduction
Given an object in 3D space and a human initial pose, we aim to generate diverse human motion sequences to approach and grasp the given object. We propose a two-stage pipeline to address this problem by generating grasping ending pose first and then infilling the in-between motion.

![alt text](https://github.com/JiahaoPlus/SAGA/blob/main/images/two-stage-pipeline.png)

<table class="center">
  <tr>
    <th>Input</th>
    <th>First-stage result</th>
    <th>Second-stage result</th>
  </tr>
  <tr>
    <th rowspan="2"><img src="https://github.com/JiahaoPlus/SAGA/blob/main/images/binoculars-0.jpg" width=200></th>
    <th><img src="https://github.com/JiahaoPlus/SAGA/blob/main/images/binoculars-60.jpg" width=400></th>
    <th><img src="https://github.com/JiahaoPlus/SAGA/blob/main/images/binoculars-video.gif" width=350></th>
  </tr>
  <tr>
    <th><img src="https://github.com/JiahaoPlus/SAGA/blob/main/images/binoculars-60-first-view.jpg" width=400></th>
    <th><img src="https://github.com/JiahaoPlus/SAGA/blob/main/images/binoculars-movie-first-view.gif" width=350></th>
  </tr>
  <tr>
    <th rowspan="2"><img src="https://github.com/JiahaoPlus/SAGA/blob/main/images/wineglass-0.jpg" width=200></th>
    <th><img src="https://github.com/JiahaoPlus/SAGA/blob/main/images/wineglass-60.jpg" width=400></th>
    <th><img src="https://github.com/JiahaoPlus/SAGA/blob/main/images/wineglass-video.gif" width=350></th>
  </tr>
  <tr>
    <th><img src="https://github.com/JiahaoPlus/SAGA/blob/main/images/wineglass-60-first-view.jpg" width=400></th>
    <th><img src="https://github.com/JiahaoPlus/SAGA/blob/main/images/wineglass-movie-first-view.gif" width=350></th>
  </tr>
</table>

## Contents
- [Installation](https://github.com/JiahaoPlus/SAGA#installation)
- [Dataset Preparation](https://github.com/JiahaoPlus/SAGA#Dataset)
- [Pretrained models](https://github.com/JiahaoPlus/SAGA#pretrained-models)
- [Train](https://github.com/JiahaoPlus/SAGA#train)
- [Grasping poses and motions generation for given object](https://github.com/JiahaoPlus/SAGA#inference) (object position and orientation can be customized)
- [Visualization](https://github.com/JiahaoPlus/SAGA#visualization)

## Installation
- <strong>Packages</strong>
    - python>=3.8  
    - pytorch==1.12.1  
    - [human-body-prior](https://pypi.org/project/human-body-prior/)  
    - [SMPLX](https://github.com/vchoutas/smplx)  
    - [Chamfer Distance](https://github.com/otaheri/chamfer_distance)  
    - Open3D

- <strong>Body Models</strong>  
Download [SMPL-X body model and vposer v1.0 model](https://smpl-x.is.tue.mpg.de/index.html) and put them under /body_utils/body_models folder as below:
```
SAGA
│
└───body_utils
    │
    └───body_models 
        │
        └───smplx
        │   └───SMPLX_FEMALE.npz
        │   └───...
        │   
        └───vposer_v1_0
        │   └───snapshots
        │       └───TR00_E096.pt
        │   └───...
        │
        └───VPoser
        │   └───vposerDecoderWeights.npz
        │   └───vposerEnccoderWeights.npz
        │   └───vposerMeanPose.npz
    │
    └───...
│
└───...
```

## Dataset
### 
Download [GRAB](https://grab.is.tue.mpg.de/) object mesh

Download dataset for the first stage (GraspPose) from [[Google Drive]](https://drive.google.com/uc?export=download&id=1OfSGa3Y1QwkbeXUmAhrfeXtF89qvZj54)

Download dataset for the second stage (GraspMotion) from [[Google Drive]](https://drive.google.com/uc?export=download&id=1QiouaqunhxKuv0D0QHv1JHlwVU-F6dWm)

Put them under /dataset as below,
```
SAGA
│
└───dataset 
    │
    └───GraspPose
    │   └───train
    │       └───s1
    │       └───...
    │   └───eval
    │       └───s1
    │       └───...
    │   └───test
    │       └───s1
    │       └───...
    │   
    └───GraspMotion
    │   └───Processed_Traj
    │   └───s1
    │   └───...
    │   
    └───contact_meshes
    │   └───airplane.ply
    │   └───...
│
└───... 
```
    
## Pretrained models
Download pretrained models from [[Google Drive]](https://drive.google.com/uc?export=download&id=1dxzBBOUbRuUAlNQGxnbmWLmtP7bmEE_9), and the pretrained models include:
- Stage 1: pretrained WholeGrasp-VAE for male and female respectively
- Stage 2: pretrained TrajFill-VAE and LocalMotionFill-VAE (end to end)

## Train
### First Stage: WholeGrasp-VAE training
```
python train_grasppose.py --data_path ./dataset/GraspPose --gender male --exp_name male
```

### Second Stage: MotionFill-VAE training
Can train TrajFill-VAE and LocalMotionFill-VAE separately first (download separately trained models from [[Google Drive]](https://drive.google.com/uc?export=download&id=1eyUW7YLmnAj-CHwIMe9qsAs6W63Aw7Ce)), and then train them end-to-end:
```
python train_graspmotion.py --pretrained_path_traj $PRETRAINED_MODEL_PATH/TrajFill_model_separate_trained.pkl --pretrained_path_motion $PRETRAINED_MODEL_PATH/LocalMotionFill_model_separate_trained.pkl
```

## Inference
### First Stage: WholeGrasp-VAE sampling + GraspPose-Opt
At the first stage, we generate grasping poses for the given object.  
The example command below generates 10 male pose samples to grasp camera, where the object's height and orientation are randomly set within a reasonable range. You can also easily customize your own setting accordingly.
```
python opt_grasppose.py --exp_name pretrained_male --gender male --pose_ckpt_path $PRETRAINED_MODEL_PATH/male_grasppose_model.pt --object camera --n_object_samples 10
```
### Second Stage: MotionFill-VAE sampling + GraspMotion-Opt
At the second stage, with generated ending pose from the first stage and a customizable human initial pose, we generate in-between motions.  
The example command below generates male motion samples to grasp camera, where the human initial pose and the initial distance away from the given object are randomly set within a reasonable range. You can also easily customize your own setting accordingly.
```
python opt_graspmotion.py --GraspPose_exp_name pretrained_male --object camera --gender male --traj_ckpt_path $PRETRAINED_MODEL_PATH/TrajFill_model.pkl --motion_ckpt_path $PRETRAINED_MODEL_PATH/LocalMotionFill_model.pkl
```

## Visualization
We provide visualization script to visualize the generated grasping ending pose results which is saved at (by default) _/results/$EXP_NAME/GraspPose/$OBJECT/fitting_results.npz_.
```
cd visualization
python vis_pose.py --exp_name pretrained_male --gender male --object camera
```

We provide visualization script to visualize the generated grasping motion result which is saved at (by default) _/results/$EXP_NAME/GraspMotion/$OBJECT/fitting_results.npy_, from 3 view points, the first-person view, third-person view and the bird-eye view.
```
cd visualization
python vis_motion.py --GraspPose_exp_name pretrained_male --gender male --object camera
```

### Contact
If you have any questions, feel free to contact us:
- Yan Wu: yan.wu@vision.ee.ethz.ch
- Jiahao Wang: jiwang@mpi-inf.mpg.de
### Citation
```bash
@inproceedings{wu2022saga,
  title = {SAGA: Stochastic Whole-Body Grasping with Contact},
  author = {Wu, Yan and Wang, Jiahao and Zhang, Yan and Zhang, Siwei and Hilliges, Otmar and Yu, Fisher and Tang, Siyu},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year = {2022}
}
```
