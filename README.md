# Multimodal Fusion Lane Detection Network based on Images and Steering Angles
## Introduction
This project solves the problem of lane line detection based on the method of vehicle corner fusion, which can effectively improve the effect of lane line detection.
## OverView
![overview](imgs/overview.jpg)
## Result
![result](imgs/result.png)
## Env
- mmcls>=1.0.0rc0
- mmcv>=2.0.0rc1,<2.1.0
- mmengine>=0.1.0,<1.0.0
- pytorch >= 1.8
- mmsegmentation >= 1.0
## Train
- Specify the config file in lane_with_angle_train.py
- python tools/lane_with_angle_train.py

