---
layout: single
title: "CNN SLAM"
description: "Simultaneous Localisation and Mapping"
---

## Mission
To implement Simultaneous Localisation and Mapping (SLAM) algorithms.

## Problem Statement
 Implement a pipeline that can take successive frames of camera input and create a map of the surrounding.

## Simultaneous Localisation and Mapping (SLAM) 
SLAM is a rather useful addition for most robotic systems, wherein the vision module takes in a video stream and attempts to map the entire field of view. This allows for all sorts of "smart" bots, such as those constrained to perform a given task. A really good read onto what SLAM is can be found [here](https://nicolovaligi.com/deep-learning-robotics-slam.html).

SLAM is usually known as an egg-or-chicken problem - since we need a precise map to localise bots and a precise localisations to create a map - and is usually classified as a hard problem in the field of computer vision.

This project explores fusing key components of CNN imaging and geometric SLAM, where deep vision based monocular depth predictions are used in combination with geometry based SLAM predictions. The objective being to see if Deep vision can impact robotic SLAM, which has otherwise been largely disjoint from developments in the former field.

{% include figure image_path="/assets/images/projects/CNN-SLAM/overview.png" alt="Ml Session" caption="Pipeline Overview" %}

## Strategy 
We broke down the problem into three major subtasks of baseline stereo matching, tracking and bundle adjustment.  

__Small baseline stereo-matching__ refers to taking two images and matching them using the block matching algorithm in order to estimate depth. Once the two images are rectified, a simple block matching algorithm was used for getting the depth/disparity map for these pair of images.  

Under __bundle adjustment__, we looked at various existing pose graph optimisation techniques and implemented some ourselves. We used this for minimising the uncertainty between the
depths of two adjacent keyframes after transformation by updating the pose. We used the TUM-RBGD dataset to test certain parts of the project.  

Under __tracking__, we tried to implement pose optimization using a gauss newton optimizer that we had to make. The algorithms for this, though, did not give converging results. We however implemented Direct Sparse Odometry successfully as mentioned in the DSO paper by J. Engel et. al.  

{% include figure image_path="/assets/images/projects/CNN-SLAM/results.png" alt="Ml Session" caption="SLAM Results" %}
