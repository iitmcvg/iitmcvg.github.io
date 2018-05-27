---
layout: single
title: "CNN SLAM"
description: "Simultaneous Localisation and Mapping in 2018"
---

__Simultaneous Localisation and Mapping__(SLAM) is a rather useful addition for most robotic systems, wherein the vision module takes in a video stream and attempts to map the entire field of view. This allows for all sorts of "smart" bots, such as those constrained to perform a given task. A really good read onto what SLAM is can be found [here](https://nicolovaligi.com/deep-learning-robotics-slam.html).

SLAM is usually known as an egg-or-chicken problem - since we need a precise map to localise bots and a precise localisations to create a map - and is usually classified as a hard problem in the field of computer vision.

This project explores fusing key components of CNN imaging and geometric SLAM, where deep vision based monocular depth predictions are used in combination with geometry based SLAM predictions. The objective being to see if Deep vision can impact robotic SLAM, which has otherwise been largely disjoint from developments in the former field.

{% include figure image_path="/assets/images/projects/CNN-SLAM/overview.png" alt="Ml Session" caption="Pipeline Overview" %}

Our main inspiration comes from the demonstrated work at CVPR 2017 by Tateno, Tombari et all. The original paper may be found here: [link](https://arxiv.org/abs/1704.03489). Since then there have been quite a few developments in each of the pipeline changes. We are working on extensively experimenting with the same and reporting our findings.

{% include figure image_path="/assets/images/projects/CNN-SLAM/results.png" alt="Ml Session" caption="SLAM Results" %}
