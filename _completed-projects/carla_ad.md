---
layout: single
title: "Autonomous Driving System"
description: "To create an Autonomous Driving System (ADS) that would run on the open-source simulator CARLA"
---

## Mission
To create a self driving car for the open-source simulator CARLA

## Problem Statement
Create an Advanced Driver Assistance System (ADAS) for the open-source simulator CARLA that closely models real world conditions including maps, weather and traffic. The agent is required to reach a target destination following a predefined route, without any traffic infractions.

## The CARLA Challenge
The objective of the __CARLA__ project is to create an Autonomous Driving System (ADS) that would run on the open-source simulator [CARLA](https://carlachallenge.org/), which closely models real-life driving conditions with diverse maps, weather conditions and traffic scenarios. 

ADS is a path-breaking development in the field of transportation, and several tech and auto giants have already ventured into this field. Now, with open-source platforms to test and improve algorithms, developers across the globe have the opportunity to contribute to autonomous driving research.

The approach we have adopted involves using sensor data such as depth maps and camera images to gauge the environment, CV techniques and deep learning CNN for object detection and segmentation, and reinforcement learning CNN for controlling the vehicle. The RL CNN is trained using Proximal Policy Optimization (PPO).

## Strategy 
Pure computer vision (CV) methods like Hough lines for lanes and Hough circles for signals did not work well. So the YOLOv3 CNN was used to classify objects like pedestrians and signals. Lane segmentation was done using LaneNet and DeepLab was used for getting the semantic segmented depth map.

Using this information, an obstacle map was created and waypoints were marked and fed to the decision/control system. A reinforcement learning solution based on Deep Q-learning was tested but had issues due to hardware limitations and it’s inability to capture the reward structure of the environment. Hence, a PID controller was used, which gave much better performance. 


{% include figure image_path="/assets/images/projects/CARLA/carla1.jpeg" caption="**Car Detection**" text-align=center %}

{% include figure image_path="/assets/images/projects/CARLA/carla2.jpeg" text-align=center %}