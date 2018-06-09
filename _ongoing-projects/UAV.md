---
layout: single
title: "UAVs for Disaster Relief Operation"
description: "Using Computer Vision and Deep Learning techniques for efficient disaster relief operations"
---
Disaster relief operations at present have the dual disadvantage of being cut off from resources and also being a risk to the rescue teams deployed. Our main aim is to create a product which will improve upon and assist the present disaster-relief scenario and also serve purposes such as SAR missions,surveillance tasks,etc.

{% include figure image_path="/assets/images/projects/UAV/uav2.jpeg" caption = "Depth Estimation" %}

We intend on deploying a team of interconnected autonomous drones which will use effective path-planning algorithms and obstacle avoidance using Computer Vision to distribute themselves over the affected zone. 

Our adaptive and intelligent path planning algorithms will be based on depth estimation and optical flow methods by estimating depth of a scene from its single RGB image $($using monocular vision$)$. The UAV will use the depth estimation algorithm combined with the control algorithm to achieve desired collision avoidance.

{% include figure image_path="/assets/images/projects/UAV/uav1.jpg" caption = "Aerial view of a disaster affected area" %}

The drones will utilize state-of-the-art deep learning models to extract essential information like number of survivors , severity of the situation , detection of fire etc. The drone will be able to do real time object detection using Convolutional Neural Networks $($CNNs$)$ and a tiny YOLO framework. It will be able to gather a lot of information like number of survivors and will also be able to do activity recognition using RNNs $($Recurrent Neural Networks$)$. The drones will then transmit this information along with GPS coordinates and live video feed to the base station $($or army headquarters$)$

{% include figure image_path="/assets/images/projects/UAV/im2.png" caption = "Object detection" %}


