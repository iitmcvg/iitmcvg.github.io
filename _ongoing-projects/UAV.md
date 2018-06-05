---
layout: single
title: "UAVs for Disaster Relief Operation"
description: "Using Computer Vision and Deep Learning techniques for efficient disaster relief operations"
---
Disaster relief at present has the dual disadvantage of being cut off from resources and being a risk for the rescue teams deployed. Our main aim is to create a product which will improve upon and assist the present disaster-relief operations scenario while additionally serving purposes such as SAR missions,surveillance tasks,etc.

{% include figure image_path="/assets/images/projects/UAV/im1.png" %}

We intend on deploying a team of interconnected autonomous drones which will use effective path-planning algorithms and obstacle avoidance using Computer Vision to distribute themselves over the affected zone. 

Our adaptive and intelligent path planning algos will be based on depth estimation and optical flow methods via estimating depth of a scene from its single RGB image (depth from monocular Vison!). The UAV will be integrated with the depth estimation algorithm combined with control algorithm to achieve desired collision avoidance.

{% include figure image_path="/assets/images/projects/UAV/im3.png" %}

The drones will utilize state-of-the-art deep learning models to extract essential information like number of survivors , severity of the situation , detection of fire in any etc. The drone will apply real time object detection based on Convolutional Neural Networks (CNNs) and tiny YOLO framework for detecting a variety of objects in images like number of survivors and also perform activity recognition using RNN’s. The drones will then transmit the info along with the live video feed and the GPS location to the
base station(army headqaurters)

{% include figure image_path="/assets/images/projects/UAV/im2.png" %}


