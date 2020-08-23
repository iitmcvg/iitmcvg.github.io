---
layout: single
title: "Traffic Analysis using CV"
description: "Developing a driver-assistance system that could assist the driver in real-time about Pedestrians,Traffic signs etc."
---

**Goal:**

We are motivated to develop a supportive eye that can assist the driver to mitigate an accident before it occurs to improve road-safety with the help of computer vision for Indian roads.

**Objectives:**

We are currently working on Pedestrian Protection System. We intend to work on several sub-divisions of Driver Assistance Systems like:

- Traffic-sign recognition
- Driver drowsiness detection
- Lane-departure Warning
- Lane-Change assistance

etc., in future.

**Method:**

- End-to-end object detection models are capable of detecting objects like pedestrians from road map in real time. Currently we are using YOLOv5 Network for this purpose. Training YOLOv5 using custom datasets of road-users or existing pedestrian datasets like Caltech, KITTI,IDD(Indian Driving Dataset) can improve accuracy.

{% include figure image_path="/assets/images/projects/Traffic_Analysis_2020/TA_1.jpg" caption="**Image Generated using custom_trained yolov5**" text-align=center %}
- Either sensor-based approaches (like LIDAR or RADAR) or vision-based approaches (like stereo-cameras or monocular video) are to be used to get the depth of the objects. If monocular depth estimation works well the cost of such a system could be reduced a lot.
- A network trained on human-pose can alert about direction of motion of pedestrians. For this purpose, labelled data of human images depending on their direction is to be used.
- Similarly, transfer learning of CNN&#39;s can achieve the purpose of traffic sign recognition and driver drowsiness detection. Traffic sign recognition involves two phases one is detection and localization and the other is text description of the localized image and with driver-eye face monitoring and vehicle-lane position monitoring the state of the driver can be assessed.
- A lane detection system used behind the lane departure warning system can be developed using the principle of Hough Transform and Canny edge detector to detect lane lines from real-time camera images from the front-end of automobile.
- With the help of cameras and sensors driver could be alerted about the vehicles that are hovering from the blind-spot of the driver.
- With the help of the above-mentioned approaches a system that could assist the driver in real-time to prevent collisions and mitigate accidents could be developed.

