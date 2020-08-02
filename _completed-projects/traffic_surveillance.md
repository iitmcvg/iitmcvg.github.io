---
layout: single
title: "Traffic Surveillance using Computer Vision"
description: "Automation of traffic surveillance using computer vision."
---

## Mission
 To improve traffic violation detection and management by automation with computer vision techniques.  

## Problem Statement
Present systems of traffic surveillance in India have not caught up with the technological developments and heavily rely on manual surveillance by traffic police, which is not only inefficient but also a cause for corruption.  
Develop a comprehensive autonomous traffic surveillance system that would eliminate the need for manual intervention and improve efficiency.  

{% include figure image_path="/assets/images/projects/TrafficManagement/traffic.jpeg" caption="**Traffic Management**" text-align=center %}

## Strategy
In our approach, we intend to use video footage from traffic cameras and adopt computer vision techniques. 

The features we implemented are:

* __Automatic License Plate Recognition(ALPR)__: Automatically link a vehicle to its owner through the vehicle registry database. If the system detects traffic violations by the vehicle, challans can be directly issued to the owner.

* __Speed Detection:__ to calculate the speed of passing vehicles and penalise those crossing the speed limit.

* __Helmet Detection:__ to detect those not following safety rules.

## Working Solution
License Plate Recognition was done using binarization, opening, closing of images and then finding out the contours. Deep learning was not implemented due to less data on motorcycle images.

For character recognition, the contours are taken and then characters in license plate are recognised using k-nearest neighbour algorithm. The contour with most characters is considered to be the license plate. PyTesseract is used for getting the character text.

For helmet detection, the pipeline used was object detection with YOLOv3 followed by OpenPose pose estimator and a ResNet classifier for helmets.

## Future Scope
__Traffic Adaptive Signal Timing:__ To adjust signals based on road traffic and help smooth the flow of city traffic

__Seatbelt detection:__ Detecting seatbelts within cars to track the seatbelt rule violations.