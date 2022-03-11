---
layout: single
title: "CVspice"
description: " Develop a real-time algorithm for the automatic recognition of hand-drawn electrical circuits based on object detection and circuit node recognition."
---



**Goal:**

The project CVspice aims to achieve a UI which inputs hand-drawn electric circuits which in turn identifies the components and the nodes which can then be used as an input to LTspice to solve the electrical circuit

**Objective:**

The UI is split into the following sub-modules:
- Component Recognition & Identification
- Connectivity of Components
- Identification of Nodes

Additionally, we are increasing the the dataset which will also include hand-drawn circuits, and we are exploring OCR for detecting the values of the components directly without the user giving an input.


**Method:**

- Firstly, Given an electric circuit image, outputs netlist describing components and their corresponding connectivity.
- We have trained around 350 annotated circuit images in
YOLOv5 which identifies the circuit components with
bounding boxes.
- Terminal points of the circuit components are found by the intersection of the binary image of A-bounding boxes and B-
adaptive thresholding of the original image. (A and B) = terminal points
- The nodes of the circuit are found using Breadth-First Search(BFS) algorithm