---
layout: single
title: "Crack Detection"
description: "Detecting crack in architecture through a hybrid DL & CV approach"
---

**Goal:**

We attempt to solve the age old problem of detecting cracks in architecture by bringing together traditional CV techniques and state-of-the-art Deep Learning models by creating an efficient pipeline.

**Objective:**

Develop a pipeline that is able to detect cracks -
Current Pipeline is a culmination of detection techniques like YOLO followed by unsupervised feature extraction which is later integrated into the segmentation model.

**Method:**

- Original image is fed into YOLO and uses features from theentire image to predict a bounding box. This bounding box gives us the exact location of the crack which will be used tocrop out the crack for further processing and segmentation. This is passed through the pre-processing step. 
- Next image is processed by encoder for further feature extraction. Feature extraction is a process of dimensional reduction by which an initial set of raw data is reduced to more manageable groupsfor processing. 
- Our model reduces the images by focusing on various aspects such as specific edges, color gradient, etc. These features are further used to identify and organize different groups into required labels within the image. 
- The generated attention maps and crack detected by YOLO are parallelly passed to our Deep Learning Framework for crack segmentation. The output from this model is passed on to U-Net for denoising the image