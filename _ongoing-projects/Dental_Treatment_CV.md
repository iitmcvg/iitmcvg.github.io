---
layout: single
title: "Dental Treatment using Computer Vision"
description: "Obtaining Cephalometric Landmarks from Lateral Cephalograms using Computer Vision and Deep Learning & thus performing Cephalometric Analysis to obtain Dental Treatment & Diagnosis "
---

**Goal:**

To identify Cephalometric Landmarks from an Lateral Cephalogram and perform Cephalometric Analysis.

**Objective:**

Develop a software that can input Lateral Cephalograms and provide instantaneous Cephalometric Landmark Identification and Dental Treatment Plan that can assist Dentists in their diagnosis.

**Method:**

- We obtained a custom dataset on Lateral Cephalograms to train our Neural Network.
- We used a superior form of CNN (Convolutional Neural networks) called Foveated Pyramid attention based landmark detection. This makes the
training much more feasible and reduces load and time required by iteratively reducing the resolution of the areas surrounding the landmark.
- Once we obtained the Cephalometric Landmarks we perform Down's & Steiner's Analysis to obtain insights into treatment and Diagnosis.
- Finally, develop a UI for easy identification and determination of Cephalometric Landmarks.