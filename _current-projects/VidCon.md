---
layout: single
title: "VidCon"
description: "Integrated Solutions for Video Conferencing Problems"
---

**Goal:**

In the recent times, due to the pandemic video conferencing tools have become an integral part of our lives. The current solutions in the market don’t solve most of the problems faced by the users and require high bandwidth to work.

We reimagine video conferencing tools, with innovative features using the latest advancements in Deep Learning. VidCon’s vision is to go beyond the usual communication space, giving the users a rich experience as close to the real world as possible.

**Objective:**

Enhance Video Conferencing by implementing:
- Super-Resolution for Video Quality
- Audio Denoising
- Gaze Correction

**Method:**

 - **Super-Resolution:** 
  - We take help of Generative adversarial networks incorporating a Generative facial prior (GFP). 
  - Here the Blind face restoration framework tries to take a facial image suffering for unknown degradation and aims to estimate a high quality image as similar as possible to the ground truth image.
  - We also try to minimize the identity preserving loss in order to get an image as close to the ground truth as possible

  - **Audio-Denoising:** Start from conventional DSP (Signal Processing) approach. Replace complicated estimators with an RNN (Recurrent Neural Network) instead of the conventional noise suppression algorithms
 

  - **Gaze Correction:** Eyeball rotation angles are dynamically estimated based on the positions of the camera and local and remote participants’ eyes. Then, a warping-based convolutional neural network is used to relocate pixels for redirecting eye gaze.

