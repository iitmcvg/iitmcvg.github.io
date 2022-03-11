---
layout: single
title: "Low_Cost_Hawkeye"
description: "Developing a lowcost hawkeye system using 2 cameras such as those used in games like Cricket and Tennis"
---

**Goal:**

The Hawk Eye technology used by cricket bodies is very expensive and is thus not available to most coaching centers. To enable them to implement this technology to and provide better facilities to the developing talents, we are trying to build a cost effective implementation of this technology

**Objectives:**

We have divided the Hawkeye system into several sub-modules:

- Ball Detection
- Triangulation Pipeline
- Hardware Integration
- End-to-End fully functional Prototype

**Method:**

- Using Deep Learning, we built a ball detection system which
gives us the coordinates of the ball in the video.
- Next, using the principles of triangulation, we combine two views to get the 3D coordinates of the ball.
- A timeseries prediction model based on Recurrent neural networks will be used to predict the trajectory post impact.
