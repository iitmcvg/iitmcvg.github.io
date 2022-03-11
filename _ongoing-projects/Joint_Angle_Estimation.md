---
layout: single
title: "Joint Angle Estimation using Cameras"
description: "Creating an end-to-end pipeline to perform joint angle estimation using cameras during practice or game session for multiple different sports."
---

**Goal:**

To be able to provide real-time joint angle estimation of sports players just using cameras.

**Objective:**

Develop a pipeline for obtaining Joint angles using cameras and apply for various use cases such as physiotherapy, illegal delivery identification etc.

**Method:**

- Construct a 2D pose from the input cameras data which can be overlayed on the image.
- Reconstruct a 3D pose from the 2D pose using a pictorial representation of Human pose.
- Use the coordinates of the joints in the 3D pose to determine the angle between the different joints
- Integrate this during video inference and provide real-time angle information
