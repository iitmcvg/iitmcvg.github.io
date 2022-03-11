---
layout: single
title: "Crime Detection"
description: "Detection of various crime for give CCTV footage"
---

**Goals:**

- Create a crime detection software that is able to detect various crime thus mitigating the manual effort required to sift through large video content


**Objective:**

The detection is split into 4 modules:
- Firearm Detection
- Action Detection
- Anomaly Detection
- Abandoned Bag Detection
- Facial Detection

**Method:**

- **Face-Detection:** 
  - Using scale invariant face detection algorithms to detect faces in a particular frame.
  - Recognising if the identified face is one of a wanted person using a light weight hybrid face recognition framework.
  - Application - Once such an accused is identified, picture and location of his/her
is sent to Police department