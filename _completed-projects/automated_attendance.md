---
layout: single
title: "Automated Attendance System"
description: "Using facial recognition to automate the attendance process."
---

{% include figure image_path="/assets/images/projects/AutomatedAttendance/AttendancePipeline.png" description="Pipeline" %}

## Mission  
To automate the cumbersome process of attendance at various institutions.  

## Problem Statement  
To create a facial recognition system to conduct attendance smoothly for a class of students.  

## Approach  
The faces are first detected from the field of view, and are affinely transformed. A neural network is then used to encode the important features of each face into a small vector. Once a database of face vectors are created, a new face is identified against the database using K Nearest Neighbours (KNN). If the face exists, then it’s attendance is marked against an institute server.  

## Working Solution
1. Pictures of classroom taken by PTZ camera.  
2. MTCNN model outputs bounding boxes around detected faces which are used to extract them.  
3. Alignment of detected faces are corrected using affine transform.  
4. Google’s FaceNet architecture converts frontal 160x160 images of faces to 128-dimensional vectors.  
5. K Nearest Neighbors algorithm is performed in the 128-dimensional vector-space to determine student ID.  
6. Attendance for the student is noted on the Attendance Server.  
