---
layout: single
title: "Textile Quality Analysis"
description: "Determination of fabric quality using fabric image under pick glass"
---

## Mission
To sharply reduce the time taken for quality assurance in the textile industry.  
## Problem Statement
Devise a robust CV based pipeline to automate the quality inspection process in the textile industry and instantly infer the thread density, warp, weft and other quality metrics.

## Strategy
The analysis was done in the following steps:

1. __Segmenting only fabric from image:__
Template matching is used for segmentation in which a small part from the centre of the image is used and matched with the total image to recognise the fabric.  
{% include figure image_path="/assets/images/projects/TextileQualityAnalysis/example" caption="**Segmented Image**" text-align=center %}
1. __Feature enhancement:__ CLAHE and Histogram Equalisation are used for enhancement of the image.
1. __Feature recognition:__ Features like horizontal and vertical threads have to be detected, after which warp and weft have to be measured. For this, filter correlation, Hough transforms and clustering algorithms are used.  
We finally settled on adaptive threshold followed by correlation with filter obtained from centre of image. Since warp and weft are in rows and columns, the image is rotated and thresholded. Threshold is determined using K-means. A template match is done subsequently.
1. __Mobile App:__ An app was developed using Android Studio, to capture an image with the camera and output the warp and weft characteristics of the cloth.



