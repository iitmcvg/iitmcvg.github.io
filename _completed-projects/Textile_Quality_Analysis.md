---
layout: single
title: "Textile Quality Analysis"
description: "Determination of fabric quality using fabric image under pick glass"
---

The project __Textile Quality Analysis__ aims at counting the _wefts_(horizontal lines), _warps_(vertical lines)_of a given fabric image uner _pick glass_(a glass used to magnify, maintain reference in a fabric).

The analysis is done in three major steps:

1. __Segmenting only fabric from image:__
Template matching is used for segmentation in which a samall part from the centre of the image is used and mtched with the total image to recognise the fabric.
{% include figure image_path="/assets/images/projects/TextileQualityAnalysis/example" caption="**Segmented Image**" text-align=center %}
1. __Feature enhancement:__ CLAHE and Histogram Equalisation are used for enhancement of the image.
1. __Feature recognition:__ Several techniques are being used for counting the wefts and warps. The robustness of the model is being tested continously to make it better.


