---
layout: single
title: "Cell Segmentation"
description: "Our take at Government of India problem statement on cell identification."
header:
  overlay_image: /assets/images/projects/cell-segmentation/screenshot1.png
  overlay_filter: 0.5 # same as adding an opacity of 0.5 to a black background
gallery:
  - image_path: /assets/images/projects/cell-segmentation/screenshot1.png
    alt: "Output image 1"
    title: ""
  - image_path: /assets/images/projects/cell-segmentation/screenshot2.png
    alt: "Output image 2"
    title: ""
  - image_path: /assets/images/projects/cell-segmentation/screenshot3.png
    alt: "Output image 3"
    title: ""
  - image_path: /assets/images/projects/cell-segmentation/screenshot4.png
    alt: "Output image 4"
    title: ""
---

## Introduction

This project aims at performing automated identification of cell boundaries from the pathological video data.
We are given the video file `cells.avi` as input. The problem statement can be found [here.](https://innovate.mygov.in/challenges/identifying-cell-boundaries-from-video-data/)

The input video file:

{% include video id="kwVH6V-_Tyc" provider="youtube" %}

## Compatibility

* This code has been tested on Ubuntu 16.04 LTS and Windows 10
* **Dependencies** - Python 2.7 & 3.5, OpenCV 3.0+

## Methods Used

* Image Processing followed by Contours
* Adaptive Thresholding
* Watershed Algorithm
* [Structured Forest](https://pdollar.github.io/files/papers/DollarPAMI15edges.pdf) Algorithm

## Usage

First clone the repository by typing: `git clone https://github.com/iitmcvg/Cell-Segmentation.git`.

### Structured Forest

* First execute `python framesaver.py` to save the frames for structured forest.
* Next execute `python StructuredForests.py` to apply the edge detection.
* Finally, execute `python videowriter.py` to write the outputs to a video file.

## Results

* The video `edge.avi` is the result after applying Structured Forest algorithm. Other outputs can be found in the `Outputs` folder.
* Outputs of all methods can be seen at once in [this](https://drive.google.com/file/d/1mmDtpkT1wQzZ-aafKzgkFz4BpQd9eV88/view?usp=sharing) video.

The output video file:

{% include video id="Bq_xuVcDV30" provider="youtube" %}

We include a few output screenshots here:

{% include gallery caption="These are a few output screenshots" %}

## References

Our *Structured Forest* is an implementation of [Artanis CV Structured Forest](https://github.com/ArtanisCV/StructuredForests).

## Future work

* U-net convolutional neural network can be used.

* Implementing the algorithm given in this [paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5096676/).

## Disclaimer

This software is published for academic and non-commerical use only.
