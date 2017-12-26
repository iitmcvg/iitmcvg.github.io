---
layout: single
title: "Morphological Transformations"
description: "Erode and sharpen outlines with morphological transforms."
---

This jupyter notebook walks you through the basic image morphological transformations. These are normally performed for binary images i.e, those having pixels that are white(255) or black(0).
The two most important methods are **erosion** and **dilation**.
They are used in the pre-processing stage to remove noise from images.


```python
import cv2
import numpy as np
```


```python
# we are reading the image in grayscale format as indicated by the 0.
img = cv2.imread('image.jpg',0)
cv2.imshow('image',img)
cv2.waitKey(0)
```

<img src='image.jpg'>


```python
# define the kernel required for transformations. This is a 5x5 numpy array with all elements being 1.
kernel = np.ones((5,5),np.uint8)
```

## Erosion

First let us perform erosion operation.
We slide the kernel through the image, and for each pixel in the image, we observe the region in the kernel. If all the pixels in this region are white, that pixel is considered as white, otherwise it is eroded (made to zero).
For erosion process it is best to keep the foreground of the image white.
Then, erosion will make the boundary pixels black and thus reduce foreground thickness.

![Erosion]({{"/assets/images/documentation/computer_vision/Morphological_Transformations/mor-pri-erosion.gif"}})


```python
# the eroded image is stored in the variable erosion
erosion = cv2.erode(img,kernel,iterations = 1)
cv2.imshow('erosion',erosion)
cv2.waitKey(0)
```

![Erosion]({{"/assets/images/documentation/computer_vision/Morphological_Transformations/i1.jpg"}})

## Dilation

Now let us perform dilation, which is the opposite of erosion.
Here, a pixel element is made white if atleast one pixel under the kernel is white. So it increases the white region in the image(size of foreground object increases).

![Dilation]({{"/assets/images/documentation/computer_vision/Morphological_Transformations/mor-pri-dilation.gif"}})


```python
dilation = cv2.dilate(img,kernel,iterations = 1)
cv2.imshow('dilation',dilation)
cv2.waitKey(0)
```

![Dilation]({{"/assets/images/documentation/computer_vision/Morphological_Transformations/i2.jpg"}})

## Opening

Now let's look at another processs called opening. This is basically erosion followed by dilation.
This is useful in removing noise in the background.


```python
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
cv2.imshow('opening',opening)
cv2.waitKey(0)
```

![Opening]({{"/assets/images/documentation/computer_vision/Morphological_Transformations/i3.jpg"}})

## Closing

The opposite process of opening is closing (dilation followed by erosion).
It is useful in removing noise in the foreground.


```python
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
cv2.imshow('closing',closing)
cv2.waitKey(0)
```

![Closing]({{"/assets/images/documentation/computer_vision/Morphological_Transformations/i4.jpg"}})

## Morphological Gradient

Another feature is the morphological gradient which is the difference between dilation and erosion of the image.
It looks like the outline of the image.


```python
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
cv2.imshow('gradient',gradient)
cv2.waitKey(0)
```

![Gradient]({{"/assets/images/documentation/computer_vision/Morphological_Transformations/i5.jpg"}})

## Top hat and Black hat

There are two other transformations called top hat and black hat.
Top hat is the difference between the input image and its opening.
Black hat is the difference between closing of the input image and the input image.


```python
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
cv2.imshow('Tophat',tophat)
cv2.imshow('Blackhat',blackhat)
cv2.waitKey(0)
```

![Top Hat]({{"/assets/images/documentation/computer_vision/Morphological_Transformations/i6.jpg"}})
![Black Hat]({{"/assets/images/documentation/computer_vision/Morphological_Transformations/i7.jpg"}})

# References:

1. [OpenCV Docs](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html#morphological-ops)

2. [Python Programming](https://pythonprogramming.net/morphological-transformation-python-opencv-tutorial/)
