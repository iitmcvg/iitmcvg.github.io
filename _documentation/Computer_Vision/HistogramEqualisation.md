---
layout: single
title: "Histogram Equalisation"
category: computer_vision
description: "Bringing in a bit of statistics ..."
---

#### By T Lokesh Kumar

## What is a Histogram??

In Statistics, Histogram is a graphical representation showing a visual impression of the distribution of data.

![exams]({{"/assets/images/documentation/computer_vision/HistogramEqualisation/exams.jpg"}})

We can note in the image above that vividly shows the distribution of marks
of a class. Along X-axis we have marks bins (each of 10 Marks width) and histograms describes how the marks of the class is spread among students. Moreover we can note that less students have got marks between 90-100 and 0-10. Many students have got marks between 50-60

## Histogram in the context of image processing

Similarly, image processing histogram normally refers to the histogram of pixel intensity values. For an 8-bit gray scale image, we have 256 possible intensity values (0 to 255) => we have 256 divisions along x axis, each representing gray scale intesities and histogram(as in the case of exams) show the distribution of pixels  amongst those gray scale values.

Example of a histogram of an image

![histogramexample]({{"/assets/images/documentation/computer_vision/HistogramEqualisation/histogramexample.jpg"}})

## Histogram Processing

Its usually advised to normalize a histogram by dividing each of its value by total number of pixels in that image, thus forming a normalised histogram.

This normalised histogram can be interpreted probability functions that denote the probability of occurrence of a gray scale intensity rk (just a variable) in the image. But it goes without mentioning that sum of of all components of a normalized histogram is 1.  

We will see how to exploit the properties of image histograms using OpenCV and python.

### Histogram calculation function in OpenCV


```python
import cv2
import matplotlib.pyplot as plt
img = cv2.imread('space.jpg')
cv2.imshow("Space", img)
cv2.waitKey()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hist = cv2.calcHist([img],[0], None, [256], [0,256])
# Note: Histograms can be calculated using numpy functions also. They will
# also the same output as the openCV functions. But the OpenCV functions are
# more faster (40X) than numpy functions
#Code in Numpy
# hist, bins = np.histogram(img.ravel(), 256, [0,256])
plt.plot(hist)
plt.xlabel('Pixel intensity values (0 - 255)')
plt.ylabel('No of pixels')
plt.title('Image Histogram for space.jpg')
plt.savefig('histo_space.jpg')
plt.show()
```

![space]({{"/assets/images/documentation/computer_vision/HistogramEqualisation/space.jpg"}})

![histo_space]({{"/assets/images/documentation/computer_vision/HistogramEqualisation/histo_space.jpg"}})

### Caluculation and plotting histograms for BGR images (colored images)


```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('space.jpg')
color = {'b','g','r'}
for i,col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0,256])
    plt.plot(histr, color = col)
    plt.xlim([0,256])
plt.show()
```

![color_hist]({{"/assets/images/documentation/computer_vision/HistogramEqualisation/color_hist.jpg"}})

### Application of mask

We generally use cv2.calcHist() to find the histogram fo the full image. There may be some cases where you may want to only find the histogram of certain regions of the image. Then we apply what is called masking.

In the regions where you want to find the histograms, you must create a white color patch, in other regions create a black color patch.Then pass this as a mask


```python
import cv2
img = cv2.imread('space.jpg', 0)

#Create a mask
mask = np.zeros(img[:2], np.uint8)
mask[100:300,100:400] = 255
masked_img = cv2.bitwise_and(img, img, mask = mask)

#Calculate histogram with mask and without mask
#Check third argument for mask
hist_full = cv2.calcHist([img], [0], None, [256], [0,256])
hist_mask = cv2.calcHist([img], [0], mask, [256], [0,256])
plt.subplot(221), plt.imshow(img, 'gray')
plt.subplot(222), plt.imshow(mask, 'gray')
plt.subplot(224), plt.imshow(masked_img, 'gray')
plt.subplot(224), plt.plto(hist_full), plt.plot(hist_mask)
plt.xlim([0,256])

plt.show()
```

Output is:
![mask_hist]({{"/assets/images/documentation/computer_vision/HistogramEqualisation/mask_hist.jpg"}})

## Histogram Equalization

### Why is histogram equalisation used??

Histogram equalization is a method in image processing of contrast adjustment using the image's histogram. It is not necessary that contrast will always be increase in this. There may be some cases were histogram equalization can be worse. In that cases the contrast is decreased.

Note: Generally, histogram equalisation is useful only when the histogram is confined to one region of the image. It does not work when there is large intensity variations, (i.e) where histogram covers a large area.

### How does Histogram equalisation enhance contrast??

This method usually increases the global contrast of many images, especially when the usable data of the image is represented by close contrast values. Through this adjustment, the intensities can be better distributed on the histogram. This allows for areas of lower local contrast to gain a higher contrast. Histogram equalization accomplishes this by effectively spreading out the most frequent intensity values.

For an example,
You can see the changes that occur after histogram equalisation, (notice the increase in contrast in the new image)
<img src = "old_hist.jpg">
The intensities are distributed evenly accross the histogram, which is shown here,
<img src = "hist_equal.png">

### Histogram equalisation in OpenCV


```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('wiki.jpg',0)
equ = cv2.equalizeHist(img)
res = np.hstack((img, equ))
cv2.imshow("Equalised Histogram", res)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

![equalised_image]({{"/assets/images/documentation/computer_vision/HistogramEqualisation/histo_equa.jpg"}})

In the above image we can note increase in contrast in the image. Generally, histogram modeling techniques (e.g. histogram equalization) provide a sophisticated method for modifying the dynamic range and contrast of an image by altering that image such that its intensity histogram has a desired shape (here a flat histogram).

 Histogram equalization employs a monotonic, non-linear mapping which re-assigns the intensity values of pixels in the input image such that the output image contains a uniform distribution of intensities (i.e. a flat histogram).

![Transfer Function]({{"/assets/images/documentation/computer_vision/HistogramEqualisation/heqtrans.gif"}})

### CLAHE (Contrast Limited Adaptive Histogram Equalization)

Applying histogram equalisation, considers global contrast of the image. It differs from ordinary histogram equalization in the respect that the adaptive method computes several histograms, each corresponding to a distinct section of the image, and uses them to redistribute the lightness values of the image. It is therefore suitable for improving the local contrast and enhancing the definitions of edges in each region of an image

### CLAHE in openCV


```python
import cv2
import numpy as np

img = cv2.imread('tsukuba_1.png', 0)

# Create a CLAHE object (Arguments are optional)
clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8,8))
cll = clahe.apply(img)

cv2.imwrite('clahe_2.jpg', cll)
```

See the result baloe and compare with the global histogram equalisation. Im attaching all the three images for reference.
![Before_CLAHE]({{"/assets/images/documentation/computer_vision/HistogramEqualisation/befor_hist.jpg"}})

After applying CLAHE

![After_CLAHE]({{"/assets/images/documentation/computer_vision/HistogramEqualisation/after_hist.jpg"}})

## Histogram Backprojection

### Theory

It is used for image segmentation or finding objects of interest in an image. In simple words, it creates an image of the same size (but single channel) as that of our input image, where each pixel corresponds to the probability of that pixel belonging to our object. In more simpler worlds, the output image will have our object of interest in more white compared to remaining part. Well, that is an intuitive explanation. (I can’t make it more simpler). Histogram Backprojection is used with camshift algorithm etc.

### Histogram backprojection in OpenCV



```python
import cv2
import numpy as np

roi = cv2.imread("rose_red.png")
hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

target = cv2.imread('rose.png')
hsvt =  cv2.cvtColor(roit, cv2.COLOR_BGR2HSV)

#Calculating the object histogram
roihist = cv2.calcHist([hsv], [0,1], None, [180,256], [0,180,0,256])

#normalize the histogram and apply backprojection
cv2.normalize(roihist, roihist, [0,1], cv2.NORM_MINMAX)
dst  cv2.calcBackProject([hsvt], [0,1], roihist, [0,180,0,256],1)

#Now convolve with circular disc
disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
cv2.filter2D(dst, -1, disc, dst)

#thresholding and binary and
ret, thresh = cv2.threshold(dst, 50,255,0)
thresh = cv2.merge((thresh, thresh, thresh))
res = cv2.bitwise_and(target, thresh)

res = np.hstack((target, thresh, res))
cv2.imwrite('res.jpg', res)

```

The output is as follows:

![Histogram BackProjection]({{"/assets/images/documentation/computer_vision/HistogramEqualisation/backproject.jpg"}})

## References

[Histogram Equalisation](http://homepages.inf.ed.ac.uk/rbf/HIPR2/histeq.htm)

[Image analysis, Intensity Histograms](homepages.inf.ed.ac.uk/rbf/HIPR2/histgram.htm)
