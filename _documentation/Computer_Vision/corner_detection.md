---
layout: single
title: "Corner Detection"
category: computer_vision
description: "Use OpenCV's built in functions for corner detection"
---

Corners are good features to find and are useful in a lot of practical situations where opencv is needed. Corners can be considered as points where the intensity varies in many directions. Two ways opencv provides to detect corners are Harris detector and Shi-Tomasi detector.

The image is taken in grayscale form and the change in intensity for a displacement of $(x,y)$ from $(a,b)$ is represented in a mathematical form as

$$ \begin{equation*} E(x,y) = \sum_{a,b} w(a,b) [I(a+x,b+y)-I(a,b)]^2 \end{equation*} $$

$w(a,b)$ is a window function and can be rectangular or gaussian. It gives weights to pixels underneath the window.
$I(u,v)$ represents the intensity at $(u,v)$ point.
Maximizing the above expression gives possible corner points. Applying Taylor expansion the final equation is
$$ \begin{equation*} E(x,y) \approx \begin{vmatrix} u  \ v \end{vmatrix} M \begin{vmatrix} u \\ v \end{vmatrix} \end{equation*} $$ where the covariation matrix $$ \begin{equation*} M = \sum_{a,b} w(a,b) \begin{vmatrix} I_{x}I_{x} \ I_{x}I_{y} \\ I_{x}I_{y} \ I_{y}I_{y} \end{vmatrix} \end{equation*} $$
$I_{x}$ and $I_{y}$ are the derivatives in x and y directions respectively. These can be found with cv2.Sobel() but that is in-built in the corner detector function.

### Harris detector
A score was created to check if a particular window can contain a corner or not. $$ R = det(M)-k \ (trace(M))^2 $$
$ \lambda_1 $ and $ \lambda_2 $ are the eigen-values of $M$.
$$ det(M) = \lambda_1\lambda_2 $$
$$ trace(M) = \lambda_1 + \lambda_2 $$
So the values of these eigen values decide whether a region is corner, edge or flat.

* When $|R|$ is small, which happens when $\lambda_1$ and $\lambda_2$ are small, the region is flat.
* When $R<0$, which happens when $\lambda_1 >> \lambda_2$ or vice versa, the region is edge.
* When $R$ is large, which happens when $\lambda_1$ and $\lambda_2$ are large and $\lambda_1 \sim \lambda_2$, the region is a corner.

The image below illustrates this with $\lambda_1$ and $\lambda_2$ as axes.

<img src="ch1.jpg">

The result of Harris Corner Detection is a grayscale image with these scores as intensities. Thresholding for a suitable gives the corners in the image.



```python
import cv2
import numpy as np
img1=cv2.imread('chesss1.jpg',1)
img2=img1.copy()
img=cv2.imread('chesss1.jpg',0) #Image in grayscale
cv2.imwrite('Chess_Gray.jpg',img)
ch = cv2.cornerHarris(img,2,3,0.1)
ch=cv2.dilate(ch,None) #Dilating the corners to make them visible
img1[ch>0.01*ch.max()]=[0,0,0] #Thresholding
img2[ch>0.001*ch.max()]=[0,0,0]
cv2.imwrite('Corners.jpg',img1)
cv2.imwrite('Corner1.jpg',img2)
cv2.imshow('Image',img1)
cv2.imshow('Img_less_threshold',img2)
cv2.waitKey(0)

```




    True



__cv2.cornerHarris(src, blockSize, ksize, k[, dst[, borderType]]) → dst__


The cv2.cornerHarris() function takes as arguments the source image, the neighbourhood size to find the covariation matrix blockSize, the aperture parameter for the cv2.Sobel() operator ksize, and the k value in the score. It returns a grayscale image containing the intensities. The methods to extrapolate the return image may be given as an argument if needed (borderType).

Original Image <img src='Chess_Gray.jpg'>
Image with corners <img src='Corners.jpg'>
Image with more corners due to lower threshold <img src='Corner1.jpg'>

From the above images it is obvious that an optimum value of threshold is necessary to find only the desired corners.
At some places the corners are cramped together, points close to each other behave as corners. Here the centroids of all these points are taken and cv2.cornerSubPix() is used to give sub-pixel accuracy to the corners.
The function iterates to find the sub-pixel accurate location of corners or radial saddle points.
For every corner, in the neighbourhood of the corner the image gradient at a point is approximately perpendicular to the position vector of the point with respect to the corner. This observation is made use of to accurately locate the corners.
<img src="gd1.png">
Consider the dot product

$$\epsilon _i = {DI_{p_i}}^T \cdot (q - p_i)$$

where ${DI_{p_i}}$ is an image gradient at one of the points $p_i$ in a neighborhood of $q$. The value of $q$ is to be found so that $\epsilon_i$ is minimized. A system of equations may be set up with $\epsilon_i$ set to zero:

$$\sum _i(DI_{p_i} \cdot {DI_{p_i}}^T) - \sum _i(DI_{p_i} \cdot {DI_{p_i}}^T \cdot p_i)$$

where the gradients are summed within a neighborhood (“search window”) of $q$. Calling the first gradient term $G$ and the second gradient term $b$ gives:

$$q = G^{-1} \cdot b$$
The new center is set to the above point. The function iterates till a specific number of iterarions are done or the center moves from the original centroid by a maximum value, whichever happens first.


__cv2.cornerSubPix(image, corners, winSize, zeroZone, criteria) → None__

This function takes as argument the input image, the centroids of the corners in matrix form containing the coordinates, half the side length of the search window (if (3,3) is passed operations are performed over a window of size (7,7) to refine the corners), half of the size of the dead region in the middle of the search zone over which the summation in the formula below is not done (it is used sometimes to avoid possible singularities of the autocorrelation matrix; the value of (-1,-1) indicates that there is no such a size).

Criteria for termination of the iterative process of corner refinement is the last argument. That is, the process of corner position refinement stops either after criteria.maxCount iterations or when the corner position moves by less than criteria.epsilon on some iteration.


```python
import cv2
import numpy as np

img = cv2.imread('polygon1.png')
grayimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

grayimg = np.float32(grayimg)
ch = cv2.cornerHarris(grayimg,2,3,0.04) #Finding corners first
ch = cv2.dilate(ch,None)
ret, ch = cv2.threshold(ch,0.01*ch.max(),255,0)
ch = np.uint8(ch)

#Finding centroids of corners which are bunched together
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(ch) #Works only with opencv3.0
#Defining the criteria to end the iterations
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(grayimg,np.float32(centroids),(5,5),(-1,-1),criteria)

#Marking the points found
res = np.hstack((centroids,corners))
res = np.int0(res)
img[res[:,1],res[:,0]]=[0,0,255]
img[res[:,3],res[:,2]] = [0,255,0]

cv2.imwrite('subpixel.jpg',img)
```

An example is shown below from the documentation with the regions surrounding the corners zoomed out.
<img src="subpixel3.png">

Green circles show the corners after the function is executed and the red circles show the Harris corners. The green circles are obviously more accurate, but the overall change in intensities in the neighbourhood regions is greater only for the Harris corners.

### Shi Tomasi detector
There is only one major difference between the Harris detector and the Shi Tomasi detector. The score is the minimum of the two eigen-values.
$$\begin{equation*} R = min(\lambda_1,\lambda_2) \end{equation*}$$
The illustration with $\lambda_1$ and $\lambda_2$ as axes looks like
<img src="stc.png">
The cv2.goodFeaturesToTrack() function uses this algorithm to find a particular number of corners (unless Harris method is specifically asked for).
__cv2.goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance[, corners[, mask[, blockSize[, useHarrisDetector[, k]]]]]) → corners__
* image - source image
* maxCorners - the maximum number of corners which are needed
* qualityLevel - all the corners which fall below this threshold are rejected irrespective of whether the maximum number is obtained or not, this is relative to the best corner's quality measure; every corner returned will atleast be of quality this parameter value $\times$ the best quality
* minDistance - minimum possible distance between the corners; for two corners satisfying the quality check closer together than this value, the one with the lower quality is chucked out
* mask - optional argument giving region to look for corners
* blockSize - size of an average block for computing a derivative covariation matrix over each pixel neighborhood
* useHarrisDetector,k - Optional parameters; k is the free parameter argument of Harris detector

The function returns the corners in descending order of quality.

##### Quality measure :
For Harris detector the quality is directly the intensities returned by the function. For Shi-Tomasi detector another function is used to find the best quality measure; this returns an image with the minimun eigen values for all the points.

__cv2.cornerMinEigenVal(src, blockSize[, dst[, ksize[, borderType]]]) → dst__

* blockSize - neighbourhood size
* kSize - the aperture parameter of the cv2.Sobel() operator
* borderType - pixel extrapolation method in the image returned


```python
import numpy as np
import cv2

img = cv2.imread('cube2.jpg')
gimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gimg,25,0.01,10)
corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),5,(0,255,0),-1) #green circles are drawn on the detected corners
cv2.imwrite("Shi_Tomasi.jpg",img)
cv2.imwrite("Gray_Cube.jpg",gimg)

```




    True



<img src="Gray_Cube.jpg">
Original grayscale image  <img src="Image.jpg"> Image with corners

The most important aspect of corner detection is giving practical values to the various parameters involved. This can be mastered by implementing it a lot of times.
