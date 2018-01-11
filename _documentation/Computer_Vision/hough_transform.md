---
layout: single
title: "Hough Transform"
category: computer_vision
description: "The start of transform techniques and interpolations in computer vision."
---
## Hough Line Transform

## Theory
#### Hough Transform is a popular technique to detect any shape, if you can represent that shape in mathematical form. It can detect the shape even if it is broken or distorted a little bit. We will see how it works for a line.


A line can be represented as 𝑦 = 𝑚𝑥 + 𝑐 or in parametric form, as 𝜌 = 𝑥 cos 𝜃 + 𝑦 sin 𝜃 where 𝜌 is the perpendicular distance from origin to the line, and 𝜃 is the angle formed by this perpendicular line and horizontal axis measured in counter-clockwise ( That direction varies on how you represent the coordinate system. This representation is used in OpenCV). Check below image:

<img src = "/assets/images/documentation/computer_vision/Hough_Transform/houghlines1.svg">


So if line is passing below the origin, it will have a positive rho and angle less than 180. If it is going above the origin, instead of taking angle greater than 180, angle is taken less than 180, and rho is taken negative. Any vertical line will have 0 degree and horizontal lines will have 90 degree.

Now let’s see how Hough Transform works for lines. Any line can be represented in these two terms, (𝜌, 𝜃). So first it creates a 2D array or accumulator (to hold values of two parameters) and it is set to 0 initially. Let rows denote the 𝜌 and columns denote the 𝜃. Size of array depends on the accuracy you need. Suppose you want the accuracy of angles to be 1 degree, you need 180 columns. For 𝜌, the maximum distance possible is the diagonal length of the image. So taking one pixel accuracy, number of rows can be diagonal length of the image.

Consider a 100x100 image with a horizontal line at the middle. Take the first point of the line. You know its (x,y) values. Now in the line equation, put the values 𝜃 = 0, 1, 2, ...., 180 and check the 𝜌 you get. For every (𝜌, 𝜃) pair, you increment value by one in our accumulator in its corresponding (𝜌, 𝜃) cells. So now in accumulator, the cell (50,90) = 1 along with some other cells.

Now take the second point on the line. Do the same as above. Increment the the values in the cells corresponding to (𝜌, 𝜃) you got. This time, the cell (50,90) = 2. What you actually do is voting the (𝜌, 𝜃) values. You continue this process for every point on the line. At each point, the cell (50,90) will be incremented or voted up, while other cells may or may not be voted up. This way, at the end, the cell (50,90) will have maximum votes. So if you search the accumulator for maximum votes, you get the value (50,90) which says, there is a line in this image at distance 50 from origin and at angle 90 degrees.

Check out this cool animation:
http://docs.opencv.org/3.0-beta/_images/houghlinesdemo.gif

<img src = "/assets/images/documentation/computer_vision/Hough_Transform/houghlinesdemo.gif">

This is how hough transform for lines works. It is simple, and may be you can implement it using Numpy on your own. Below is an image which shows the accumulator. Bright spots at some locations denotes they are the parameters of possible lines in the image.

<img src = "/assets/images/documentation/computer_vision/Hough_Transform/houghlines2.jpg">

## Now Let us Check out the code


```python
import cv2
import numpy as np
```

##### Read the image here:
Image is attached in the same folder


```python
img = cv2.imread('sudoku.jpg')
img2=img.copy()
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
```

##### To apply Hough Line Transform, the image must be binary. So, we either threshold or apply Canny Edge detector to the image:
#### We have used Canny edge detector here. Try using thresholding. For details about thresholding look into the respective tutorial.


```python
edges = cv2.Canny(gray,50,150,apertureSize = 3)
```

In OpenCV,
##### cv2.HoughLines( )
performs the hough line transform as illustrated above.

It simply returns an array of (𝜌, 𝜃) values. 𝜌 is measured in pixels and 𝜃 is measured in radians.

##### Its, parameters are:

First parameter, Input image should be a binary image, so we pass the thresholded or edge detected image.

Second and third parameters are 𝜌 and 𝜃 accuracies respectively.

##### Fourth argument is the threshold, which means minimum vote it should get for it to be considered as a line. Remember, number of votes depend upon number of points on the line. So it represents the minimum length of line that should be detected.

##### Pass the image. 𝜌 accuracy is a natural number. 𝜃 accuracy is in radians. Threshold is a natural number(set it high to avoid false lines). Pass the required parameters.


```python
lines=cv2.HoughLines(edges,1,np.pi/180,200)
```

NOTE: lines[0] is an array storing r and theta values for different lines detected in the image
We have drawn the detected lines extending to a distance of 1000 above and below the given coordinates( 𝜌 * cos(𝜃) , 𝜌 * sin(𝜃))

#### Pass correct values for x1,x2,y1,y2 to complete a line. And using cv2.line( ), draw the lines passing the image where line has to be drawn, x1,y1,  x2, y2, colour of line (list of 3 integers representing BGR) and thickness.

### EXECUTE THE MARKDOWN CELL BELOW THE CODE CELL.


```python
for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)




cv2.imwrite('houghlines.jpg',img)
```




    True



#### OUTPUT:
<img src = '/assets/images/documentation/computer_vision/Hough_Transform/houghlines.jpg'>

## Probabilistic Hough Line Transform

In the hough transform, you can see that even for a line with two arguments, it takes a lot of computation. Probabilistic Hough Transform is an optimization of Hough Transform we saw. It doesn’t take all the points into consideration, instead take only a random subset of points and that is sufficient for line detection. Just we have to decrease the threshold. See below image which compare Hough Transform and Probabilistic Hough Transform in hough space.

This image illustrates Probabilistic Hough Transform better:
<img src = houghlines4.png>

##### The function used is cv2.HoughLinesP(). It has two new arguments.

minLineLength - Minimum length of line. Line segments shorter than this are rejected.

maxLineGap - Maximum allowed gap between line segments to treat them as single line.

##### PLEASE NOTE THAT THE THRESHOLD FOR Probabilistic Hough Transform is LOWER. Pass the first 4 parameters as in cv2.HoughLines( ). Pass values to minLineLength and maxLineGap. Both are natural numbers.


```python
minLineLength =
maxLineGap =
linesP = cv2.HoughLinesP()
```

#### Best thing is that, it directly returns the two endpoints of lines. In previous case, you got only the parameters of lines, and you had to find all the points. Here, everything is direct and simple.
#### Fill in cv2.line() with the correct parameters to draw a line of a desired colour and thickness. Pass img2 to avoid overlap of lines from the previous transform.

### EXECUTE THE MARKDOWN CELL BELOW THIS CODE CELL.


```python
for x1,y1,x2,y2 in linesP[0]:
    cv2.line()

cv2.imwrite('houghlinesP.jpg',img2)
```

#### OUTPUT:
<img src = '/assets/images/documentation/computer_vision/Hough_Transform/houghlinesP.jpg'>

## HOUGH CIRCLE TRANSFORM

### Similar to a line, a circle can be represented using certain parameters. This drives us towards a transform for an image to detect circles in it.

#### A circle is represented mathematically as (𝑥 − 𝑥𝑐𝑒𝑛𝑡𝑒𝑟 )2 + (𝑦 − 𝑦𝑐𝑒𝑛𝑡𝑒𝑟 )2 = 𝑟2 where (𝑥𝑐𝑒𝑛𝑡𝑒𝑟 , 𝑦𝑐𝑒𝑛𝑡𝑒𝑟 ) is the center of the circle, and 𝑟 is the radius of the circle. From equation, we can see we have 3 parameters. So we need a 3D accumulator for hough transform, which would be highly ineffective. So OpenCV uses more trickier method, Hough Gradient Method which uses the gradient information of edges.

#### Now we move on to the code:

##### Read the image here (preferably an image with circles) :


```python
cimg = cv2.imread('/assets/images/documentation/computer_vision/Hough_Transform/opencv.png')
img_circ = cv2.cvtColor(cimg,cv2.COLOR_BGR2GRAY)
```

##### We blur the image for denoising.


```python
img_circ = cv2.medianBlur(img_circ,5)
```

#### Complete the function to detect circles in the image.
##### The parameters are as follows:
##### Parameters:
###### image – 8-bit, single-channel, grayscale input image. Hence, use img_circ
###### method – Detection method to use. Currently, the only implemented method is CV_HOUGH_GRADIENT
###### dp – Inverse ratio of the accumulator resolution to the image resolution. For example, if dp=1 , the accumulator has the same resolution as the input image. If dp=2 , the accumulator has half as big width and height.
###### minDist – Minimum distance between the centers of the detected circles. If the parameter is too small, multiple neighbor circles may be falsely detected in addition to a true one. If it is too large, some circles may be missed.
###### param1 – First method-specific parameter. In case of CV_HOUGH_GRADIENT , it is the higher threshold of the two passed to the Canny() edge detector used in this method (the lower one is twice smaller). PLEASE CHECK OUT EDGE DETECTION TUTORIAL FOR MORE DETAILS.
###### param2 – Second method-specific parameter. In case of CV_HOUGH_GRADIENT , it is the accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected. Circles, corresponding to the larger accumulator values, will be returned first.
###### minRadius – Minimum circle radius.
###### maxRadius – Maximum circle radius.

##### Returns:
###### circles – Output vector of found circles. Each vector is encoded as a 3-element floating-point vector (x, y, radius) .


```python
circles = cv2.HoughCircles()
```

Using np.around(), we round off array of (x,y,radius) to the nearest integer

#### Use cv2.circle( ) and draw the detected circles. You need to pass cimg, (x_centre,y_centre), radius, colour of circle(list of 3 integers representing the colour), thickness of circle
#### Again using cv2.circle( ), plot the centres in cimg. Again we need to pass cimg, (x_centre,y_centre), hardcoded radius of point(very small), colour of point, thickness of drawn circle around point(-1 for filling the complete point)
### EXECUTE THE MARKDOWN CELL BELOW THIS CODE CELL.


```python

circles = np.uint16(np.around(circles))

for i in circles[0,:]:
    # draw the outer circle
    cv2.circle()
    # draw the center of the circle
    cv2.circle()
    cv2.imwrite('detected_circles.jpg',cimg)
```

#### OUTPUT:
<img src='/assets/images/documentation/computer_vision/Hough_Transform/detected_circles.jpg'>
