---
layout: single
title: "Object Detection and Tracking"
description: "Computer vision approach to object tracking."
---

In this notebook we are going to learn about different algorithms available for object tracking and thier usage. Object detection and tracking has been a great problem in computer vision for ages. To tackle this problem many people have comeup with different ideas.

To perform video tracking an algorithm analyzes sequential video frames and outputs the movement of targets between the frames. There are a variety of algorithms, each having strengths and weaknesses. Considering the intended use is important when choosing which algorithm to use. There are two major components of a visual tracking system: target representation and localization, as well as filtering and data association.

In this notebook,we will look through these algorithms

* Mean-Shift
* Cam-shift
* HOG
* Using HAAR like features

## Meanshift

The mean-shift algorithm is an efficient approach to tracking objects whose appearance is defined by histograms. (not limited by colour ). To understand this concept intuitively, see the pictures below. The main objective of the algorithm is to maximise the density of the ROI (Region of Interest). So the algorithm iteratively, moves the ROI such that after each iteration distance between centroid and center of the ROI decreases. This is pictorially depicted below (the points can be a pixel distribution like histogram backprojection)..

![Iteration 1]({{"/assets/images/documentation/computer_vision/Object_Detection_and_Tracking/Selection_001.png"}})

The orange point indicates the centroid or centre of mass of all the points in the ROI. The blue point indicates the center of the ROI. Sothe vector joining these two points as shown in the image, represents the movement of the center of the ROI in the next iteration.

![Iteration 2]({{"/assets/images/documentation/computer_vision/Object_Detection_and_Tracking/Selection_002.png"}})

Note that when compared to the previous image, ROI has moved along that vector(calculated during the previous iteration). Now a new set of points come into ROI. This changes the position of the center of mass of the ROI. So this algorithm iteratively minimizes the distance between the center and centroid of the ROI.

![Iteration 3]({{"/assets/images/documentation/computer_vision/Object_Detection_and_Tracking/Selection_003.png"}})

This is the mean shift algorithm. So how is it useful in computer vision's object tracking???

Here's the Answer...

So we normally pass the histogram backprojected image and initial target location. When the object moves, obviously the movement is reflected in histogram backprojected image. As a result, meanshift algorithm moves our window to the new location with maximum density.



```python
# Code for Mean_Shift in OpenCV
import cv2
import numpy as np

cap = cv2.VideoCapture("vaigai.mp4") # Enter your video here

ret, frame = cap.read() # Extracting the first frame

#Hard Coding the values of the initial ROI

r,h,c,w = 150,50,550,125 # simply hardcoded the values
track_window = (c,r,w,h)

# Setting up the ROI for tracking the object in subsequent frames
roi_frame = frame[r:r+h, c:c+w]   #ROI extractes from the image
roi_frame_hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)   #Converting to HSV so that we can effectively
                                                             # generate the mask
mask = cv2.inRange(roi_frame_hsv, np.array((0., 60.,32.)), np.array((180.,255.,255.))) #Hardcoding the values
                            # For effective values use general statistical methods fo find the range
roi_hist = cv2.calcHist([roi_frame_hsv], [0], mask, [180], [0,180])
cv2.normalize(roi_frame_hsv, roi_frame_hsv, 0,255, cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
frame_count  = 0
while(cap.isOpened()):
    frame_count = frame_count + 1
    ret, frame = cap.read()
    if frame_count > 50:
        break
    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0,180], 1)

        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        #To draw on the image
        (x,y,w,h) = track_window
        image = cv2.rectangle(frame, (x,y), (x+w, y+h), 255, 2)
        cv2.imshow('img2', image)

        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k)+".jpg", image) #Writing image
    else:
        break
cv2.destroyAllWindows()
cap.release()
```


Here you can note that the bounding box size is not adaptively changing and not rotating if the ROI rotates. This is not good since in somecases, box can move out of our region of interest.

## CAMshift

CAMshift (Continuously Adaptive Meanshift) which can consider changes in scale and rotation of ROI.

What does it do???

Camshift skilfully exploits the algorithm of mean-shift by modifying the size of the window when the latter has arrived To convergence. The coupled Camshift is an adaptation to the sequences of color images, and is exploited in object tracking in real time.

How does it actually work??

* Initialise the window W.
* At each iteration, the mean shift is applied with a window (W) of given size.
* After Mean shift converges, let the final Window, Wf be obtained.
* Increase the size of Wf by ± 5 pixels: evaluate the ellipse from the second order moments of the probability distribution located under Wf.
* Calculate the initial rectangle of the mean shift for the next image, and enlarge it by ±20% to define the search region. (20% is just for example, its an arbitary choice. It is adapted to the case of the segmentation of a face in relatively frontal pose).
* Re-iterate the previous steps for the next image.

See this image and may be your understanding might improve.....

![CAMshift for Videos]({{"/assets/images/documentation/computer_vision/Object_Detection_and_Tracking/camshift.gif"}})

Now we will code this algorithm with help of OpenCV's readymade function.



```python
# This code has a slight problem in it. Some corrections and modifications in this code will
# for sure yield awesome results. This code will be corrected as soon as possible.
import numpy as np
import cv2

cap = cv2.VideoCapture('vaigai.mp4')

# Taking the first frame of the Video
ret, frame = cap.read()

#Setting up the location of the window
r,h,c,w = 150,300,550,300
track_window = (c,r,w,h)

#Set up ROI for tracking
roi = frame[r:r+h, c:c+w]
hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# Creating a mask
mask = cv2.inRange(hsv_roi, np.array((0., 0.,0.)), np.array((30.,255.,255.)))
#cv2.imshow("mask", mask)
# Calculating Histogram for ROI
mask = None
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0,180])
cv2.normalize(roi_hist, roi_hist, 0,255,cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
while(1):
    ret, frame = cap.read()
    if ret: #(Checking if the frame is read)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        # Obtaining track window using CAMshift
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        # Draw it on the image
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv2.polylines(frame, [pts], True, 255, 2)
        cv2.imshow("img2", img2)

        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k) + ".jpg", img2)
    else:
        break
cv2.destroyAllWindows()
cap.release()



```

## Histogram of Oriented Gradients (HOG) for Object Detection


Histogram of Oriented Gradients (HOG) is a feature descriptor widely employed on several domains to characterize objects through their shapes. Local object appearance and shape can often be described by the distribution of local intensity gradients or edge directions.

HOG is widely utilized as a feature described image region for object detection such as human face or human body detection.

The image below roughly describes the sequence of procedures we follow while using Histogram of Oriented Gradients as features

![HOG_FlowChart]({{"/assets/images/documentation/computer_vision/Object_Detection_and_Tracking/hog-object-detection-sequence1.jpg"}})

The input in the chart, is generally the image we want to classify. The first procedure is Gamma and Color normalisation. This normalisation is done to increase the efficiency of the classifier.


* The object search is based on the detection technique applied for the small images defined by sliding detector window that probes region by region of the original input image and its scaled versions.


![sliding_window]({{"/assets/images/documentation/computer_vision/Object_Detection_and_Tracking/sliding_window_example.gif"}})
This is an example of the sliding a window approach, where we slide a window from left-to-right and top-to-bottom. **Note** : Only a single scale is shown. In practice this window would be applied to multiple scales of the image.


* The first step in HOG detection is to divide the source image into blocks (for example 16×16 pixels). Each block is divided by small regions, called cells (for example 8×8 pixels). Usually blocks overlap each other, so that the same cell may be in several blocks. For each pixel within the cell the vertical and horizontal gradients are obtained.


* The simplest method used to compute image gradients is to use 1-D Sobel vertical and horizontal operators:

   **Gx(y,x) = Y(y,x+1) – Y(y,x-1)**

   **Gy(y,x) = Y(y+1,x) – Y(y-1,x) **

   Where here, Gx(y,x) represents the gradient of image along X axis (horizontal gradient) and Gy(y,x) represents the gradient along y (vertical gradient). Y(y,x) is actually the pixel value at coordinates x and y. (Notice its just directional derivative along x and y axis)


* Gradients of the image can be calculated by convolving the image with Sobel_x and Sobel_y kernels. Okay, now we have obtained directional derivatives, but how to compute gradient at that point? We can compute the gradient magnitude and phase we will use this formula.


![gradient_calculation](hog-object-detection-eq1.jpg)


* Now that we have the gradient vector for each pixel, we can now go ahead and calculate the Histogram of Oriented Gradient for each cell (that 8x8 window example => you will be having 64 gradient vectors). For the histogram, Q bins for the angle are chosen (for example Q=9 => 20 degrees per bin). Usually unsigned orientation is used, so angles below 0 are increased by 180. For each gradient vector, it’s contribution to the histogram is given by the magnitude of the vector (so stronger gradients have a bigger impact on the histogram). We split the contribution between the two closest bins. So, for example, if a gradient vector has an angle of 85 degrees, then we add 1/4th of its magnitude to the bin centered at 70 degrees, and 3/4ths of its magnitude to the bin centered at 90.


Okay, we found gradients of each pixel and why should we even put it in a histogram firstly? Why cant we use these values directly?

The gradient histogram is a form of **“quantization”** , where in this case we are reducing 64 vectors with 2 components each down to a string of just 9 values (the magnitudes of each bin). Compressing the feature descriptor may be important for the performance of the classifier, but I believe the main intent here is actually to generalize the contents of the 8x8 cell.

![histogram]({{"/assets/images/documentation/computer_vision/Object_Detection_and_Tracking/histogram.png"}})


* Since different images may have different contrast, contrast normalization can be very useful. Normalization is done on the histogram vector v within a block. One of the following norms could be used:


![normalisation]({{"/assets/images/documentation/computer_vision/Object_Detection_and_Tracking/hog-object-detection-eq2.jpg"}})


* A descriptor is assigned to each detector window. This descriptor consists of all the cell histograms for each block in the detector window. The detector window descriptor is used as information for object recognition.


* Training and testing happens using this descriptor. Many possible methods exist to classify objects using the descriptor such as **SVM classifier** (support vector machine), **Neural networks classifier (more accuracy in the classification compared to other classifiers (SVM))** , etc

Now that we have an idea how HOG works, we will go to code them...




```python
# Aim of this program is to use HOG to detect pedestrians (they are our objects now).
# We will be importing a pretrained SVM classifier for pedestrian detection.
import cv2
import numpy as np

# Initialise HOG descriptor
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
# we call the setSVMDetector  to set the Support Vector Machine to be pre-trained pedestrian detector,
#loaded via the cv2.HOGDescriptor_getDefaultPeopleDetector()  function.

image = cv2.imread("crop_000001.png") #(Taken from INRIA Person Dataset)
image = cv2.resize(image, (400,400), interpolation = cv2.INTER_LINEAR)
# Resizing to reduce detection time and improve Detection accuracy
(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),padding=(8, 8), scale=1.05)
# The detectMultiScale  method constructs an image pyramid with scale=1.05
# and a sliding window step size of (4, 4)  pixels in both the x and y direction, respectively.

# Drawing the rectangle in our image
for (x, y, w, h) in rects:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Note: Non-Maxima Suppression is not done. Non maxima Suppression will improve the results quite well. This will
# suppress the bounding boxes which overlap more than certain given threshold.

cv2.imshow("Image_with_people_detected", image)
cv2.waitKey()
cv2.destroyAllWindows()

```

And yes, finally we come to our last topic...

## Haar Cascades

Object Detection using Haar feature-based cascade classifiers is an effective object detection method.

This uses a machine learning based approach where a cascade function is trained from a lot of positive and negative images. It is then used to detect objects in other images.


Here we will work with face detection. Initially, the algorithm needs a lot of positive images (images of faces) and negative images (images without faces) to train the classifier. Then we need to extract features from it. For this, haar features shown in below image are used.

They are just like our convolutional kernel. Each feature is a single value obtained by subtracting sum of pixels under white rectangle from sum of pixels under black rectangle.

![haar_features]({{"/assets/images/documentation/computer_vision/Object_Detection_and_Tracking/haar1.jpg"}})


We apply each and every feature on all the training images. For each feature, it finds the best threshold which will classify the faces to positive and negative. But obviously, there will be errors or misclassifications. We select the features with minimum error rate, which means they are the features that best classifies the face and non-face images.

Final classifier is a weighted sum of these weak classifiers. It is called weak because it alone can’t classify the image, but together with others forms a strong classifier. The paper says even 200 features provide detection with 95% accuracy.

Since there is a huge number of features to process, meaning it might be time consuming, authors introduced a concept called **Cascade of Classifiers**.

* Instead of applying all the 6000 features on a window, group the features into different stages of classifiers and apply one-by-one.

* If a window fails the first stage, discard it. We don’t consider remaining features on it.If it passes, apply the second stage of features and continue the process. The window which passes all stages is a face region.


This method of analysing the image greatly reduces the computational burden of the algorithm. This also means that this algorithm can process much faster than before.

OpenCV comes with a trainer as well as detector. If you want to train your own classifier for any object like car, planes etc. you can use OpenCV to create one.

OpenCV already contains many pre-trained classifiers for face, eyes, smile etc.Those XML files are stored in "opencv/data/haarcascades/" folder. Now its coding time!!


```python
import cv2
import numpy as np
# put the path to your .xml files here in both face_cascade and eye_cascade.
face_cascade = cv2.CascadeClassifier('/home/tlokeshkumar/anaconda3/pkgs/opencv-2.4.11-nppy27_0/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/home/tlokeshkumar/anaconda3/pkgs/opencv-2.4.11-nppy27_0/share/OpenCV/haarcascades/haarcascade_eye.xml')

img = cv2.imread('crop001623.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)


faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    img = cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)
cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()
```

## Further Readings

We have explored famous Computer Vision methods for Object detection and tracking. There are other tracker algorithms like MIL algorithm, etc. We have intentionally left it for you explore. More recently, object recognition, localisation and segmentation are performed with a Machine Learning algorithm called ** Convolutional Neural Networks ** which are performing much better than the above mentioned CV algorithms we saw here.
