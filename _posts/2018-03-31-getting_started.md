---
title: "Getting Started"
categories:
  - tutorials
description: "A 2018 take on an introduction to deep vision, reinforcement learning and NLP; an age when cross domain boundaries are fast dissolving."
header:
  teaser: https://www.nvidia.com/content/dam/en-zz/Solutions/research/research-home-areas-computer-vision-407-ud@2X.jpg
tags:
  - tutorials
  - primer
  - references

toc: true
toc_label: "Contents"
toc_icon: "gear"
---

We are in 2018, a year where computer vision seems more dominated by Deep Learning and Reinforcement learning than photogrammetry, and when  language is the next big barrier to be crossed. Where does one then start on understanding each domain and their cross-linking? With a bit of fooling around in these topics, we attempt to piece out a slightly coherent path.

How one goes about this is still quite conflicting- for instance a lot skip classical computer vision (processing and filter based) and get away with neural nets for pattern recognition. We are open to suggestions in the path advised, so feel free to drop in your comments below.

{% include figure image_path="https://i.ytimg.com/vi/aBVXfqumTXc/maxresdefault.jpg" alt="LSD Slam" caption="Credits: LSD SLAM Foodcourt dataset" %}

For those who would like a really extensive list of references, we have a compiled list in our content repository, similar to an _Awesome Deep Vision_ initiative. This can be found [here](https://github.com/iitmcvg/Content), under the _References_ folder.

Here are some references to get started with Computer Vision, Deep Learning and AI in general:

## Computer Vision:

### Courses
* The udacity course [ud810]() is a MOOC version of the Georgia Tech course : CS4495. You can find the 2015 run [here](https://www.cc.gatech.edu/~afb/classes/CS4495-Spring2015-OMS/) and the 2017 run [here](https://www.cc.gatech.edu/~hays/compvision/). The 2015 run was taken primarily by Aaron Bobbick, while the 2017 counterpart by [James Hays](https://www.cc.gatech.edu/~hays/).

Both run closely follow the Szeliski book, so you might find it easier to refer to this.

* MIT's computer vision course: this is available on [youtube](https://www.youtube.com/watch?v=CLOAswsxudo).

* UCF CRCV's 2012 course taken by Mubarak Shah is also a solid MOOC. You can find it [here](https://youtu.be/715uLCHt4jE).

### References

* [Computer Vision:  Models, Learning, and Inference](http://www.computervisionmodels.com/) - Simon J. D. Prince 2012. Quite a standard including Rick's book.
* [Computer Vision: Theory and Application](http://szeliski.org/Book/) - Rick Szeliski 2010. Could be your primary reference if you happen to follow the udacity course.
* [Computer Vision: A Modern Approach (2nd edition)](http://www.amazon.com/Computer-Vision-Modern-Approach-2nd/dp/013608592X/ref=dp_ob_title_bk) - David Forsyth and Jean Ponce 2011. Slightly more rigourous in terms of the signal processing.
* [Multiple View Geometry in Computer Vision](http://www.robots.ox.ac.uk/~vgg/hzbook/) - Richard Hartley and Andrew Zisserman 2004. This is nearly the gold standard of reference for geometry based computer vision (epipolar, stereo and perspective).

### Libraries

1. OpenCV remains the single biggest library for open-source implementations with both python and Cpp.
2. Scikit (and scipy) are very useful when considering signal processing techniques as well as non-standard implementations (general Hough transforms,..etc).
3. PCL: Point Cloud Library, primarily written in Cpp, with a few python bindings around. The library is widely used, however not many updates have been issued since 2014. You can find strawlab's python bindings for PCL [here](https://github.com/strawlab/python-pcl).


A more exhaustive list (of resources) for computer vision is hosted on our repository, [here](https://github.com/iitmcvg/Content/blob/master/References/awesome_CV.md).

### Sample Code

We have included jupyter notebooks for most computer vision techniques under our [Content](https://github.com/iitmcvg/Content) repository. This is a work in progress, so do feel free to hit up a pull request for certain notebooks.

## Machine Learning

### Courses

* [**CS229**](http://cs229.stanford.edu/syllabus.html) Stanford course dealing with ML. It has notes and problem statements, recorded video lectures are available on [YouTube](https://www.youtube.com/watch?v=UzxYlbK2c7E).

* [**Machine Learning - Coursera**](https://www.coursera.org/learn/machine-learning) This course is taught by Andrew NG, which gives an introduction to Machine Learning, Data Mining, Statistical Pattern Recognition. It has programming excercises in *Octave*.

* [**Introduction to Machine Learning**](https://in.udacity.com/course/intro-to-machine-learning--ud120) is an online course (Udacity) where feature engineering, evaluation of Machine Learning models are taught. Its approximately a 10 week, Intermediate Level course.

* [**Machine Learning**](https://in.udacity.com/course/machine-learning--ud262) is also an online course (Udacity) by Georgia Tech which is a graduate level course covering area of Artificial Intelligence. Introduction to Supervised, Unsupervised and Reinforcement Learning. Requires Prerequisites.

### References
### Libraries
### Sample Code

## Deep Learning

### Courses

* [**Neural Networks for Machine Learning**](https://www.coursera.org/learn/neural-networks?siteID=SAyYsTvLiGQ-Es75F2LQK_4jLsTIRNfi4A&utm_content=10&utm_medium=partners&utm_source=linkshare&utm_campaign=SAyYsTvLiGQ) A coursera online course by Geoffrey Hinton ( University of Toronto). In this course you will be taught about ANNs and how they are applied to speech recognition, image classification, image segmentation, NLP etc.

* [**Introduction to Deep Learing**](http://introtodeeplearning.com/index.html) This is a MIT course where deep learning is introduced and applied to machine translation, image recognition, game playing, image generation and more.

* [**Deep Learning**](https://in.udacity.com/course/deep-learning--ud730?utm_medium=referral&utm_campaign=api) This is a udacity course by Google where basic deep learing is introduced in a friendly manner.Learn to solve problems which were considered difficult, using deep learning.

* [**CS244n: Natural Language Processing with Deep Learning**](http://web.stanford.edu/class/cs224n/) In this course, deep learning is specifically applied to Natural language processing, text processing and speech recognition. You will learn how to use Recurrent Neural nets with attention to solve Machine translation, seq2seq mapping and other NLP related applications.

* [**CS231n: Convolutional Neural Networks for Visual Recongition**](http://cs231n.stanford.edu/) Here deep learning is specifically applied to solve computer vision applications.  This course is a deep dive into details of the deep learning architectures with a focus on learning end-to-end models for these tasks, particularly image classification. During the 10-week course, students will learn to implement, train and debug their own neural networks and gain a detailed understanding of cutting-edge research in computer vision.

### References
### Libraries
### Sample Code

## Reinforcement Learning

### Courses

* [David Silver’s Lectures](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html): A 10 hour intro to classical RL with a little bit of Deep RL mixed in from DeepMind’s David Silver. He’s one of the guys who worked on the famous Deep Q-Network that achieved super-human performance on most Atari games as well as Alpha Go which beat the world’s best Go players. Sutton & Barto’s book makes a great companion for these lectures since a lot of the material is borrowed from here.

* [B Ravindran’s NPTEL Lectures](http://nptel.ac.in/syllabus/106106143/): An extremely rigorous 12-week course from IITM’s very own Ravi Sir. This set of lectures is one of the most complete RL resources you can find online.
(Bonus fact: Ravi Sir’s PhD guide was Andrew Barto. The more you know!)

* [Berkley Deep RL Bootcamp](https://sites.google.com/view/deep-rl-bootcamp/lectures?authuser=0): This 10 lecture series features Vlad Mnih(DeepMind), John Schulman(OpenAI) and several other top researchers in the field of Deep RL. It should only be watched you’ve  understood all the concepts of classical RL very well since this discusses recent advances in Deep RL research.

### References

* [Introduction to Reinforcement Learning](https://drive.google.com/file/d/1xeUDVGWGUUv1-ccUMAZHJLej2C7aAFWY/view)- Sutton & Barto: Pretty much the go-to guide for RL these days. It introduces all the classical algorithms that you need to gain a solid understanding of the field.

* [Andrej Karpathy’s intro to Deep RL](http://karpathy.github.io/2016/05/31/rl/): This isn’t a textbook but should give you a flavour of what Deep RL consists of and how it’s different from classical RL.

### Libraries

* Environments: In order to train an RL agent, it needs to interact with an environment or a game so that it can receive information about its performance. [OpenAI Gym](https://gym.openai.com/) and [DeepMind Lab](https://deepmind.com/blog/open-sourcing-deepmind-lab/) are 2 great packages for python that are easy to use.

* Classic Algorithmns: Pybrain and RLLab offer implementations of several RL algorithms like SARSA, Q-learning and so on.

* Deep RL: Any standard Deep Learning library like [Tensorflow](https://www.tensorflow.org/) or [Pytorch](http://pytorch.org/) is sufficient to implement your Deep RL agent.

## Natural Language Processing (and Generation)
### Courses
### References
### Libraries
### Sample Code

Do share this post if you find it helpful.
