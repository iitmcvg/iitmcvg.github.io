---
layout: single
title: "Deep nets analysis, exploration and visualization"
description: "Using different methods and techniques to try to understand what exactly dictates the decision a neural network makes"
---

**Goal:**

Deep nets have gotten extremely effective at solving problems that we want them to, and as we use them to solve furthermore complex problems, we also use more and more complex architectures. With increase in complexity, even the architects of the net stop truly understanding what the net does. They just possess knowledge about how the net trains and what data it is trained on, but the network itself remains a black box. The aim of our project is to use different methods and techniques to try to understand what exactly dictates the decision a neural network makes.

**Objective:**

- The first way we plan on increasing our understanding of neural nets is to develop methods to visualize what a neural network sees.
- The second part of the project involves using more human like methods in image classification to implement a top down approach to image classification.

**Method:**

- For the first part we are trying to visualize is a RESNET trained to classify flowers. There are 3 techniques we are implementing, those being:
  - **Activation Maximization** : In this method we create an image in which all the pixel values are variables, pass the image forward through the trained network and train the image to maximally activates a single neuron in the final layer. The image thus produced is a representation of what that neuron represents in the network and shows the network&#39;s abstraction of what the class is. This is done by maximizing the inner product of a one hot vector X0 and the output of a layer in the net after the variable image is passed through it.

	{% include figure image_path="/assets/images/projects/Deep_nets_2020/deep_1.png" text-align=center %}

  - **Caricaturization:** This method is like activation maximization, but instead of being restricted to just the final layer, this method allows the maximization of any neuron in the network. This gives more flexibility and allows us to see what individual characteristics make up the class that we were trying to reproduce.

	{% include figure image_path="/assets/images/projects/Deep_nets_2020/deep_2.png" caption="An image created by the deep dream algorithm by maximizing features" text-align=center %} 

  - **Inversion:** In this method, the output class is represented as a function f(X0) of a standard image X0 where the function is the trained neural net. We then try to recreate X0 from the function by inverting it and passing the class as the input. Since most neural nets are not invertible, this gives a result different from the initial image. This new representation helps us understand what information was lost during the classification process, as some of it must be lost to generalize a class of images. For example, if all roses are to give the same output, some of the details about each rose image must be lost when the initial image is passed through the function.

 Of course, some of the images obtained using these methods are non-sensical. We restrict the space using regularizers to make the outputs only those that are &quot;natural&quot; and understandable to humans.

- In the second part the human vision system works in a hierarchy in which we recognize overarching large patterns before observing the finer details. Traditional ConvNets do not use this technique. This kind of architecture is much easier to understand as it is more like how human brains work, and more robust against adversarial attacks.

{% include figure image_path="/assets/images/projects/Deep_nets_2020/deep_3.png" caption= "The top down network involves passing a downscaled version of the initial image to the first layer, and the feeding in higher resolution versions of the image in the subsequent layers to imitate the sequential gathering of finer details" text-align=center %}

