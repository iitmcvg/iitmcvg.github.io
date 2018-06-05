---
layout: single
title: "Deep Fake"
description: "Developing face morphing systems for images and videos"
---
__Deep Fake__ {% raw %}(Deep Learning + Fake){% endraw %} is a human image synthesis technique using artificial intelligence methods. It has often been used for doing face swaps, especially with celebrities.
This project aims to develop a robust and fast face morphing system for images and video, using Autoencoders and GANs (Generative Adversarial Networks).The main application of the project is to make personalised marketing software for clothes and fashion items in online retail stores.

{% include figure image_path="/assets/images/projects/DeepFakes/facemorph.png" caption="Face morphing" %}

We aim to extend the previous work done by the subreddit of the same name. In addition to swapping two given faces, we improve on the previous work to make it more usable for deployment in the following manner:
* Computing a metric for semblance of two faces. We have decided to adopt a facenet type anchor while defining corresponding loss metrics.
* Swapping for multiple people in a given frame.

