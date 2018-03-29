---
title: "A Quickstart Container"
categories:
  - tutorials tools
description: "Our nifty docker container containing just the essentials for computer vision and deep learning"
header:
  teaser: /assets/images/posts/2018-01-30-dockerfile/dockerhero.png
  image: /assets/images/posts/2018-01-30-dockerfile/dockerhero.png
tags:
  - tutorials
  - tools
  - software
  - containers

toc: true
toc_label: "Contents"
toc_icon: "gear"
---

While conducting some of our sessions, we've often had that wiry need where practically no-one has the bare minimum tools installed. Sometimes, it's an entirely different OS. And that makes us think, how do we come up with scalable solutions for porting a myriad amount of libraries? This keeping in mind that we often find ourselves using a mix of tensorflow, caffee, scipy, pcl and opencv.

There's a tradeoff between complete portablity and access to all tools. To be honest, we have included all of them here, so a few of you might find this container slightly bloated.

The link to the dockerfile may be found [here](https://hub.docker.com/r/varun19299/cvi-iitm/).
