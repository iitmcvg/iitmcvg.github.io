---
layout: single
title: "Session 1: Intro to Computer Vision"
category: sessions
description: "Session announcement on 10th Sept 2018"
header:
  teaser: https://scontent-maa2-1.xx.fbcdn.net/v/t1.0-9/39040858_2015575558474735_7568607226030456832_o.jpg?_nc_cat=0&oh=3a8a421ecca018a7479881f81c196e42&oe=5C0220F7
toc: true
---

# Session 1: Introduction to Computer Vision

Greetings from the Computer Vision and Intelligence group, CFI!

We are overwhelmed with the response we received for our introductory session. And it's time to get into some nitty-gritty of Computer Vision. Our next session will cover the fundamentals of Computer Vision using OpenCV.

**Github Link:** https://github.com/iitmcvg/Content/tree/master/Sessions/CV_Intro_Session_1_2018
**Date :** 10th September 2018 (Monday)
**Venue :** ESB 127
**Time :**8:00 pm- 10:30 pm

<iframe src="https://www.google.com/maps/embed?pb=!1m23!1m12!1m3!1d124406.89289444235!2d80.16030355909216!3d12.990045923321086!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!4m8!3e6!4m0!4m5!1s0x3a52677fdb777ceb%3A0xb9d8a78a4b0ef7d3!2sClass+Room+Complex%2C+IIT+Madras%2C+Indian+Institute+Of+Technology%2C+Chennai%2C+Tamil+Nadu+600036!3m2!1d12.9900553!2d80.2303441!5e0!3m2!1sen!2sin!4v1522947421266" width="400" height="300" frameborder="0" style="border:0" allowfullscreen></iframe>

## Getting Started with Docker 

Firstly, download a compatible version of docker from here:

* For installation of Docker in Windows - https://download.docker.com/win/stable/DockerToolbox.exe 
* For installation of Docker in Linux(Ubuntu) - https://docs.docker.com/install/linux/docker-ce/ubuntu/   
* For installation of Docker in Mac -   
https://download.docker.com/mac/stable/Docker.dmg

_Note: while installing dockertools for windows, check all options inclduing UEFI and Virtualisation_


### Installation

* Before installation, additional software packages like Kitematic and Virtualbox can be unchecked. 
* The installer adds Docker Toolbox to your Applications folder.   
* On your Desktop, find the Docker QuickStart Terminal icon.  
* Click the Docker QuickStart icon to launch a pre-configured Docker Toolbox terminal.    
* If the system displays a User Account Control prompt to allow VirtualBox to make changes to your computer. Choose Yes.  
* The terminal does several things to set up Docker Toolbox for you. When it is done, the terminal displays the $ prompt.  
* Make the terminal active by clicking your mouse next to the $ prompt.
* The prompt is traditionally a $ dollar sign. You type commands into the command line which is the area after the prompt. Your cursor is indicated by a highlighted area or a | that appears in the command line. After typing a command, always press RETURN.

* Type the **docker run hello-world** command and press RETURN.
* The command does some work for you, if everything runs well, the command’s output looks like this:

```
$ docker run hello-world
 Unable to find image 'hello-world:latest' locally
 Pulling repository hello-world
 91c95931e552: Download complete
… … … …
```

### Running the Container

* Run the command **docker-machine ip** and make a note of the IP address shown as output.

* Run the container:
```
docker run -it --name cvi --rm -p 8888:8888 iitmcvg/session:intro_CV bash
```

The image has the following tools:

  * OpenCV 3.4.1
  * Tensorflow 1.10
  * Keras
  * Jupyter
  * Scientific python: Numpy, Scipy, Matplotlib ... etc.

* The command does some work for you. Downloading takes around 5 minutes. Be patient. Once the extraction is complete, you should see a terminal shell corresponding to the container (eg: root@xxxxxx).

* Now, update session contents by giving the following command:

```
git pull
```

* Run Jupyter with the following command.

```
jupyter notebook --ip=0.0.0.0 --allow-root
```

* If everything goes well, the command’s output should look like this:

```
root@5e3ca2f04d54:/Content/Sessions/CV_Intro_Session_1_2018# jupyter notebook --ip=0.0.0.0 --allow-root
[I 13:42:26.419 NotebookApp] Writing notebook server cookie secret to /root/.local/share/jupyter/runtime/notebook_cookie_secret
[I 13:42:26.667 NotebookApp] Serving notebooks from local directory: /Content/Sessions/CV_Intro_Session_1_2018
[I 13:42:26.667 NotebookApp] The Jupyter Notebook is running at:
[I 13:42:26.667 NotebookApp] http://(5e3ca2f04d54 or 127.0.0.1):8888/?token=56d48e36ca256e00823506c4f2cf1fc89264a3ba025d3307
[I 13:42:26.667 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[W 13:42:26.668 NotebookApp] No web browser found: could not locate runnable browser.
[C 13:42:26.669 NotebookApp]

    Copy/paste this URL into your browser when you connect for the first time,
    to login with a token:
        http://(5e3ca2f04d54 or 127.0.0.1):8888/?token=56d48e36ca256e00823506c4f2cf1fc89264a3ba025d3307
```

* Copy this URL, replace everything within () to the IP address that you noted down in the first step and paste this new URL in your browser.

For example, http://192.168.99.100:8888/?token=e99ef0776ac2c2d848d580e7e86d10a5f8e187fe20be8ae3

* You are good to go if Jupyter Notebooks successfully opens up in your browser.





