---
layout: single
title: "Using our Docker Containers"
category: docker
description: "Docker Container Usage"
header:
  teaser: /assets/images/posts/2018-01-30-dockerfile/dockerhero.png
toc: true
---

In this post, we shall explain how you can get started with our containers - for either python, deep learning, computer vision or probabilistic modelling. 

We are using these containers to manage session requirements for the Fall of 2018.

## Getting Started with Docker 

Firstly, download a compatible version of docker from here:

* For installation of Docker in Windows - [link](https://download.docker.com/win/stable/DockerToolbox.exe) 
* For installation of Docker in Linux(Ubuntu) - direct curl command available.  
* For installation of Docker in Mac - [link](https://download.docker.com/mac/stable/Docker.dmg)

_Note: while installing dockertools for windows, check all options inclduing UEFI and Virtualisation_

## Installation

### Linux

* Run `curl -fsSL get.docker.com | sh` to get the latest version of docker.

* Open your terminal, run `docker --version` to output the version.

### macOS

* After you have installed docker, drag it to your applications folder.

* Run the docker app, and open a new terminal.

* Verify your docker version with `docker --version`.

### Windows:

* Before installation, additional software packages like Kitematic and Virtualbox need to be marked, ensure you check all of them during the installer. 
* The installer adds Docker Toolbox to your Applications folder.   
* On your Desktop, find the Docker QuickStart Terminal icon.  
* Click the Docker QuickStart icon to launch a pre-configured Docker Toolbox terminal.    
* If the system displays a User Account Control prompt to allow VirtualBox to make changes to your computer. Choose Yes.  
* The terminal does several things to set up Docker Toolbox for you. When it is done, the terminal displays a prompt.  

* The prompt is traditionally a \$ dollar sign. You type commands into the command line which is the area after the prompt. Your cursor is indicated by a highlighted area or a \| that appears in the command line. After typing a command, always press RETURN.

## Docker Hello World

* Type the **docker run hello-world** command and press RETURN.
* The command does some work for you, if everything runs well, the command’s output looks like this:

```
$ docker run hello-world
 Unable to find image 'hello-world:latest' locally
 Pulling repository hello-world
 91c95931e552: Download complete
… … … …
```

## Running the Container

* If you are on windows, run the command **docker-machine ip** and make a note of the IP address shown as output.

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
[I 13:42:26.667 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[W 13:42:26.668 NotebookApp] No web browser found: could not locate runnable browser.
[C 13:42:26.669 NotebookApp]

    Copy/paste this URL into your browser when you connect for the first time,
    to login with a token:
        http://(5e3ca2f04d54 or 127.0.0.1):8888/?token=56d48e36ca256e00823506c4f2cf1fc89264a3ba025d3307
```

* Go to `localhost:8888` and enter the token (everthing from `token = `) there. Again, if on windows copy this URL, replace everything within () to the IP address that you noted down in the first step and paste this new URL in your browser.

For example, http://192.168.99.100:8888/?token=e99ef0776ac2c2d848d580e7e86d10a5f8e187fe20be8ae3

* You are good to go if Jupyter Notebooks successfully opens up in your browser. One noted issue on windows is that Edge doesnot support using localhost. You would require chrome or firefox for the same.

* Feel free to raise any issues regarding installation at [github issues](https://github.com/iitmcvg/Content/issues) with the tag `docker install issue`. Elaborate on the specifics of the issue and we'll try to address them.





