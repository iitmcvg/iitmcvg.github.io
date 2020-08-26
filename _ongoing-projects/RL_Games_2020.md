---
layout: single
title: "Exploration into RL algorithms through games"
description: "Analyzing the performance of various RL algorithms on different games such as Minesweeper,Slither.io and Reconnaissance Blind Chess"
---

**Goal:**

To analyze the performance of various RL algorithms on different games such as Minesweeper, [Slither.io](http://slither.io/) and Reconnaissance Blind Chess.

**Objective:**

By working on multiplayer environments and incomplete information problems, we intend to find improvements of the current state-of-the-art methods which can find applications in sophisticated problems such as robotics and autonomous driving.

**Method:**

- The agent is made to interact with an emulated game environment. In case of slither, the OpenAI universe package is used to create a container image of the online version of the game while in case of Minesweeper, a pygame environment is used.
- We stack 4-6 images as one training input to add a sense of direction to the game and pass this image to a CNN followed by DENSE LAYERED NEURAL NETWORK. The output is value of the different states or policy depending on the algorithm. (Value of a state tells how good or bad the state/snapshot/frame of the game is, and policy is the strategy based on which the bot takes actions)
- The agent is trained using Q-value based and Policy based methods such as Deep Q-learning, Policy gradients and Actor-Critic methods (to get the best of both worlds)
- The effects of reward shaping, priority experience replay queues, recurrent and LSTM memory layers (in case of Partially Observable MDPs) etc. on the performance of the agent are analyzed.
- Document the training and compare the success of the different algorithms