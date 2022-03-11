---
layout: single
title: "AI Games"
description: "Using Reinforcement learning & Other AI paradigms to play partial information games such as Reconnaissance Blind Chess"
---

**Goals:**

- Create a software to compete in RBC competition and beat other competitors
- Focussing on solving the problem of POMDP in RBC

**Objective:**

The aim is to create a working Chess bot that performs better than the top chess engines in RBC

**Method:**

LSTM Approach:
- One hot encoding the belief state for input to LSTM 
which has history of length h.
- Output from LSTM is probability distribution over chess board for the position of 3x3 grid to be chosen.
- 3x3 grid is sampled according to the output of LSTM which is then multiplied with this probability distribution(LSTM output) for the gradient to flow.
- For the loss part we have weighted mean absolute error of 3x3 grid at the position revealed by LSTM at time-step t and t-1 and then maximizing this loss to capture piece movement.
- After solving the problem of POMDP in RBC we look forward to
integrate this with a RL based chess algorithm such as MuZero,
AlphaZero.
