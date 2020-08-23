---
layout: single
title: "VR Shooter with Dodging agent"
description: "To make a VR shooter game with agents trained by reinforcement learning"
---

**Goal:**

To make a VR shooter game with agents trained by reinforcement learning

**Objective:**

Investigate factors like reward signals, training practices and environment design that favour cooperation or competition in a multi-agent RL setting.

**Method:**

- Using the ML-Agents framework in Unity3D, agents are trained with RL algorithms like Proximal Policy Optimization (PPO), Soft Actor Critic (SAC) and others.
- Reinforcement Learning (RL) has mainly looked upon the single agent setting, akin to a control system.
- The field of multi-agent RL is nascent, with papers being published nearly every day. This draws parallels with daily human life too. We are trained to choose between working as a team, and competing with one other, to achieve our goals, like hunting animals in stone-age or playing a game of football currently.
- This project wants to find out what roles an agent can learn to take up in a multi-agent setting.
- RL is known for it is highly unstable training phase, and subtle choices in the environment setup and reward signals give rise to different behaviour in agents.
- We plan to implement a shooter game, like Counter-Strike, where a team of agents will have to take up different roles, like pursuit of an opponent, giving cover fire, sniper positions and so on.