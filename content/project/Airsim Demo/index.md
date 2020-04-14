---
title: AirSim Demo [UNDER CONSTRUCTION]
summary:  Integration of our neural estimator in the closed-loop flight control system of Airsim 
date: "2018-06-28T00:00:00Z"


reading_time: false  # Show estimated reading time?
share: false  # Show social sharing links?
profile: false  # Show author profile?
comments: true  # Show comments?
featured: true
math: true

# Optional header image (relative to `static/img/` folder).
image: 
  caption: "Downtown Environment"
  image: "scenario.png"

project: []
---

We integrate our data-driven odometry module in a closed-loop flight control system, providing a new method for real-time autonomous navigation and landing. 

To this end, we generate a simulated \textit{Downtown} environment using Airsim, a flight simulator available as a plugin for Unreal Engine [airsim2017fsr]. 

![png](./sgscenario.png)

We collect images and inertial measurements flying in the simulated environment and we train the model on the new synthetic dataset. 

The network outputs are now the input to the flight control system that generates velocity commands for the UAV system. 

We show through real-time simulations that our closed-loop data-driven control system can successfully navigate and land the UAV on the designed target with less than $10$ cm of error.

![png](./controlsys.png)

In this project I will show the whole pipeline to integrate our architecture in the AirSim flight control system. 

