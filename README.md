# ExperienceReplay_RealTimeControl

This project contains the implementation of the experience replay algorithm for solving two real-time control problems:
the stabilization of a pendulum and a robotic manipulator in an equilibrium position.
The aim of the project is to verify the efficiency of the algorithm applied to the two contexts considered.
<h3>How to use the code</h3>
The project includes a script that allows you to run tests. <br>
The code can be used in two ways:
<ul>
<li><b>Train mode:</b> allows you to train the model</li>
<li><b>Simulation mode:</b> allows you to use the trained model to perform a simulation</li>
</ul>
To use the code, the following parameters must be set in the app.config file:
<ul>
<li><b>Train:</b> boolean that determines the mode of execution (True for train mode and False for simulation mode)</li>
<li><b>Experience replay parameters:</b> learning rate, discount factor, number of replays, number of sample for trajectory
exploration parameter. These parameters are required for the train mode.</li>
<li><b>File paths:</b> file paths in which to store datasets, weights, performance( (train mode) and result of a
trajectory (simulation mode) </li>
<li><b>Tests parameters:</b>: num of tests, num of trajectory to each test, dynamic to test, for train mode 
how to perform the tests (parallel or sequential).</li>
<li><b>Simulation parameters:</b> length of simulation and its initial state</li>
</ul>
To train a set of models, set the TRAIN parameter to True, 
set the number of models to be generated in NUM_TEST and 
the number of paths of each model in NUM_TRAJECTORY_TEST. 
The process generates a graph showing the performances of the models as a function of the trajectories visited by the algorithm.<br>
<div align="center">
    <img src="/image/pendulum/pendulum_performance_result.png" width="400px"</img> 
</div>
To use the generated models to run a simulation, set TRAIN to False,
the initial state of the simulation and its length in MAX_ELEMENT_TRAJECTORY.
At the end of the execution the trend of the best simulation is shown in the output.
<div align="center">
    <img src="/image/pendulum/pendulum__-1.0_0.0_Best_theta.png" width="400px"</img> 
    <img src="/image/pendulum/pendulum__-1.0_0.0_Best_thetadot.png" width="400px"</img> 
    <img src="/image/pendulum/pendulum__-1.0_0.0_Best_force.png" width="400px"</img> 
    <img src="/image/pendulum/pendulum__-1.0_0.0_Best_reward.png" width="400px"</img> 
</div>
<video width="320" height="240" autoplay>
  <source src="/animation/animation_pendulum_-1.0_0.0.mp4" type="video/mp4">
</video>
