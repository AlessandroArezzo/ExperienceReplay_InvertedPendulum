import numpy as np
from math import sin,pi
from random import uniform

# Class that implements the inverted pendulum dynamic

class PendulumDynamics:
    m = 0.055     # (kg)
    l = 0.042     # (m)
    g = 9.81    # (m/s^2)
    b = 3 * 10**-6     # (N*m*s/rad)
    K= 0.0536
    J=1.91*10**-4
    R=9.5
    dt = 0.005   # (s)
    Q_rew = np.diag([5,0.1])
    R_rew = 1
    actions = np.array([-3,0,3])
    max_abs_thetadot=15 * pi
    bound_theta=np.array([-pi,pi])
    bound_thetadot=np.array([-15*pi,15*pi])

    # Return theta value in interval [-pi, pi]
    def getThetaInterval(self,theta):
        if abs(theta)<pi:
            return theta
        elif theta>pi:
            diff = -pi+(theta-pi)
        else:
            diff = pi+(theta+pi)
        if abs(diff) > pi:
            return self.getThetaInterval(abs(diff))
        else:
            return diff


    def getState(self, state):
        return state[0],state[1]

    # Return theta double point
    def dinamics(self,theta,thetadot,u):
        dynamic=(self.m * self.g * self.l * sin(theta) -self.b * thetadot - (self.K**2)*thetadot/self.R + self.K * u /self.R)/self.J
        return dynamic

    # Return next state if the force is applied in current_state
    def step_simulate(self, current_state ,force):
        current_theta, current_thetadot=self.getState(current_state)
        theta_acceleration=self.dinamics(current_theta,current_thetadot,force)
        thetadot = current_thetadot + theta_acceleration * self.dt
        if thetadot>self.max_abs_thetadot: thetadot=self.max_abs_thetadot
        if thetadot<-self.max_abs_thetadot: thetadot=-self.max_abs_thetadot
        theta = self.getThetaInterval(current_theta + thetadot * self.dt)
        next_state=np.array([theta,thetadot])
        #print("("+str(current_theta)+","+str(current_thetadot)+") - Acceleration: "+str(theta_acceleration)+" , Force: "+str(force)+"-->""("+str(theta)+","+str(thetadot)+")")
        return next_state

    # Return reward if the force is applied in current_state
    def reward(self,current_state,force):
        current_theta, current_thetadot = self.getState(current_state)
        x = np.array([current_theta,current_thetadot])
        p = (-x.T.dot(self.Q_rew)).dot(x) - self.R_rew * force**2
        return p

    # Return a random state in the space state
    def casualState(self):
        random_theta=uniform(-pi,pi)
        random_thetadot=uniform(-15*pi,15*pi)
        return np.array([random_theta,random_thetadot])

    # Generate the equidistant grid for the gaussian RBF
    def generateRBFGrid(self, n):
        angle = np.linspace(self.bound_theta[0], self.bound_theta[1],n)
        velocity = np.linspace(self.bound_thetadot[0], self.bound_thetadot[1], n)
        grid = np.zeros(shape=(len(angle) * len(velocity), 2))
        for i in range(0, len(angle)):
            for j in range(0, n):
                grid[(i * n) + j, 0] = angle[i]
        for i in range(0, len(velocity)):
            for j in range(0, n):
                grid[(i * n) + j, 1] = velocity[j]
        return grid


