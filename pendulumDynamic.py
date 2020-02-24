import numpy as np
from math import sin,pi
from random import uniform
from matplotlib import pyplot as plt
from flask import Flask
import os
from dynamic import Dynamic
import matplotlib.animation as animation
from sklearn.utils.extmath import cartesian

# Class that implements the inverted pendulum dynamic

class PendulumDynamic(Dynamic):
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
    bound_theta=np.array([-pi,pi])
    bound_thetadot=np.array([-15*pi,15*pi])
    max_abs_thetadot=max([abs(t) for t in bound_thetadot])
    label_states=["theta","thetadot"]
    label_action=["force"]
    num_RBF_grid=11

    # Return theta value in interval [-pi, pi)
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
    def dinamics(self,current_state,u):
        theta, thetadot = self.getState(current_state)
        dynamic=(self.m * self.g * self.l * sin(theta) -self.b * thetadot - (self.K**2)*thetadot/self.R + self.K * u /self.R)/self.J
        return dynamic

    # Return next state if the force is applied in current_state
    def step_simulate(self, current_state, action):
        current_theta, current_thetadot=self.getState(current_state)
        theta_acceleration=self.dinamics(current_state,action)
        thetadot = current_thetadot + theta_acceleration * self.dt
        if thetadot>self.max_abs_thetadot: thetadot=self.max_abs_thetadot
        if thetadot<-self.max_abs_thetadot: thetadot=-self.max_abs_thetadot
        theta = self.getThetaInterval(current_theta + thetadot * self.dt) #theta(t) ed eliminare controllo su theta
        next_state=np.array([theta,thetadot])
        #print("("+str(current_theta)+","+str(current_thetadot)+") - Acceleration: "+str(theta_acceleration)+" , Force: "+str(force)+"-->""("+str(theta)+","+str(thetadot)+")")
        return next_state

    # Return reward if the force is applied in current_state
    def reward(self,current_state, action):
        current_theta, current_thetadot = self.getState(current_state)
        x = np.array([current_theta,current_thetadot])
        p = (-x.T.dot(self.Q_rew)).dot(x) - self.R_rew * action**2
        return p

    # Return a random state in the space state
    def casualState(self):
        random_theta=uniform(self.bound_theta[0],self.bound_theta[1])
        random_thetadot=uniform(self.bound_thetadot[0],self.bound_thetadot[1])
        return np.array([random_theta,random_thetadot])

    # Generate the equidistant grid for the gaussian RBF
    def generateRBFGrid(self):
        angle_ = np.linspace(self.bound_theta[0], self.bound_theta[1], self.num_RBF_grid + 1)
        angle = np.empty(self.num_RBF_grid)
        for i in range(0, len(angle)):
            angle[i] = angle_[i]
        velocity = np.linspace(self.bound_thetadot[0], self.bound_thetadot[1], self.num_RBF_grid)
        grid = cartesian([angle, velocity])
        return grid

    # Plot states, actions and rewards of a trajectory
    def plotTrajectory(self,states,actions, rewards, label=None):
        print("Plot graph for "+label.lower()+" result...")
        app = Flask(__name__)
        app.config.from_pyfile(os.path.join(".", "./config/app.conf"), silent=False)
        path_trajectory = app.config.get("PATH_TRAJECTORY_RESULT")
        for i in range(0,len(states[0])):
            path_trajectory+="_"+str(states[0][i]/np.pi)
        x=np.arange(0,len(states)*self.dt,self.dt)
        y_theta=[]
        y_thetadot=[]

        for s in states:
            y_theta.append(s[0])
            y_thetadot.append(s[1])

        plt.xlabel('Time (s)')

        plt.title(label+" result: theta")
        plt.ylabel('Theta')
        plt.plot(x,y_theta)
        plt.savefig(path_trajectory+"_"+label+"_theta.png")
        plt.show()

        plt.title(label+" result: thetadot")
        plt.ylabel('Thetadot')
        plt.plot(x,y_thetadot)
        plt.savefig(path_trajectory+"_"+label+"_thetadot.png")
        plt.show()

        plt.title(label+" result: force")
        plt.ylabel('Force')
        plt.plot(x, actions)
        plt.savefig(path_trajectory+"_"+label+"_force.png")
        plt.show()

        plt.title(label + " result: rewards")
        plt.ylabel('Reward')
        plt.plot(x, rewards)
        plt.savefig(path_trajectory+"_"+label+"_reward.png")
        plt.show()



    def readStateFromTrajectory(self,row):
        state=np.array([float(row[1]),float(row[2])])
        return state

    def readActionFromTrajectory(self, row):
        action =float(row[3])
        return action

    def readRewardFromTrajectory(self, row):
        reward = float(row[4])
        return reward

    def appendStateToList(self,state,list):
        list.append(state[0])
        list.append(state[1])
        return list

    def appendActionToList(self, action, list):
        list.append(action)
        return list

    def getTestStates(self):
        app = Flask(__name__)
        app.config.from_pyfile(os.path.join(".", self.path_config_file), silent=False)
        initial_theta_states = np.asarray(app.config.get("PENDULUM_THETA_TEST")) * np.pi
        initial_thetadot_states = np.asarray(app.config.get("PENDULUM_THETADOT_TEST")) * np.pi
        initial_test_state = cartesian([initial_theta_states, initial_thetadot_states])
        return initial_test_state


    def getInitialState(self):
        app = Flask(__name__)
        app.config.from_pyfile(os.path.join(".", self.path_config_file), silent=False)
        initial_state =np.array([app.config.get("PENDULUM_INIT_THETA_RAD")*np.pi, app.config.get("PENDULUM_INIT_THETADOT_RAD")*np.pi])
        return initial_state

    def showAnimation(self,states):

        animate=AnimatedPendulum(states,self.l,self.dt)
        animate.show(True)

# Class that stores and shows the animation of a pendulum simulation
class AnimatedPendulum:

    def __init__(self, data_points, l, t, blit=True, **fig_kwargs):
        self.animation_length = len(data_points)
        self.rod_length = l
        self.delta_t = t
        self.index_data=0

        self.data = data_points

        self.fig = plt.figure(**fig_kwargs)
        self._create_pendulum()
        self.x0 = 0
        self.y0 = 0
        self.path_config_file="./config/app.conf"

        self.animation = animation.FuncAnimation(self.fig, self.animation_step,
                                                 frames=self.animation_length,
                                                 init_func=self.animation_init,
                                                 interval=self.delta_t*1000,
                                                 blit=blit, repeat=False)

    def _create_pendulum(self):
        xmin, xmax = (-self.rod_length*2,
                     self.rod_length*2)
        ymin, ymax = (-self.rod_length * 2, self.rod_length * 2)
        self.pendulum = self.fig.add_subplot(1, 2, 1,
                                             aspect='equal', autoscale_on=False,
                                             xlim=(xmin, xmax), ylim=(ymin, ymax))
        self.pendulum.grid()
        self.pendulum.title.set_text('Animated Pendulum')
        self.pendulum.plot([0., 0.], [ymin, ymax], 'b--')
        self.pendulum.xaxis.label.set_text('Position (cm)')
        self.rod, = self.pendulum.plot([], [], 'o-', lw=2)

    def _plot_pendulum(self, t):
        x1, y1 = (self.x0 - self.rod_length * np.cos(self.data[t][0] + np.pi / 2),
                  self.y0 + self.rod_length * np.sin(self.data[t][0] + np.pi / 2))
        self.rod.set_data([self.x0, x1], [self.y0, y1])

    def animation_init(self):
        self.rod.set_data([], [])
        return self.rod,

    def animation_step(self, i):
        t = i % self.animation_length
        self._plot_pendulum(t)
        return self.rod,

    def show(self, saved=False):
        if saved:
            self.save()
        print("Show animation...")
        plt.show()

    def save(self):
        print("Saving animation...")
        app = Flask(__name__)
        app.config.from_pyfile(os.path.join(".",self.path_config_file), silent=False)
        path_animation = app.config.get("PATH_ANIMATION")
        for i in range(0,len(self.data[0])):
            path_animation+="_"+str(self.data[0][i]/np.pi)
        path_animation+=".mp4"
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=self.animation_length/(self.animation_length*self.delta_t),metadata=dict(artist='Me'))
        self.animation.save(path_animation, writer=writer)