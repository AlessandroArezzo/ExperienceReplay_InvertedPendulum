from dynamic import Dynamic
import numpy as np
from math import sin,cos,pi
from random import uniform
from sklearn.utils.extmath import cartesian
from matplotlib import pyplot as plt
from flask import Flask
import os
import matplotlib.animation as animation

# Class that implements the dynamic of the two link manipulator robotic

class LinkRoboticDynamic(Dynamic):
    m1 = 1.25  # (kg)
    m2 = 0.8  # (kg)
    g = 9.81  # (m/s^2)
    b1 = 0.08  # (kg/s)
    b2 = 0.02  # (kg/s)
    l1 = 0.4  # (m)
    l2 = 0.4  # (m)
    c1 = 0.2 # (m)
    c2 = 0.2  # (m)
    I1 = 0.066 # (kg *m^2)
    I2 = 0.043  # (kg *m^2)
    P1=m1 * (c1**2) + m2* (l1**2) +I1
    P2=m2 * (c2**2) + I2
    P3=m2 * l1 * c2
    Q_rew = np.diag([1, 0.05, 1, 0.05])
    bound_theta = np.array([-pi, pi])
    bound_thetadot=np.array([-2*pi,2*pi])
    max_abs_thetadot = max([abs(t) for t in bound_thetadot])
    actions_link1=np.array([-1.5, 0 , 1.5])
    actions_link2 = np.array([-1, 0, 1])
    actions = np.transpose([np.tile(actions_link1, len(actions_link2)), np.repeat(actions_link2, len(actions_link1))])
    label_states = ["theta1", "thetadot1","theta2", "thetadot2"]
    label_action = ["force1","force2"]
    dt = 0.05  # (s)
    num_RBF_grid = 5

    # Return theta value in interval [-pi, pi]
    def getThetaInterval(self, theta):
        if abs(theta) < pi:
            return theta
        elif theta > pi:
            diff = -pi + (theta - pi)
        else:
            diff = pi + (theta + pi)
        if abs(diff) > pi:
            return self.getThetaInterval(abs(diff))
        else:
            return diff

    def setThetaDot(self,current_thetadot, acceleration):
        thetadot = current_thetadot + acceleration * self.dt
        if thetadot > self.max_abs_thetadot: thetadot = self.max_abs_thetadot
        if thetadot < -self.max_abs_thetadot: thetadot = -self.max_abs_thetadot
        return thetadot

    def getState(self, state):
        return state[0], state[1], state[2], state[3]

    def defineMatrixM(self, theta2):
        m11=self.P1+self.P2 +2*self.P3*cos(theta2)
        m12=self.P2+self.P3*cos(theta2)
        m21=self.P2+self.P3*cos(theta2)
        m22=self.P2
        return np.array([[m11,m12],[m21,m22]])

    def defineMatrixC(self, thetadot1, theta2, thetadot2):
        c11=self.b1-self.P3*thetadot2*sin(theta2)
        c12=-self.P3*(thetadot1+thetadot2)*sin(theta2)
        c21=self.P3*thetadot1*sin(theta2)
        c22=self.b2
        return np.array([[c11,c12],[c21,c22]])

    # Return theta double point
    def dynamic(self,current_state,u):
        theta1, thetadot1, theta2, thetadot2 = self.getState(current_state)
        M=self.defineMatrixM(theta2)
        C=self.defineMatrixC(thetadot1,theta2,thetadot2)
        thetadot=np.array([thetadot1,thetadot2])
        accelerations=np.matmul(np.linalg.inv(M),(u-np.matmul(C,thetadot)))
        return accelerations[0],accelerations[1]


    def step_simulate(self, current_state, action):
        current_theta1, current_thetadot1, current_theta2, current_thetadot2=self.getState(current_state)
        theta1_acceleration,theta2_acceleration=self.dynamic(current_state,action)
        thetadot1 = self.setThetaDot(current_thetadot1,theta1_acceleration)
        thetadot2 = self.setThetaDot(current_thetadot2, theta2_acceleration)
        theta1 = self.getThetaInterval(current_theta1 + thetadot1 * self.dt)
        theta2=self.getThetaInterval(current_theta2 + thetadot2 * self.dt)
        next_state=np.array([theta1,thetadot1,theta2,thetadot2])
        return next_state

    # Return reward if the force is applied in current_state
    def reward(self,current_state, action):
        current_theta1, current_thetadot1, current_theta2, current_thetadot2 = self.getState(current_state)
        x = np.array([current_theta1,current_thetadot1,current_theta2, current_thetadot2])
        p = (-x.T.dot(self.Q_rew)).dot(x)
        return p

    # Return a random state in the space state
    def casualState(self):
        random_theta1=uniform(self.bound_theta[0],self.bound_theta[1])
        random_thetadot1=uniform(self.bound_thetadot[0],self.bound_thetadot[1])
        random_theta2 = uniform(self.bound_theta[0], self.bound_theta[1])
        random_thetadot2 = uniform(self.bound_thetadot[0], self.bound_thetadot[1])
        return np.array([random_theta1,random_thetadot1,random_theta2,random_thetadot2])

    # Generate the equidistant grid for the gaussian RBF
    def generateRBFGrid(self):
        angle_ = np.linspace(self.bound_theta[0], self.bound_theta[1], self.num_RBF_grid + 1)
        angle = np.empty(self.num_RBF_grid)
        for i in range(0, len(angle)):
            angle[i] = angle_[i]
        velocity = np.linspace(self.bound_thetadot[0], self.bound_thetadot[1], self.num_RBF_grid)
        grid=cartesian([angle, velocity, angle, velocity])
        return grid

    # Plot states, actions and rewards of a trajectory
    def plotTrajectory(self,states, actions, rewards, label=""):
        print("Plot graph for "+label.lower()+" result...")
        app = Flask(__name__)
        app.config.from_pyfile(os.path.join(".", "./config/app.conf"), silent=False)
        path_trajectory = app.config.get("PATH_TRAJECTORY_RESULT")
        x=np.arange(0,len(states)*self.dt,self.dt)
        y_theta1=[]
        y_thetadot1=[]
        y_theta2 = []
        y_thetadot2 = []

        for s in states:
            y_theta1.append(s[0])
            y_thetadot1.append(s[1])
            y_theta2.append(s[2])
            y_thetadot2.append(s[3])

        actions1=[]
        actions2 = []

        for u in actions:
            actions1.append(u[0])
            actions2.append(u[1])


        plt.xlabel('Time (s)')

        plt.title(label+" result: theta")
        plt.ylabel('Theta')
        plt.plot(x, y_theta1,label="Theta link 1")
        plt.plot(x, y_theta2, label="Theta link 2")
        plt.legend(loc="upper left")
        plt.savefig(path_trajectory+"_"+label+"_theta.png")
        plt.show()

        plt.xlabel('Time (s)')
        plt.title(label+" result: thetadot")
        plt.ylabel('Thetadot')
        plt.plot(x, y_thetadot1, label="Thetadot link 1")
        plt.plot(x, y_thetadot2, label="Thetadot link 2")
        plt.legend(loc="upper left")
        plt.savefig(path_trajectory+"_"+label+"_thetadot.png")
        plt.show()

        plt.xlabel('Time (s)')
        plt.title(label+" result: force")
        plt.ylabel('Force')
        plt.plot(x, actions1, label="Force link 1")
        plt.plot(x, actions2, label="Force link 2")
        plt.legend(loc="upper left")
        plt.savefig(path_trajectory+"_"+label+"_force.png")
        plt.show()

        plt.xlabel('Time (s)')
        plt.title(label + " result: rewards")
        plt.ylabel('Reward')
        plt.plot(x, rewards)
        plt.savefig(path_trajectory+"_"+label+"_reward.png")
        plt.show()

    def readStateFromTrajectory(self,row):
        state=np.array([float(row[1]),float(row[2]), float(row[3]), float(row[4])])
        return state

    def readActionFromTrajectory(self, row):
        action = np.array([float(row[5]),float(row[6])])
        return action

    def readRewardFromTrajectory(self, row):
        reward = float(row[7])
        return reward

    def appendStateToList(self,state,list):
        list.append(state[0])
        list.append(state[1])
        list.append(state[2])
        list.append(state[3])
        return list

    def appendActionToList(self, action, list):
        list.append(action[0])
        list.append(action[1])
        return list

    def getTestStates(self):
        app = Flask(__name__)
        app.config.from_pyfile(os.path.join(".", self.path_config_file), silent=False)
        initial_theta1_states = np.asarray(app.config.get("LINK_THETA1_TEST")) * np.pi
        initial_thetadot1_states = np.asarray(app.config.get("LINK_THETADOT1_TEST")) * np.pi
        initial_theta2_states = np.asarray(app.config.get("LINK_THETA2_TEST")) * np.pi
        initial_thetadot2_states = np.asarray(app.config.get("LINK_THETADOT2_TEST")) * np.pi
        initial_test_state = cartesian([initial_theta1_states, initial_thetadot1_states, initial_theta2_states,
                                        initial_thetadot2_states])
        return initial_test_state

    def getInitialState(self):
        app = Flask(__name__)
        app.config.from_pyfile(os.path.join(".", self.path_config_file), silent=False)
        initial_state =np.array([app.config.get("LINK_INIT_THETA1_RAD")*np.pi, app.config.get("LINK_INIT_THETADOT1_RAD")*np.pi,
                                 app.config.get("LINK_INIT_THETA2_RAD")*np.pi, app.config.get("LINK_INIT_THETADOT2_RAD")*np.pi])
        return initial_state


    def showAnimation(self,states):

        animate=AnimatedLinkRobotic(states,self.l1, self.l2, self.dt)
        animate.show(True)

# Class that stores and shows the animation of a manipulator link robotic simulation
class AnimatedLinkRobotic:
    def __init__(self, data_points, l1, l2, t, blit=True, **fig_kwargs):
        self.animation_length = len(data_points)
        self.rod_length_1 = l1
        self.rod_length_2 = l2
        self.delta_t = t
        self.index_data=0

        self.data = data_points

        self.fig = plt.figure(**fig_kwargs)
        self._createLinkRobotic()
        self.x0 =0
        self.y0 = 0
        self.path_config_file="./config/app.conf"

        self.animation = animation.FuncAnimation(self.fig, self.animation_step,
                                                 frames=self.animation_length,
                                                 init_func=self.animation_init,
                                                 interval=self.delta_t*1000,
                                                 blit=blit, repeat=False)

    def _createLinkRobotic(self):
        xmin, xmax = (min(-self.rod_length_1,-self.rod_length_2) * 2,
                      max(self.rod_length_1,self.rod_length_2) * 2)
        ymin, ymax = (min(-self.rod_length_1,-self.rod_length_2) * 2,
                      max(self.rod_length_1,self.rod_length_2) * 2)
        self.pendulum = self.fig.add_subplot(1, 2, 1,
                                             aspect='equal', autoscale_on=False,
                                             xlim=(xmin, xmax), ylim=(ymin, ymax))
        self.pendulum.grid()
        self.pendulum.title.set_text('Animated Two Link Robotic')
        self.pendulum.plot([0., 0.], [ymin, ymax], 'b--')
        self.pendulum.xaxis.label.set_text('Position (cm)')
        self.link_1, = self.pendulum.plot([], [], 'o-', lw=2)
        self.link_2, = self.pendulum.plot([], [], 'o-', lw=2)

    def _plot_two_link(self, t):
        x1, y1 = (self.x0 - self.rod_length_1 * np.cos(self.data[t][0] + np.pi / 2),
                  self.y0 + self.rod_length_1 * np.sin(self.data[t][0] + np.pi / 2))
        self.link_1.set_data([self.x0, x1], [self.y0, y1])
        x1, y1 = (self.x0 - self.rod_length_2 * np.cos(self.data[t][2] + np.pi / 2),
                  self.y0 + self.rod_length_2 * np.sin(self.data[t][2] + np.pi / 2))
        self.link_2.set_data([self.x0, x1], [self.y0, y1])

    def animation_init(self):
        self.link_1.set_data([], [])
        self.link_2.set_data([], [])
        return self.link_1, self.link_2,

    def animation_step(self, i):
        t = i % self.animation_length
        self._plot_two_link(t)
        return self.link_1, self.link_2,

    def show(self,saved=False):
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