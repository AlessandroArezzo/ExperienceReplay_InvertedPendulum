import numpy as np
import random
from neural_network import RBFNet
from memory import Memory
from PendulumDynamics import PendulumDynamics
import time
from flask import Flask
import os
import csv
# Class implements the experience replay algorithm

class ExperienceReplay:

    def __init__(self,dimension_neural_network, centersRBF, learning_rate, T ,discount_factor, initial_exploration_prob, decays_exploration_prob, N , dynamics,pathCsvMemory, pathFileParameters,train=True, result_pathFile=None):
        self.dynamics=dynamics # dynamics of the problem
        self.neural_net=RBFNet(dimension_neural_network, learning_rate, centersRBF, dynamics.actions, pathFileParameters) # RBF which approximates Q function
        self.T=T # number of examples for each trajectory
        self.N=N # number of fitting for each sample
        self.discount_factor=discount_factor # discount factor
        self.greedy_param=initial_exploration_prob # exploration parameter
        self.factor_decays_greedy=decays_exploration_prob # degrowth fact of exploration parameter
        self.memory=Memory(pathCsvMemory,train,result_pathFile) # object where the examples are stored
        self.pathFileParameters=pathFileParameters # path of file that contains neural network parameters
        self.train=train # determines whether learning is taking place or not
        self.time_extracting_samples = 0 # total time needed to store all the examples of the dataset saved so far

    # Returns the best actions from the actual state (action that brings the greatest reward)
    def bestAction(self,state):
        actions=self.dynamics.actions
        best_action=actions[0]
        best_reward=self.neural_net.predict(state,best_action)
        for i in range(1,len(actions)):
            reward=self.neural_net.predict(state,actions[i])
            if reward > best_reward:
                best_reward = reward
                best_action=actions[i]
        return best_action

    """ Return an action from the actual state. 
     If you are learning, choose a better action with a certain probability or a random action with a uniform probability.
     Otherwise always chooses the best action """
    def selectAction(self,current_state):
        if not self.train:
            return self.bestAction(current_state)
        if random.random() < self.greedy_param:
            return random.choice(self.dynamics.actions)
        else:
            return self.bestAction(current_state)

    # Performs fitting by sample
    def Q_Learn_Samples(self,l):
        index = 0
        while index < l * self.T:
            sample_to_fit = self.memory.buffer[index]
            self.learn_by_sample(sample_to_fit,self.N)
            if index % 1000 == 0:
                print("Training sample #"+str(index)+"...")
            index=index+1
        #np.save(self.pathFileParameters,  self.neural_net.w)
        self.updateWeights()

    def updateWeights(self):
        with open(self.pathFileParameters, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                ['u', 'index', 'value'])
        with open(self.pathFileParameters, 'a+', newline='') as f:
            writer = csv.writer(f)
            for a in self.dynamics.actions:
                index=0
                weights_action=self.neural_net.w.get(a)
                for w in weights_action:
                    writer.writerow(
                        [a, index, w])
                    index+=1


    # Performs fitting by trajectory
    def Q_Learn_Trajectories(self, l):
        index=0
        while index < l:
            trajectory=random.randint(1, l)
            samples=self.memory.get_trajectory(trajectory, self.T)
            for i in range(0, self.N):
                for sample_to_fit in samples:
                    self.learn_by_sample(sample_to_fit,1)
            index = index + 1
        np.save(self.pathFileParameters, self.neural_net.w)

    # Performs fitting n times the RBF network using one example passed as a parameter
    def learn_by_sample(self,sample_to_fit, n):
        sample_state = self.memory.get_current_state(sample_to_fit)
        sample_u = self.memory.get_u(sample_to_fit)
        reward = self.memory.get_reward(sample_to_fit)
        sample_next = self.memory.get_next_state(sample_to_fit)
        for i in range(0, n):
            exact_y = reward + self.discount_factor * (
                self.neural_net.predict(sample_next, self.bestAction(sample_next)))
            self.neural_net.fit(sample_state, exact_y, sample_u)

    # Execute ER algorithm. If it is in learning mode execute example and performs fitting the RBF network
    # Otherwhise it is use the RBF network that approximate Q function to find the final state
    def execute_algorithm(self,initial_state, final_state):
        l = 1
        k = 0
        if self.train:
            k = self.memory.final_k()
            if k > 0:
                k += 1
            l=self.memory.final_l()
            self.greedy_param=self.greedy_param*self.factor_decays_greedy**l
            initial_state=self.dynamics.casualState()
            self.time_extracting_samples=self.memory.final_time()
            self.Q_Learn_Samples(l)
            l = l + 1
        start=time.time()
        current_state = initial_state
        while True:
            u=self.selectAction(current_state)
            next_state = self.dynamics.step_simulate(current_state,u)
            reward=self.dynamics.reward(next_state,u)
            t=k-(l-1)*self.T
            self.memory.appendElement(k,l,t,current_state,u,next_state,reward,self.time_extracting_samples+(time.time() - start))
            k=k+1
            current_state=next_state
            if k == l * self.T and self.train:
                end = time.time()
                self.time_extracting_samples += end - start
                print("Time spent extracting examples --->" + str(self.time_extracting_samples))
                self.Q_Learn_Samples(l)
                l = l +1
                self.greedy_param=self.greedy_param*self.factor_decays_greedy
                current_state=self.dynamics.casualState()
                start = time.time()

                # Performance evaluation to be added

            if not self.train and (abs(current_state - final_state) <= 0.01).all():
                break


if __name__ == '__main__':
    # Read ER algorithm parameters from config files
    app = Flask(__name__)
    app.config.from_pyfile(os.path.join(".", "./config/app.conf"), silent=False)
    lr=app.config.get("LEARNING_RATE")
    T=app.config.get("SAMPLES_FOR_TRAJECTORY")
    gamma = app.config.get("DISCOUNT_FACTOR")
    greedy_param_init = app.config.get("EXPLORATION_PARAM_INIT")
    greedy_param_rate = app.config.get("EXPLORATION_RATE")
    n = app.config.get("FIT_FOR_SAMPLE")
    num_rbf_grid = app.config.get("NUM_RBF_GRID")
    train = app.config.get("TRAIN")
    path_dataset = app.config.get("PATH_DATASET")
    path_weights = app.config.get("PATH_WEIGHTS")
    path_result = app.config.get("PATH_RESULT")

    initial_state=None
    if not train:
        initial_state=np.array( [app.config.get("INIT_THETA_RAD") * np.pi, app.config.get("INIT_THETADOT_RAD") * 15* np.pi])

    # Prepare data for the execution of the algorithm
    dynamics=PendulumDynamics()
    grid=dynamics.generateRBFGrid(num_rbf_grid)
    ER=ExperienceReplay(len(grid),grid,lr,T,gamma,greedy_param_init,greedy_param_rate,n,dynamics,path_dataset,path_weights,train
                        ,path_result)

    # Execute algorithm
    ER.execute_algorithm(initial_state,np.array([0,0]))