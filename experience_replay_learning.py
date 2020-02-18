import random
from neural_network import RBFNet
from memory import Memory
import csv

# Class implements the experience replay algorithm

#Inserire grafici per traiettorie e per performance

class ExperienceReplay:

    def __init__(self,dimension_neural_network, centersRBF, learning_rate, T ,discount_factor, initial_exploration_prob, decays_exploration_prob, N , dynamics,pathCsvMemory, pathFileParameters,train=True,path_trajectory=None, path_performance=None, test_state=None):
        self.dynamics=dynamics # dynamics of the problem
        self.neural_net=RBFNet(dimension_neural_network, learning_rate, centersRBF, dynamics.actions,pathFileParameters) # RBF which approximates Q function
        self.T=T # number of examples for each trajectory
        self.N=N # number of fitting for each sample
        self.discount_factor=discount_factor # discount factor
        self.greedy_param=initial_exploration_prob # exploration parameter
        self.factor_decays_greedy=decays_exploration_prob # degrowth fact of exploration parameter
        self.memory=Memory(pathCsvMemory,path_trajectory, train,dynamics) # object where the examples are stored
        self.pathFileParameters=pathFileParameters # path of file that contains neural network parameters
        self.train=train # determines whether learning is taking place or not
        self.path_performance = path_performance
        self.test_state = test_state

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
        print("Training after trajectory #"+str(l)+"...")
        index = 0
        while index < l * self.T * self.N:
        #while index < l * self.T:
            #sample_to_fit = self.memory.buffer[index]
            sample_to_fit = random.choice(self.memory.buffer)
            #self.learn_by_sample(sample_to_fit,self.N)
            self.learn_by_sample(sample_to_fit, 1)
            if index % 1000 == 0:
                print("Fitting #"+str(index)+"...")
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
        print("Training after trajectory #" + str(l) + "...")
        index=0
        while index < l:
            if index % 1000 == 0:
                print("Training sample #"+str(index)+"...")
            trajectory=random.randint(1, l)
            samples=self.memory.get_trajectory(trajectory, self.T)
            for i in range(0, self.N):
                for sample_to_fit in samples:
                    self.learn_by_sample(sample_to_fit,1)
            index = index + 1
        #np.save(self.pathFileParameters, self.neural_net.w)
        self.updateWeights()

    # Performs fitting n times the RBF network using one example passed as a parameter
    def learn_by_sample(self,sample_to_fit, n):
        sample_state = self.memory.get_current_state(sample_to_fit)
        sample_u = self.memory.get_u(sample_to_fit)
        reward = self.memory.get_reward(sample_to_fit)
        sample_next = self.memory.get_next_state(sample_to_fit)
        exact_y = reward + self.discount_factor * (
            self.neural_net.predict(sample_next, self.bestAction(sample_next)))
        self.neural_net.fit(sample_state, exact_y, sample_u,n)

    def learningPerformance(self,l):
        sum=0
        for s in self.test_state:
            current_state=s
            for i in range(0,1000):
                best_action=self.bestAction(current_state)
                reward=self.dynamics.reward(current_state,best_action)
                sum+=reward
                next_state=self.dynamics.step_simulate(current_state,best_action)
                current_state=next_state
        mean=sum/len(self.test_state)
        self.updatePerformance(l,mean)


    def updatePerformance(self,l,mean):
        if l ==1:
            with open(self.path_performance, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(
                    ['trajectory', 'performance'])
        try:
            with open(self.path_performance, 'a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(
                    [l, mean])
        except FileNotFoundError:
            self.updatePerformance(mean)



    # Execute ER algorithm. If it is in learning mode execute example and performs fitting the RBF network
    # Otherwhise it is use the RBF network that approximate Q function to find the final state
    def trainModel(self,max_trajectories=None):
        #Load previous result and generate random state for first trajectory
        k = 0
        l = 1
        current_state = self.dynamics.casualState()
        print("===== Start training model... =====")
        print("== Simulate trajectory #"+str(l)+" ==")
        while True:
            u=self.selectAction(current_state)
            next_state = self.dynamics.step_simulate(current_state,u)
            reward=self.dynamics.reward(next_state,u)
            t=k-(l-1)*self.T
            self.memory.appendElement(k,l,t,current_state,u,next_state,reward)
            k=k+1
            current_state=next_state
            if k == l * self.T:
                self.Q_Learn_Samples(l)
                #self.Q_Learn_Trajectories(l)
                self.greedy_param=self.greedy_param*self.factor_decays_greedy
                current_state=self.dynamics.casualState()
                self.learningPerformance(l)
                if max_trajectories!= None and max_trajectories == l:
                    break
                l += 1
                print("== Simulate trajectory #" + str(l)+" ==")



    def simulate(self,initial_state,final_state,max_samples):
        k=0
        current_state=initial_state
        while k < max_samples:
            u = self.selectAction(current_state)
            reward=self.dynamics.reward(current_state,u)
            self.memory.writeElementTrajectory(k,current_state,u,reward)
            current_state = self.dynamics.step_simulate(current_state, u)
            k+=1
        #dynamics.plotTrajectory(states_samples,actions_samples,l)
