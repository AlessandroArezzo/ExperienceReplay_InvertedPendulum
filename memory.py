import numpy as np
import csv

# Class that stores samples.
class Memory:
    def __init__(self,dataset_pathFile,trajectory_pathFile,train,dynamic):
        self.buffer=[]
        self.dataset_pathFile=dataset_pathFile
        self.trajectory_pathFile=trajectory_pathFile
        self.dynamic=dynamic
        if train:
            with open(self.dataset_pathFile, 'w', newline='') as f:
                writer = csv.writer(f)
                label_to_append=['k', 'l', 't']
                for label in self.dynamic.label_states:
                    label_to_append.append("current_"+label)
                for label in self.dynamic.label_action:
                    label_to_append.append(label)
                for label in self.dynamic.label_states:
                    label_to_append.append("next_"+label)
                label_to_append.append('reward')
                writer.writerow(
                    label_to_append)
        else:
            with open(self.trajectory_pathFile, 'w', newline='') as f:
                writer = csv.writer(f)
                label_to_append = ['k']
                for label in self.dynamic.label_states:
                    label_to_append.append("current_" + label)
                for label in self.dynamic.label_action:
                    label_to_append.append(label)
                label_to_append.append('reward')
                writer.writerow(
                    label_to_append)


    # Append one example to the memory and save it in the dataset file. If it is not in learning mod save it in result file
    def appendElement(self,k,l,t,current_state,u,next_state,reward):
        try:
            with open(self.dataset_pathFile, 'a+', newline='') as f:
                writer = csv.writer(f)
                data_to_append = [k, l, t]
                data_to_append = self.dynamic.appendStateToList(current_state, data_to_append)
                data_to_append.append(u)
                data_to_append = self.dynamic.appendStateToList(next_state, data_to_append)
                writer.writerow(data_to_append)
            self.buffer.append(np.array([k,l,t,current_state,u,next_state,reward]))
            #print("Trajectory: " + str(l) + " - #" + str(t) + " --> Action u: " + str(u) + " by state (" + str(
            #    current_state[0]) + "," + str(current_state[1]) + ") --> (" + str(next_state[0]) + "," + str(
            #    next_state[1]) + ")")

        except:
            with open(self.dataset_pathFile, 'w', newline='') as f:
                writer = csv.writer(f)
                label_to_append = ['k', 'l', 't']
                for label in self.dynamic.label_states:
                    label_to_append.append("current_" + label)
                for label in self.dynamic.label_action:
                    label_to_append.append(label)
                for label in self.dynamic.label_states:
                    label_to_append.append("next_" + label)
                label_to_append.append('reward')
                writer.writerow(
                    label_to_append)
                self.appendElement(k,l,t, current_state, u, next_state, reward)

    def writeElementTrajectory(self,k,current_state,u,reward):
        try:
            with open(self.trajectory_pathFile, 'a+', newline='') as f:
                writer = csv.writer(f)
                data_to_append = [k]
                data_to_append = self.dynamic.appendStateToList(current_state, data_to_append)
                data_to_append.append(u)
                data_to_append.append(reward)
                writer.writerow(data_to_append)
            #print(str(k) + "--> (" + str(current_state[0]) + ", " + str(current_state[1]) + ")")
        except FileNotFoundError:
            with open(self.trajectory_pathFile, 'w', newline='') as f:
                writer = csv.writer(f)
                label_to_append = ['k']
                for label in self.dynamic.label_states:
                    label_to_append.append("current_" + label)
                for label in self.dynamic.label_action:
                    label_to_append.append(label)
                label_to_append.append('reward')
                writer.writerow(
                    label_to_append)
                self.writeElementTrajectory(k,current_state,u,reward)

    # Return all sample of a trajectory
    def get_trajectory(self,l,T):
        samples_trajectory=[]
        for sample in self.buffer:
            if self.get_l(sample)==l:
                samples_trajectory.append(sample)
            if len(samples_trajectory)==T:
                break
        return samples_trajectory


    # Methods returns information of one sample in memory
    def get_K(self,element):
        return element[0]

    def get_l(self,element):
        return element[1]

    def get_t(self,element):
        return element[2]

    def get_current_state(self,element):
        return element[3]

    def get_u(self, element):
        return element[4]

    def get_next_state(self, element):
        return element[5]

    def get_reward(self, element):
        return element[6]