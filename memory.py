import numpy as np
import csv

# Class that stores samples.
class Memory:
    def __init__(self,dataset_pathFile,train,result_pathFile):
        self.buffer=[]
        self.dataset_pathFile=dataset_pathFile
        self.train=train
        self.result_pathFile=result_pathFile
        # If it is in learning mode read previous samples
        if self.train:
            try:
                with open(dataset_pathFile) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=',')
                    line_count = 0
                    for row in csv_reader:
                        if line_count > 0:
                            data_to_append=[int(row[0]), int(row[1]), int(row[2]),[float(row[3]), float(row[4])],int(row[5]),[float(row[6]),float(row[7])],float(row[8]), float(row[9])]
                            self.buffer.append(data_to_append)
                        line_count += 1
            except FileNotFoundError:
                with open(self.dataset_pathFile, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        ['k', 'l', 't', 'current_theta', 'current_thetadot', 'force', 'next_theta', 'next_thetadot', 'reward', 'time'])

        # If it is not in learning mod it overwrite the result file with new header row
        else:
            with open(self.result_pathFile, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(
                    ['theta', 'thetadot', 'force', 'next_theta', 'next_thetadot'])

    # Append one example to the memory and save it in the dataset file. If it is not in learning mod save it in result file
    def appendElement(self,k,l,t,current_state,u,next_state,reward,time):
        if self.train:
            data_to_append = [k, l, t, current_state, u, next_state, reward]
            self.buffer.append(np.array(data_to_append))
            with open(self.dataset_pathFile, 'a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([k, l, t, current_state[0], current_state[1], u, next_state[0], next_state[1], reward, time])
        else:
            with open(self.result_pathFile, 'a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([current_state[0], current_state[1], u, next_state[0], next_state[1]])
        print("Iteration "+str(k)+"--> Action u: "+str(u)+" by state ("+str(current_state[0])+","+str(current_state[1])+") --> ("+str(next_state[0])+","+str(next_state[1])+")")

    # Return all sample of a trajectory
    def get_trajectory(self,l,T):
        samples_trajectory=[]
        for sample in self.buffer:
            if self.get_l(sample)==l:
                samples_trajectory.append(sample)
            if len(samples_trajectory)==T:
                break
        return samples_trajectory

    # Methods returns information of the last sample of the last experiment performed
    def final_k(self):
        try:
            return self.buffer[len(self.buffer)-1][0]
        except IndexError:
            return 0

    def final_l(self):
        try:
            return self.buffer[len(self.buffer)-1][1]
        except IndexError:
            return 0

    def final_time(self):
        try:
            return self.buffer[len(self.buffer)-1][7]
        except IndexError:
            return 0


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