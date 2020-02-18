from PendulumDynamics import PendulumDynamics
from CartDynamics import CartDynamics
from flask import Flask
import os
from experience_replay_learning import ExperienceReplay
import numpy as np
import csv
from matplotlib import pyplot as plt

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
    path_trajectory = app.config.get("PATH_TRAJECTORY")
    path_performance = app.config.get("PATH_PERFORMANCE")
    path_performance_result=app.config.get("PATH_PERFORMANCE_RESULT")
    num_test=app.config.get("NUM_TEST")
    num_trajectory_test=app.config.get("NUM_TRAJECTORY_TEST")
    max_element_trajectory = app.config.get("MAX_ELEMENT_TRAJECTORY")
    #dynamics=CartDynamics()
    dynamics=PendulumDynamics()
    grid=dynamics.generateRBFGrid(num_rbf_grid)
    if train:
        initial_test_state = dynamics.getTestStates()
        initial_state = None

        # Prepare data for the execution of the algorithm
        print("============ START TRAINING... ============")
        for i in range(1, num_test + 1):
            path_performance_i = path_performance + "_" + str(i) + ".csv"
            try:
                with open(path_performance_i):
                    print("======== TEST #" + str(i) + " FOUND ========")
                    continue
            except FileNotFoundError:
                print("====== TEST #"+str(i)+"... ======")
                path_dataset_i = path_dataset + "_" + str(i) + ".csv"
                path_weights_i = path_weights + "_" + str(i) + ".csv"
                ER = ExperienceReplay(len(grid), grid, lr, T, gamma, greedy_param_init, greedy_param_rate, n, dynamics,
                                      path_dataset_i, path_weights_i, train
                                      , path_trajectory, path_performance_i, initial_test_state)
                ER.trainModel(num_trajectory_test)
        min_performance = []
        mean_performance = []
        max_performance = []
        for i in range(1, num_trajectory_test + 1):
            performance_i = []
            for j in range(1, num_test + 1):
                path_performance_j = path_performance + "_" + str(j) + ".csv"
                with open(path_performance_j) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=',')
                    line_count = 0
                    for row in csv_reader:
                        if line_count == i:
                            performance_i.append(float(row[1]))
                            break
                        line_count += 1
            min_performance.append(min(performance_i))
            mean_performance.append(sum(performance_i) / len(performance_i))
            max_performance.append(max(performance_i))

        x = np.arange(1, num_trajectory_test + 1)
        plt.plot(x, min_performance, label="min")
        plt.plot(x, mean_performance, label="mean")
        plt.plot(x, max_performance, label="max")
        plt.xlabel("trajectory")
        plt.ylabel("reward")
        plt.legend(loc="upper left")
        print("Plot performance result...")
        plt.savefig(path_performance_result)
        plt.show()
    else:
        print("====== START EXPERIMENT... ======")
        initial_test_state=None
        initial_state = dynamics.getInitialState()
        final_state=dynamics.getFinalState()
        best_index=1
        worst_index=1
        best_reward=None
        worst_reward=0
        for i in range(1, num_test + 1):
            print("SIMULATE TRAJECTORY #"+str(i)+"...")
            path_performance_i = path_performance + "_" + str(i) + ".csv"
            path_dataset_i = path_dataset + "_" + str(i) + ".csv"
            path_weights_i = path_weights + "_" + str(i) + ".csv"
            path_trajectory_i=path_trajectory+ "_" + str(i) + ".csv"
            ER = ExperienceReplay(len(grid), grid, lr, T, gamma, greedy_param_init, greedy_param_rate, n, dynamics,
                                  path_dataset_i, path_weights_i, train
                                  , path_trajectory_i, path_performance_i, initial_test_state)
            ER.simulate(initial_state, final_state, max_element_trajectory)
            reward=0
            with open(path_trajectory_i) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count = 0
                for row in csv_reader:
                    if line_count > 0:
                        reward += dynamics.readRewardFromTrajectory(row)
                    line_count += 1
            theta_final = dynamics.readStateFromTrajectory(row)[0]
            print("Trajectory #" + str(i) + " stabilized in "+dynamics.label_states[0]+" = " + str(theta_final))
            if best_reward == None or reward > best_reward:
                best_reward=reward
                best_index=i
            if reward < worst_reward:
                worst_reward=reward
                worst_index=i

        print("The worst result found is in trajectory #"+str(worst_index))
        path_worst_trajectory = path_trajectory + "_" + str(worst_index) + ".csv"
        states = []
        actions= []
        rewards=[]
        with open(path_worst_trajectory) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count > 0:
                    states.append(dynamics.readStateFromTrajectory(row))
                    actions.append(dynamics.readActionFromTrajectory(row))
                    rewards.append(dynamics.readRewardFromTrajectory(row))
                line_count += 1
        #dynamics.plotTrajectory(states,actions,rewards,"Worst")

        print("The best result found is in trajectory #" + str(best_index))
        path_best_trajectory = path_trajectory + "_" + str(best_index) + ".csv"
        states = []
        actions = []
        rewards = []
        with open(path_best_trajectory) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count > 0:
                    states.append(dynamics.readStateFromTrajectory(row))
                    actions.append(dynamics.readActionFromTrajectory(row))
                    rewards.append(dynamics.readRewardFromTrajectory(row))
                line_count += 1
        #dynamics.plotTrajectory(states, actions, rewards, "Best")
        try:
            print("Try to perform the animation of the best trajectory")
            dynamics.showAnimation(states)
        except AttributeError:
            print("Method for animation is not defined")



