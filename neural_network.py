import numpy as np
import math
import csv
from mpl_toolkits import mplot3d
from PendulumDynamics import PendulumDynamics
from matplotlib import pyplot as plt


def rbf(x, c, s):
    dist = np.linalg.norm(x - c)
    value = np.exp(-1 / (2 * s ** 2) * (dist) ** 2)
    return value

# Class implements the neural network for approximate the Q function in ER algorithm
class RBFNet(object):

    def __init__(self, k, lr, centers, actions, pathFileParameters , standard_deviations=None, rbf=rbf):
        self.k = k # Number of bases
        self.lr = lr # learning rate
        self.rbf = rbf
        self.centers=centers
        try:
            self.w = {}
            for a in actions:
                self.w[a] = np.zeros(k)
            with open(pathFileParameters) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count = 0
                for row in csv_reader:
                    if line_count > 0:
                        self.w[int(row[0])][int(row[1])] = float(row[2])
                    line_count += 1

        except FileNotFoundError:
            self.w = {}
            for a in actions:
                self.w[a] = np.zeros(k)
        if standard_deviations:
            self.standard_deviations=standard_deviations
        else:
            dMax = max([np.abs(np.linalg.norm(c1 - c2)) for c1 in self.centers for c2 in self.centers])
            self.standard_deviations = np.repeat(dMax / np.sqrt(2 * self.k), self.k)


    def fit(self, X, y, u):
        a = np.array([self.rbf(X, c, s) for c, s, in zip(self.centers, self.standard_deviations)])
        F = a.T.dot(self.w.get(u))
        error = (y - F).flatten()
        self.w[u] = self.w.get(u) + self.lr * a * error

    def predict(self, X, u):
        a = np.array([self.rbf(X, c, s) for c, s, in zip(self.centers, self.standard_deviations)])
        y_pred = a.T.dot(self.w.get(u))
        return y_pred

#Neural network testing

if __name__ == '__main__':
    dynamics=PendulumDynamics()
    grid=dynamics.generateRBFGrid(11)
    neural_net=RBFNet(len(grid),0.1,grid,dynamics.actions,"./data/weights_testNet.csv")
    n_samples=1000
    X_SAMPLES_THETA=[]
    X_SAMPLES_THETADOT=[]
    X_SAMPLES=[]
    Y_SAMPLES=[]

    for i in range(0,n_samples):
        randomState=dynamics.casualState()
        X_SAMPLES_THETA.append(randomState[0])
        X_SAMPLES_THETADOT.append(randomState[1])
        Y_SAMPLES.append(randomState[0]+randomState[1]*2)
        X_SAMPLES.append(randomState)
        for j in range(0,10):
            neural_net.fit([X_SAMPLES_THETA[i],X_SAMPLES_THETADOT[i]],Y_SAMPLES[i],0)

    ax = plt.axes(projection='3d')
    ax.plot3D(X_SAMPLES_THETA, X_SAMPLES_THETADOT, Y_SAMPLES, 'gray')
    plt.show()

    X_THETA=[]
    X_THETADOT=[]
    X=[]
    y=[]
    for i in range(0,n_samples):
        randomState=dynamics.casualState()
        X_THETA.append(randomState[0])
        X_THETADOT.append(randomState[1])
        X.append(randomState)
        y.append(neural_net.predict(randomState,0))

    ax = plt.axes(projection='3d')
    ax.plot3D(X_THETA, X_THETADOT, y, 'gray')
    plt.show()
