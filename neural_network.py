import numpy as np
import math

def calculateDistance(x1,y1,x2,y2):
     dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
     return dist

def rbf(x, c, s):
    value=np.exp(-1 / (2 * s**2) * (calculateDistance(x[0],x[1],c[0],c[1]))**2)
    return value

# Class implements the neural network for approximate the Q function in ER algorithm
class RBFNet(object):

    def __init__(self, k, lr, centers, actions, pathFileParameters , standard_deviations=None, rbf=rbf):
        self.k = k # Number of bases
        self.lr = lr # learning rate
        self.rbf = rbf
        self.centers=centers
        try:
            parameters=np.load( pathFileParameters,allow_pickle=True )
            self.w={}
            for a in actions:
                self.w[a]=parameters.item(0).get(a)
        except FileNotFoundError:
            self.w={}
            for a in actions:
                self.w[a] = np.zeros(k)
        if standard_deviations:
            self.standard_deviations=standard_deviations
        else:
            dMax = max([np.abs(calculateDistance(c1[0],c2[0],c1[1],c2[1])) for c1 in self.centers for c2 in self.centers])
            self.standard_deviations = np.repeat(dMax / np.sqrt(2 * self.k), self.k)


    def fit(self, X, y, u):
        a = np.array([self.rbf(X, c, s) for c, s, in zip(self.centers, self.standard_deviations)])
        F = a.T.dot(self.w.get(u))
        error = -(y - F).flatten()
        self.w[u] = self.w.get(u) - self.lr * a * error

    def predict(self, X, u):
        a = np.array([self.rbf(X, c, s) for c, s, in zip(self.centers, self.standard_deviations)])
        y_pred = a.T.dot(self.w.get(u))
        return y_pred


