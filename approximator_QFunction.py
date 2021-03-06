from __future__ import division
import numpy as np
import csv
np.seterr(divide='ignore', invalid='ignore')


# Class implements the neural network for approximate the Q function in ER algorithm
class ApproximatorQFunction(object):

    def __init__(self, k, lr, centers, actions, pathFileParameters , standard_deviations=None):
        self.k = k # Number of bases
        self.lr = lr # learning rate
        self.centers=centers # RBFs centers
        self.rbfs={} # dictionary that contains RBFi(x)
        self.w = {} # dictionary that contains paramaters. It associate k parameters to each action.
        try:
            # If old file exists read parameters
            for a in actions:
                self.w[repr(a)] = np.zeros(k)
            with open(pathFileParameters) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count = 0
                for row in csv_reader:
                    if line_count > 0:
                        self.w[row[0]][int(row[1])] = float(row[2])
                    line_count += 1

        except FileNotFoundError:
            for a in actions:
                self.w[repr(a)] = np.zeros(k)

        if standard_deviations:
            self.standard_deviations=standard_deviations
        else:
            # Calculate standard deviations as the distance between two points on the centers (to each dimension)
            std=[]
            for i in range(0,self.centers.shape[1]):
                dMin = max([np.abs(np.linalg.norm(c1[i] - c2[i])) for c1 in self.centers for c2 in self.centers])
                for c1 in self.centers:
                    for c2 in self.centers:
                        dist=np.abs(np.linalg.norm(c1[i]-c2[i]))
                        if dist > 0 and dist < dMin:
                            dMin=dist
                std.append(dMin)
            self.standard_deviations=[]
            for i in range(0,self.k):
                self.standard_deviations.append(std)

    # Calculate all normalized RBFs value in x (if this had not already been calculated)
    def calculateRBFsValue(self,X):
        if repr(X) not in self.rbfs:
            normalized_factor=0
            a=np.zeros(self.k)
            for i in range(0,self.k):
                rbf=self.rbf(X,self.centers[i],self.standard_deviations[i])
                a[i]=rbf
                normalized_factor += rbf
            self.rbfs[repr(X)]=a/normalized_factor
        return self.rbfs[repr(X)]


    # Fitting the model
    def fit(self, X, y, u, N):
        u=repr(u)
        a=self.calculateRBFsValue(X)
        for i in range(0, N):
            F = a.T.dot(self.w.get(u))
            error = (y - F).flatten()
            self.w[u] = self.w.get(u) + self.lr * a * error

    # Use the model to predict value in (X,u)
    def predict(self, X, u):
        u = repr(u)
        a = self.calculateRBFsValue(X)
        y_pred = a.T.dot(self.w.get(u))
        return y_pred

    # Calculate x value of the gaussian RBF with center c and radius s
    def rbf(self, x, c, s):
        exponent = 0
        for i in range(0, len(x)):
            dist = (x[i] - c[i]) ** 2
            exponent += -dist /(s[i] ** 2)
        value = np.exp(exponent)
        return value


