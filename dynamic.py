from abc import ABC

class Dynamic(ABC):
    path_config_file="./config/app.conf"

    def getState(self,state):
        pass

    def dinamics(self, current_state, u):
        pass

    def step_simulate(self, current_state, action):
        pass

    def reward(self, current_state, action):
        pass

    def casualState(self):
        pass

    def generateRBFGrid(self, n):
        pass

    def plotTrajectory(self, states, actions, rewards, label=None):
        pass

    def readStateFromTrajectory(self, row):
        pass

    def readActionFromTrajectory(self, row):
        pass

    def readRewardFromTrajectory(self, row):
        pass

    def appendStateToList(self, state, list):
        pass

    def appendActionToList(self, action, list):
        pass

    def getTestStates(self):
        pass

    def getInitialState(self):
        pass

    def getFinalState(self):
        pass

