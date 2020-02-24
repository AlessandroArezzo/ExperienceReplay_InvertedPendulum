from abc import ABC

# Abstract class of a dynamic. All concrete problems must be represented by a classes that implement it.
class Dynamic(ABC):
    path_config_file="./config/app.conf"

    @property
    def actions(self):
        raise NotImplementedError

    @property
    def label_states(self):
        raise NotImplementedError

    @property
    def label_action(self):
        raise NotImplementedError

    def getState(self,state):
        pass

    def dynamic(self, current_state, u):
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


