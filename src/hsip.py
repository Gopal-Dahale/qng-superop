import numpy as np


class HSIP:
    def __init__(self, observable, wires):
        self.observable = observable
        self.obs_vec = observable.matrix(wires).flatten()

    def __call__(self, state, wires):
        return np.dot(self.obs_vec, state).real
