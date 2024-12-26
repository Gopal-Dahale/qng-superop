import pennylane.numpy as pnp
import pennylane as qml
import autograd.numpy as anp


class HSIP:
    def __init__(self, observable, wires):
        self.observable = observable
        self.obs_vec = observable.matrix(wires).flatten()

    def __call__(self, state):
        # print(type(state))
        # print(type(self.obs_vec))
        out = pnp.dot(self.obs_vec, state)
        # print(type(out))
        # print(dir(out))
        # print(out.shape)
        return anp.real(out)
