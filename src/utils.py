import pennylane as qml
import numpy as np
from pennylane.operation import AnyWires, Channel, Operation
from functools import lru_cache
import pennylane.numpy as pnp
from functools import reduce
from sympy import lambdify
from sympy.functions import conjugate
from sympy.physics.quantum import TensorProduct


class CRX(qml.CRX):
    def generator(self) -> "qml.Projector":
        return (
            -0.5
            * qml.Projector(np.array([1]), wires=self.wires[0])
            @ qml.PauliX(self.wires[1])
        )


class CRY(qml.CRY):
    def generator(self) -> "qml.Projector":
        return (
            -0.5
            * qml.Projector(np.array([1]), wires=self.wires[0])
            @ qml.PauliY(self.wires[1])
        )


class CRZ(qml.CRZ):
    def generator(self) -> "qml.Projector":
        return (
            -0.5
            * qml.Projector(np.array([1]), wires=self.wires[0])
            @ qml.PauliZ(self.wires[1])
        )


class SuperOpChannel(Operation):
    num_params = 1
    num_wires = 2
    _queue_category = "_ops"
    grad_method = "F"

    def __init__(self, p, wires, channel_name, id=None):

        # channel_name is not trainable but influences the action of the operator,
        # which is why we define it to be a hyperparameter
        self._hyperparameters = {"channel_name": channel_name}

        super().__init__(p, wires, id=id)

    @staticmethod
    @lru_cache()
    def compute_matrix(p, channel_name):
        op_class = getattr(qml, channel_name)
        kraus_mats = pnp.array(op_class.compute_kraus_matrices(p))
        superops = [pnp.kron(pnp.conjugate(mat), mat) for mat in kraus_mats]
        matrix = reduce(lambda a, b: a + b, superops)
        return matrix


class QubitChannel(qml.operation.Channel):
    num_wires = AnyWires
    grad_method = None

    def __init__(self, K_param, K_symbol, K_list, wires=None, id=None):
        K_list_fn = [lambdify(K_symbol, K) for K in K_list]

        super().__init__(K_param, wires=wires, id=id)

        self._hyperparameters = {
            "K_symbol": K_symbol,
            "K_list": K_list,
            "K_list_fn": K_list_fn,
        }

        ######### HAVE TO WRITE CHECKS HERE SAME AS PENNYLANE ##########

    def _flatten(self):
        return (self.data,), (self.wires, ())

    # pylint: disable=arguments-differ, unused-argument
    @classmethod
    def _primitive_bind_call(cls, K_list, wires=None, id=None):
        return super()._primitive_bind_call(*K_list, wires=wires)

    @staticmethod
    def compute_kraus_matrices(
        K_param, K_symbol, K_list, K_list_fn
    ):  # pylint:disable=arguments-differ
        K_list_val = [K_fn(K_param) for K_fn in K_list_fn]
        return K_list_val


class SuperOpQubitChannel(Operation):
    num_wires = AnyWires
    grad_method = "F"

    def __init__(self, K_param, K_symbol, K_list, wires=None, id=None):
        K_list_fn = [lambdify(K_symbol, K) for K in K_list]

        super().__init__(K_param, wires=wires, id=id)

        self._hyperparameters = {
            "K_symbol": K_symbol,
            "K_list": K_list,
            "K_list_fn": K_list_fn,
        }

        ######### HAVE TO WRITE CHECKS HERE SAME AS PENNYLANE ##########

    @staticmethod
    # @lru_cache() # unhashable type: 'numpy.ndarray'
    def compute_matrix(
        K_param, K_symbol, K_list, K_list_fn
    ):  # pylint:disable=arguments-differ

        # kraus_mats = [K_fn(K_param) for K_fn in K_list_fn]
        # superops = [pnp.kron(pnp.conjugate(mat), mat) for mat in kraus_mats]
        # matrix = reduce(lambda a, b: a + b, superops)
        # return matrix

        superops = [TensorProduct(conjugate(mat), mat) for mat in K_list]
        matrix = reduce(lambda a, b: a + b, superops)
        return lambdify(K_symbol, matrix)(K_param)
