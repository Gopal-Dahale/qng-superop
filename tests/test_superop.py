from unittest import TestCase
from ddt import data, ddt
from tests.circuits import circuit_factories
import numpy as np
from tests.utils import generate_random_hamiltonians
import pennylane as qml
import pennylane.numpy as pnp
from src.hsip import HSIP
from src.superop import superoperator


@ddt
class TestSuperOp(TestCase):

    @data(*circuit_factories)
    def test_expval(self, circuit_factory):
        n_qubits = circuit_factory["n_qubits"]
        params = circuit_factory["params"]
        mt = circuit_factory["mt"]
        ansatz = circuit_factory["ansatz"]
        initial_rho = circuit_factory.get("initial_rho", None)

        hams = generate_random_hamiltonians(10, n_qubits, 5)

        dev = qml.device("default.mixed", wires=n_qubits)
        dev_super = qml.device("default.qubit", wires=2 * n_qubits)

        @qml.qnode(dev)
        def cost(weights, obs):
            if initial_rho is not None:
                qml.QubitDensityMatrix(initial_rho, range(n_qubits))
            ansatz(weights)
            return qml.expval(obs)

        @qml.qnode(dev_super)
        @superoperator(initial_rho=initial_rho)
        def qnode_superop(weights):
            ansatz(weights)
            return qml.state()

        def cost_superop(weights, meas):
            sv = qnode_superop(weights)
            return meas(sv, range(n_qubits))

        for obs in hams:
            expval = cost(params, obs)

            meas = HSIP(obs, range(n_qubits))  # Hilbert-Schmidt inner product
            expval_superop = cost_superop(params, meas)

            np.testing.assert_allclose(expval, expval_superop, atol=1e-6)
