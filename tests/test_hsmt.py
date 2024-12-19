from unittest import TestCase
from ddt import data, ddt
from tests.circuits import circuit_factories
import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
from src.superop import superoperator
from src.hsmt import hsmt


@ddt
class TestHSMT(TestCase):

    @data(*circuit_factories)
    def test_hsmt(self, circuit_factory):
        n_qubits = circuit_factory["n_qubits"]
        params = circuit_factory["params"]
        mt = circuit_factory["mt"]
        ansatz = circuit_factory["ansatz"]
        initial_rho = circuit_factory.get("initial_rho", None)
        diff_method = "parameter-shift" if initial_rho is not None else "best"

        dev_super = qml.device("default.qubit", wires=2 * n_qubits)

        @qml.qnode(dev_super, diff_method=diff_method)
        @superoperator(initial_rho=initial_rho)
        def qnode_superop(weights):
            ansatz(weights)
            return qml.state()

        mt_fn = hsmt(qnode_superop)
        metric_tensor = mt_fn(params)
        np.testing.assert_allclose(
            mt.real * 1000, metric_tensor.real * 1000, atol=1e-3, rtol=1e-6
        )
        np.testing.assert_allclose(
            mt.imag * 1000, metric_tensor.imag * 1000, atol=1e-3, rtol=1e-6
        )
