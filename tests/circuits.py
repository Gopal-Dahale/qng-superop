import pennylane as qml
import numpy as np
import pennylane.numpy as pnp
from sympy import symbols, sqrt, Matrix
from src import QubitChannel


def ansatz_1(params):

    # init state
    qml.Hadamard(0)
    qml.Hadamard(1)
    qml.Hadamard(2)

    # circuit
    qml.Hadamard(0)

    # S and T defined with Phase gate for easy conjugate
    qml.PhaseShift(np.pi / 4, 1)  # T gate
    qml.PhaseShift(np.pi / 2, 2)  # S gate

    qml.PhaseFlip(0.1, wires=0)
    qml.RX(params[0], wires=0)

    qml.DepolarizingChannel(0.1, wires=1)
    qml.DepolarizingChannel(0.1, wires=2)

    qml.CRX(params[1], [2, 1])

    # ccry
    qml.ctrl(qml.RY, (0, 1))(params[2], wires=2)

    qml.AmplitudeDamping(0.05, wires=0)
    qml.PhaseFlip(0.05, wires=1)
    qml.PhaseFlip(0.05, wires=2)


def ansatz_2(params):
    qml.CRX(params[0], [1, 0])
    qml.DepolarizingChannel(0.1, wires=1)
    qml.PhaseFlip(0.2, wires=0)
    qml.IsingXX(params[1], [1, 0])
    qml.Hadamard(1)
    qml.PhaseShift(np.pi / 2, 0)
    qml.AmplitudeDamping(0.3, wires=1)
    qml.PhaseFlip(0.4, wires=0)
    qml.IsingYY(params[2], [1, 0])
    qml.CRZ(params[3], [0, 1])
    qml.IsingZZ(params[0], [1, 0])


d = symbols("d")
alpha = 1 / sqrt(10)
beta = sqrt(1 - d)
gamma = sqrt(d)

K0 = Matrix([[3 * alpha, 0], [0, 3 * beta * alpha]])
K1 = 3 * gamma * alpha * Matrix([[0, 1], [0, 0]])
K2 = Matrix([[beta * alpha, 0], [0, alpha]])
K3 = gamma * alpha * Matrix([[0, 0], [1, 0]])


def ansatz_3(params):
    qml.Hadamard(0)
    qml.Hadamard(1)
    qml.U3(params[0], params[1], params[2], 1)
    qml.DepolarizingChannel(params[1], 1)
    qml.CRY(params[0], [1, 0])
    QubitChannel(params[3], d, [K0, K1, K2, K3], 0)


circuit_factories = [
    {
        "ansatz": ansatz_1,
        "params": pnp.array([0.5, 0.7, 0.3], requires_grad=True),
        "n_qubits": 3,
        "mt": np.array(
            [
                [0.30494013 - 0.0j, 0.00002033 - 0.0j, -0.00011782 - 0.0j],
                [0.00002033 - 0.0j, 0.10272189 - 0.00007835j, 0.00062103 + 0.0j],
                [-0.00011782 - 0.0j, 0.00062103 + 0.0j, 0.00671961 + 0.0j],
            ]
        ),
    },
    {
        "ansatz": ansatz_2,
        "params": pnp.array([0.1, 0.2, 0.3, 0.4], requires_grad=True),
        "n_qubits": 2,
        "mt": np.array(
            [
                [
                    0.06920799 + 0.00005748j,
                    -0.0006866 - 0.0j,
                    -0.00233381 + 0.0j,
                    0.0000954 - 0.0j,
                ],
                [
                    -0.0006866 - 0.0j,
                    0.00269074 - 0.0j,
                    -0.00200614 + 0.0j,
                    0.00172744 + 0.0j,
                ],
                [
                    -0.00233381 + 0.0j,
                    -0.00200614 + 0.0j,
                    0.16519609 - 0.0j,
                    -0.00103876 - 0.0j,
                ],
                [
                    0.0000954 - 0.0j,
                    0.00172744 + 0.0j,
                    -0.00103876 - 0.0j,
                    0.00385051 - 0.0j,
                ],
            ]
        ),
        "initial_rho": np.array(
            [
                [0.475 + 0.0j, 0.175 + 0.0j, 0.175 + 0.0j, 0.175 + 0.0j],
                [0.175 + 0.0j, 0.175 + 0.0j, 0.175 + 0.0j, 0.175 + 0.0j],
                [0.175 + 0.0j, 0.175 + 0.0j, 0.175 + 0.0j, 0.175 + 0.0j],
                [0.175 + 0.0j, 0.175 + 0.0j, 0.175 + 0.0j, 0.175 + 0.0j],
            ]
        ),
    },
    {
        "ansatz": ansatz_3,
        "params": pnp.array([0.1, 0.2, 0.3, 0.4], requires_grad=True),
        "n_qubits": 2,
        "mt": np.array(
            [
                [
                    0.25363448 + 0.00104875j,
                    0.03763647 - 0.00189757j,
                    0.00057615 - 0.00174482j,
                    -0.10601619 - 0.0j,
                ],
                [
                    0.03763647 - 0.00189757j,
                    0.9712772 - 0.0j,
                    0.224923 + 0.00005889j,
                    0.06238552 + 0.0j,
                ],
                [
                    0.00057615 - 0.00174482j,
                    0.224923 + 0.00005889j,
                    0.22605195 + 0.0001073j,
                    0.00004527 - 0.0j,
                ],
                [
                    -0.10601619 - 0.0j,
                    0.06238552 + 0.0j,
                    0.00004527 - 0.0j,
                    0.4406787 + 0.0j,
                ],
            ]
        ),
    },
]
