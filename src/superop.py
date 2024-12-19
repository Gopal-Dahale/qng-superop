import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
from src.utils import *
import functools

sign = {
    qml.RY: 1,
    qml.RX: -1,
    qml.RZ: -1,
    CRX: -1,
    CRY: 1,
    CRZ: -1,
    qml.PhaseShift: -1,
    qml.IsingXX: -1,
    qml.IsingYY: -1,
    qml.IsingZZ: -1,
    qml.U3: np.array([1, -1, -1]),
    qml.Rot: np.array([-1, 1, -1]),
}

custom_crot = {"CRX": CRX, "CRY": CRY, "CRZ": CRZ}

noise_gates = [
    "BitFlip",
    "PhaseDamping",
    "DepolarizingChannel",
    "AmplitudeDamping",
    "PhaseFlip",
]


def superoperator(circuit_func=None, initial_rho=None):
    def inner(circuit_func):

        @functools.wraps(circuit_func)
        def wrapper(params):

            tape = qml.transforms.make_tape(circuit_func)(params)
            n_wires = tape.num_wires

            if initial_rho is not None:
                qml.StatePrep(
                    initial_rho.flatten(order="F"),
                    wires=range(2 * n_wires),
                    validate_norm=False,
                )

            for op in tape.operations:
                op_name = op.name
                op_wires = np.array(list(op.wires))

                requires_grad = False
                if qml.operation.is_trainable(op):
                    requires_grad = True
                op_param = pnp.array(op.data, requires_grad=requires_grad)

                if op_name == "PauliY":
                    qml.Y(op_wires)

                    # conjugate of Y
                    qml.Z(op_wires + n_wires)
                    qml.Y(op_wires + n_wires)
                    qml.Z(op_wires + n_wires)
                elif op_name in noise_gates:
                    SuperOpChannel(
                        *op_param, [op_wires[0], op_wires[0] + n_wires], op_name
                    )
                elif op_name == "QubitChannel":
                    SuperOpQubitChannel(
                        *op_param,
                        K_symbol=op.hyperparameters["K_symbol"],
                        K_list=op.hyperparameters["K_list"],
                        wires=[op_wires[0], op_wires[0] + n_wires],
                    )
                else:
                    if op_name in ["CRX", "CRY", "CRZ"]:
                        op_class = custom_crot[op_name]
                        op_class_conj = op_class
                        sign_val = sign.get(op_class, 1)
                    elif op_name in ["C(RY)", "C(RX)", "C(RZ)"]:

                        base_op_class = getattr(qml, op.base.name)
                        base_op_wires = np.array(list(op.base.wires))

                        op_class_conj = qml.ctrl(
                            base_op_class,
                            control=op.control_wires,
                            control_values=op.control_values,
                        )

                        op_class = qml.ctrl(
                            base_op_class,
                            control=np.array(list(op.control_wires)) + n_wires,
                            control_values=op.control_values,
                        )

                        op_wires = base_op_wires
                        sign_val = sign.get(base_op_class, 1)
                    else:
                        op_class = getattr(qml, op_name)
                        op_class_conj = op_class
                        sign_val = sign.get(op_class, 1)

                    op_class_conj(*(op_param * sign_val), wires=op_wires)
                    op_class(*op_param, wires=op_wires + n_wires)
            return qml.state()

        return wrapper

    if circuit_func:
        return inner(circuit_func)

    return inner
