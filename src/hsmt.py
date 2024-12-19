from functools import partial
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import pennylane.numpy as pnp

# pylint: disable=too-many-statements,unused-argument
# from pennylane.gradients.metric_tensor import _contract_metric_tensor_with_cjac
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms import transform
from pennylane.typing import PostprocessingFn
from pennylane.gradients.adjoint_metric_tensor import (
    _expand_trainable_multipar,
    _reshape_real_imag,
)
from pennylane.gradients.metric_tensor import _contract_metric_tensor_with_cjac


def finite_diff(op, epsilon=1e-7):
    """
    Need to have this since SupeopChannel does not have a generator.
    Also cannot use autograd jacobian (numpy or pennylane.numpy) since it does not support complex numbers.
    Actually it supports complex number but then I have to define the matrices of noisy gates by myself.
    For complex numbers, probably need to move to jax.
    """
    params_plus = [param + epsilon for param in op.parameters]
    params_minus = [param - epsilon for param in op.parameters]
    mat_plus = op.compute_matrix(*params_plus, **op.hyperparameters)
    mat_minus = op.compute_matrix(*params_minus, **op.hyperparameters)
    return (mat_plus - mat_minus) / (2 * epsilon)


def inverse_op(op):
    return qml.QubitUnitary(
        pnp.linalg.inv(op.compute_matrix(*op.parameters, **op.hyperparameters)),
        wires=op.wires,
    )


def grad_op_lookup(op):
    if op.name in ["CRX", "CRY", "CRZ"]:
        return qml.Identity(op.control_wires) @ op.base

    return op


def derive_op(op):
    generator = qml.generator(op, format="observable")
    return 1j * generator, grad_op_lookup(op)


@partial(
    transform,
    expand_transform=_expand_trainable_multipar,
    classical_cotransform=_contract_metric_tensor_with_cjac,
    is_informative=True,
    use_argnum_in_expand=True,
)
def hsmt(
    tape: QuantumScript,
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    def processing_fn(tapes):
        tape = tapes[0]
        if tape.shots:
            raise ValueError(
                "The adjoint method for the metric tensor is only implemented for shots=None"
            )
        if set(tape.wires) != set(range(tape.num_wires)):
            wire_map = {w: i for i, w in enumerate(tape.wires)}
            tapes, fn = qml.map_wires(tape, wire_map)
            tape = fn(tapes)

        dim = 2**tape.num_wires

        # generate and extract initial state
        prep = (
            tape[0]
            if len(tape) > 0 and isinstance(tape[0], qml.operation.StatePrep)
            else None
        )

        interface = qml.math.get_interface(*tape.get_parameters(trainable_only=False))
        psi = qml.devices.qubit.create_initial_state(tape.wires, prep, like=interface)

        # initialize metric tensor components (which all will be real-valued)
        like_real = qml.math.real(psi[0])
        L = qml.math.convert_like(
            qml.math.zeros((tape.num_params, tape.num_params)), like_real
        )

        ops = tape.operations
        trainable_par_info = []
        for i in tape.trainable_params:
            # print(tape.par_info[i])
            trainable_par_info.append(tape.par_info[i])
        indices_trainable = [info["op_idx"] for info in trainable_par_info]
        fin_trainable_gate_idx = indices_trainable[-1]

        diag = qml.devices.qubit.create_initial_state(tape.wires, prep, like=interface)
        for op in tape.operations[int(prep is not None) : fin_trainable_gate_idx]:
            diag = qml.devices.qubit.apply_operation(op, diag)
        idx_of_last_gate_on_diag = fin_trainable_gate_idx - 1

        for t in range(len(indices_trainable) - 1, -1, -1):
            left_gate_idx = indices_trainable[t]

            # print(t, "Current left trainable op at : ", tape.operations[left_gate_idx])

            # print("Applying Inverse on diag")
            for i in range(idx_of_last_gate_on_diag, left_gate_idx - 1, -1):
                # print(tape.operations[i])
                # if tape.operations[i].name == "StatePrep":
                # print("*" * 10)
                if tape.operations[i].name in ["SuperOpChannel", "SuperOpQubitChannel"]:
                    inv_op = inverse_op(tape.operations[i])
                    diag = qml.devices.qubit.apply_operation(inv_op, diag)
                else:
                    adj_op = qml.adjoint(tape.operations[i], lazy=True)
                    diag = qml.devices.qubit.apply_operation(adj_op, diag)
            idx_of_last_gate_on_diag = left_gate_idx - 1
            # print("-----------------")

            left = diag.copy()
            # apply derivative of gate at index t on left
            # print(
            # 	f"Applying derivative of trainable op {tape.operations[left_gate_idx]} on left"
            # )

            if tape.operations[left_gate_idx].name in [
                "SuperOpChannel",
                "SuperOpQubitChannel",
            ]:
                grad_unitary = finite_diff(tape.operations[left_gate_idx])
            else:
                grad_unitary = qml.operation.operation_derivative(
                    tape.operations[left_gate_idx]
                )
            u = qml.QubitUnitary(grad_unitary, tape.operations[left_gate_idx].wires)
            left = qml.devices.qubit.apply_operation(u, left)

            if tape.operations[left_gate_idx].name not in [
                "SuperOpChannel",
                "SuperOpQubitChannel",
            ]:
                generator, grad_op = derive_op(tape.operations[left_gate_idx])
                grad_unitary_1 = generator.matrix() @ qml.matrix(grad_op)

                # print(grad_unitary)
                # print(generator, grad_op)
                # print(grad_unitary_1)

                assert pnp.allclose(
                    grad_unitary, grad_unitary_1
                ), f"Grad unitary 1 mismatch for {tape.operations[left_gate_idx]}"

            # apply gates on left from left_gate_idx+1 till the end
            # print("Applying gates on left")
            for op in tape.operations[left_gate_idx + 1 :]:
                # print(op)
                # if op.name == "StatePrep":
                # 	print("*" * 10)
                left = qml.devices.qubit.apply_operation(op, left)

            right = left.copy()
            left_real, left_imag = _reshape_real_imag(left, dim)
            right_real, right_imag = _reshape_real_imag(right, dim)
            value = qml.math.dot(left_real, right_real) + qml.math.dot(
                left_imag, right_imag
            )
            # print(f"M_{t, t} = {value}")
            L = qml.math.scatter_element_add(L, (t, t), value)
            # print(left)

            right = diag.copy()

            idx_of_last_gate_on_left = len(tape.operations)
            idx_of_last_gate_on_right = left_gate_idx - 1
            # print("^" * 20)

            for s in range(t - 1, -1, -1):
                right_gate_idx = indices_trainable[s]

                # print(
                #     s,
                #     "Current right trainable op at : ",
                #     tape.operations[right_gate_idx],
                # )

                # apply adjoint to left from idx_of_last_gate_on_left to right_gate_idx + 1 (inclusive)
                # print("Applying adjoint on left")
                for i in range(idx_of_last_gate_on_left - 1, right_gate_idx, -1):
                    # print(tape.operations[i])
                    # if tape.operations[i].name == "StatePrep":
                    #     print("*" * 10)
                    adj_op = qml.adjoint(tape.operations[i], lazy=True)
                    left = qml.devices.qubit.apply_operation(adj_op, left)
                idx_of_last_gate_on_left = right_gate_idx + 1

                # apply adjoint to right from index_of_last_gate_on_right to right_gate_idx (inclusive)
                # print("Applying inverse on right")
                for i in range(idx_of_last_gate_on_right, right_gate_idx - 1, -1):
                    # print(tape.operations[i])
                    # if tape.operations[i].name == "StatePrep":
                    #     print("*" * 10)
                    if tape.operations[i].name in [
                        "SuperOpChannel",
                        "SuperOpQubitChannel",
                    ]:
                        inv_op = inverse_op(tape.operations[i])
                        right = qml.devices.qubit.apply_operation(inv_op, right)
                    else:
                        adj_op = qml.adjoint(tape.operations[i], lazy=True)
                        right = qml.devices.qubit.apply_operation(adj_op, right)
                idx_of_last_gate_on_right = right_gate_idx - 1

                deriv = right.copy()

                # apply derivative of gate at index s on deriv
                # print(
                #     f"Applying derivative of trainable op {tape.operations[right_gate_idx]} on deriv"
                # )

                if tape.operations[right_gate_idx].name in [
                    "SuperOpChannel",
                    "SuperOpQubitChannel",
                ]:
                    grad_unitary = finite_diff(tape.operations[right_gate_idx])
                else:
                    grad_unitary = qml.operation.operation_derivative(
                        tape.operations[right_gate_idx]
                    )
                u = qml.QubitUnitary(
                    grad_unitary, tape.operations[right_gate_idx].wires
                )
                deriv = qml.devices.qubit.apply_operation(u, deriv)

                if tape.operations[right_gate_idx].name not in [
                    "SuperOpChannel",
                    "SuperOpQubitChannel",
                ]:
                    generator, grad_op = derive_op(tape.operations[right_gate_idx])
                    grad_unitary_2 = generator.matrix() @ qml.matrix(grad_op)

                    # print(grad_unitary)
                    # print(generator, grad_op)
                    # print(grad_unitary_2)

                    assert pnp.allclose(
                        grad_unitary, grad_unitary_2
                    ), f"Grad unitary 1 mismatch for {tape.operations[right_gate_idx]}"

                left_real, left_imag = _reshape_real_imag(left, dim)
                deriv_real, deriv_imag = _reshape_real_imag(deriv, dim)

                real_part = qml.math.dot(left_real, deriv_real) + qml.math.dot(
                    left_imag, deriv_imag
                )
                imag_part = qml.math.dot(deriv_real, left_imag) - qml.math.dot(
                    deriv_imag, left_real
                )
                value = real_part + 1j * imag_part

                # value = qml.math.dot(left_real, deriv_real) + qml.math.dot(
                #     left_imag, deriv_imag
                # )

                L = qml.math.scatter_element_add(
                    L,
                    [(t, s), (s, t)],
                    value * qml.math.convert_like(qml.math.ones((2,)), value),
                )
                # print(f"M_{t, s} = {value}")

        metric_tensor = L
        # print(metric_tensor)
        return metric_tensor

    return [tape], processing_fn
