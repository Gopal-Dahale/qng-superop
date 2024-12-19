import numpy as np
import pennylane as qml


def generate_random_hamiltonians(n_hams, n_qubits, n_terms, seed=42):
	"""
	Generate a random Hamiltonian with n_terms for n_qubits.

	Args:
		n_qubits (int): Number of qubits in the system.
		n_terms (int): Number of terms in the Hamiltonian.

	Returns:
		qml.Hamiltonian: The generated random Hamiltonian.
	"""
	np.random.seed(seed)
	paulis = ["X", "Y", "Z", "I"]
	hams = []
	for _ in range(n_hams):
		coeffs = np.random.uniform(-1, 1, size=n_terms)
		ops = []
		for _ in range(n_terms):
			pstr = "".join(np.random.choice(paulis, n_qubits))
			op = qml.pauli.string_to_pauli_word(pstr)
			ops.append(op)

		# Construct the Hamiltonian
		hamiltonian = qml.Hamiltonian(coeffs, ops)
		hams.append(hamiltonian)

	return hams
