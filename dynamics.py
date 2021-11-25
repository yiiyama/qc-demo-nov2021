import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def tensor_product(ops):
    """Recursively apply np.kron to construct a tensor product of the operators.
    
    Args:
        ops (List): List of (2, 2) arrays.

    Returns:
        np.ndarray(shape=(2 ** nops, 2 ** nops), dtype=np.complex128): Tensor product of ops.
    """
    
    prod = 1.
    for op in ops:
        prod = np.kron(op, prod)

    return prod


def make_hamiltonian(paulis, coeffs=None):
    """Compute the Hamiltonian matrix from its Pauli decomposition.

    Args:
        paulis (List(List(str))): Terms in Pauli decomposition. Form [['i', 'x', 'z', ..], ['x', 'y', 'i', ..], ..]
            All inner lists must be of the same length (number of qubits n).
        coeffs (None or List(float)): If not None, the list must be of the same length as the outer list of paulis
            and should specify the coefficient of each term.

    Returns:
        np.ndarray(shape=(2 ** n, 2 ** n), dtype=np.complex128): The numerical Hamiltonian matrix. The first qubit
        corresponds to the least significant digit.
    """
    
    if len(paulis) == 0:
        return np.array([[0.]], dtype=np.complex128)

    qubit_nums = set(len(term) for term in paulis)
    assert len(qubit_nums) == 1, 'List of paulis must all have the same length.'

    if coeffs is None:
        coeffs = [1.] * len(paulis)

    num_qubits = qubit_nums.pop()

    # Basis matrices
    basis_matrices = {
        'i': np.array([[1., 0.], [0., 1.]], dtype=np.complex128),
        'x': np.array([[0., 1.], [1., 0.]], dtype=np.complex128),
        'y': np.array([[0., -1.j], [1.j, 0.]], dtype=np.complex128),
        'z': np.array([[1., 0.], [0., -1.]], dtype=np.complex128)
    }

    # Start with an empty matrix
    hamiltonian = np.zeros((2 ** num_qubits, 2 ** num_qubits), dtype=np.complex128)

    for iterm, term in enumerate(paulis):
        try:
            ops = list(basis_matrices[op.lower()] for op in term)
        except KeyError as err:
            sys.stderr.write('Invalid operator {} in term {}\n'.format(err.args[0], iterm))
            raise

        hamiltonian += coeffs[iterm] * tensor_product(ops)

    return hamiltonian


def diagonalized_evolution(hamiltonian, initial_state, time, num_steps=100):
    """Diagonalize the given reduced Hamiltonian and evolve the initial state by exp(-i time*hamiltonian).
    
    Args:
        hamiltonian (np.ndarray(shape=(D, D), dtype=np.complex128)): Hamiltonian matrix divided by hbar.
        initial_state (np.ndarray(shape=(D,), dtype=np.complex128)): Initial state vector.
        time (float): Evolution time.
        num_steps (int): Number of steps (T) to divide time into.
        
    Returns:
        np.ndarray(shape=(T,), dtype=float): Time points.
        np.ndarray(shape=(D, T), dtype=np.complex128): State vector as a function of time.
    """

    num_dim = hamiltonian.shape[0]
    num_qubits = np.round(np.log2(num_dim)).astype(int)

    # Create the array of time points
    time_points = np.linspace(0., time, num_steps, endpoint=True)

    ## Diagonalize the Hamiltonian
    eigvals, eigvectors = np.linalg.eigh(hamiltonian)

    ## Decompose the initial state vector into a linear combination of eigenvectors
    # Matrix eigvectors has the form [v_0 v_1 v_2 ..], where v_i . v_j = delta_ij
    # -> eigvectors^dagger @ initial_state = coefficients for the eigenvector decomposition of the initial state vector
    initial_coeff = eigvectors.T.conjugate() @ initial_state
    # Initial state as a matrix [c_0 v_0, c_1 v_1, ...] (shape (D, D))
    initial_state_matrix = eigvectors * initial_coeff

    ## Time-evolve the initial state to each time point
    # Phase at each time point (shape (D, T))
    phase = np.outer(-1.j * eigvals, time_points)
    phase_factor = np.exp(phase)
    statevectors = initial_state_matrix @ phase_factor # shape (D, T)

    return time_points, statevectors


def bit_expectations_sv(time_points, statevectors):
    """Compute the bit expectation values at each time point from statevectors.
    
    Args:
        time_points (np.ndarray(shape=(T,), dtype=float)): Time points.
        statevectors (np.ndarray(shape=(D, T), dtype=np.complex128)): State vector as a function of time.

    Returns:
        np.ndarray(shape=(T, n), dtype=float): Time points tiled for each bit.
        np.ndarray(shape=(T, n), dtype=float): Bit expectation values.
    """

    num_bits = np.round(np.log2(statevectors.shape[0])).astype(int)
    if num_bits > 8:
        raise NotImplementedError('Function not compatible with number of qubits > 8')

    # Probability of seeing each bitstring at each time point
    probs = np.square(np.abs(statevectors)) # shape (D, T)

    # Unpack each index into a binary
    indices = np.expand_dims(np.arange(2 ** num_bits, dtype=np.uint8), axis=1) # shape (D, 1)
    bits = np.unpackbits(indices, axis=1, count=num_bits, bitorder='little').astype(float) # shape (D, num_bits)

    # For each bit, expectation = sum_j [prob_j * bit_j]
    y = probs.T @ bits # shape (T, num_bits)
        
    # Tile the time points to have one x array per spin
    x = np.tile(np.expand_dims(time_points, 1), (1, num_bits)) # shape (T, num_bits)
    
    return x, y


def bit_expectations_counts(time_points, counts_list, num_bits):
    """Compute the bit expectation values from experiment results.

    Args:
        time_points (np.ndarray(shape=(T,), dtype=float)): Time points.
        counts_list (List(Dict)): List (length T) of quantum experiment results, as given by Qiskit job.result().get_counts()
        num_bits (int): Number of qubits

    Returns:
        np.ndarray(shape=(nstep, num_bits), dtype=float): Time points tiled for each bit.
        np.ndarray(shape=(nstep, num_bits), dtype=float): Bit expectation values.
    """

    if num_bits > 8:
        raise NotImplementedError('Function not compatible with number of qubits > 8')

    num_steps = len(counts_list)

    x = np.tile(np.expand_dims(time_points, axis=1), (1, num_bits)) # shape (T, num_bits)
    y = np.zeros_like(x)
    
    for istep, counts in enumerate(counts_list):
        counts = counts_list[istep]

        total = 0
        for bitstring, count in counts.items():
            # 1. reverse the bitstring (last bit is the least significant)
            # 2. map all bits to integers
            # 3. convert to array
            bits = np.array(list(map(int, reversed(bitstring))), dtype=float)
            y[istep] += count * bits
            total += count
        
        y[istep] /= total

    return x, y

    
def plot_spins(counts_list, num_spins, omegadt, hamiltonian=None, initial_state=None, spin_component='z'):
    """Compute the expectation value of the Z(/X/Y) component of each spin in the Heisenberg model from the quantum
    measurement results.

    Args:
        counts_list (List(Dict)): List of quantum experiment results, as given by Qiskit job.result().get_counts()
        num_spins (int): Number of spins in the system.
        omegadt (float): Hamiltonian parameter (H = -0.5 hbar omega sum_j [xx + yy + zz]) times time step.
        hamiltonian (np.ndarray, optional): If provided, plot a theory curve according to the given Hamiltonian.
        initial_state (np.ndarray, optional): Initial state vector. Required when hamiltonian is given.
        spin_component (str, optional): Spin component to plot. Values 'x', 'y', or 'z'. Only affects the theory curve.
    """
    
    # Number of steps (counts_list includes the initial state counts too)
    num_steps = len(counts_list) - 1

    # Figure and axes for the plot
    fig, ax = plt.subplots(1, 1)
    legend_items = []
    legend_labels = []

    if hamiltonian is not None:
        # Compute the statevector as a function of time from Hamiltonian diagonalization
        time_points, statevectors = diagonalized_evolution(-0.5 * hamiltonian, initial_state, omegadt * num_steps)

        spin_basis_change = None
        if spin_component == 'x':
            spin_basis_change = np.array([[1., 1.], [1., -1.]], dtype=np.complex128) * np.sqrt(0.5)
        elif spin_component == 'y':
            spin_basis_change = np.array([[1., -1.j], [-1.j, 1.]], dtype=np.complex128) * np.sqrt(0.5)

        if spin_basis_change is not None:
            basis_change = tensor_product([spin_basis_change] * num_spins)
            statevectors = basis_change @ statevectors
            iniital_state = basis_change @ initial_state

        x, y = bit_expectations_sv(time_points, statevectors)

        # Convert the bit expectations ([0, 1]) to spin expectations ([1, -1])
        y = 1. - 2. * y

        # Plot
        lines = ax.plot(x, y)
        colors = list(line.get_color() for line in lines)
        
        dummy_line = mpl.lines.Line2D([0], [0])
        dummy_line.update_from(lines[0])
        dummy_line.set_color('black')
        legend_items.append(dummy_line)
        legend_labels.append('exact')
    else:
        colors = None

    # Time points
    time_points = np.linspace(0., num_steps * omegadt, num_steps + 1, endpoint=True)
    
    # Compute the bit expectation values from the counts
    x, y = bit_expectations_counts(time_points, counts_list, num_spins)

    # Convert the bit expectations ([0, 1]) to spin expectations ([1, -1])
    y = 1. - 2. * y

    # Plot
    markers = ax.plot(x, y, 'o')
    if colors is not None:
        for marker, color in zip(markers, colors):
            marker.set_color(color)
    
    legend_items += markers
    legend_labels += ['bit%d' % i for i in range(num_spins)]
    ax.legend(legend_items, legend_labels)
    
    ax.set_xlabel(r'$\omega t$')
    ax.set_ylabel(r'$\langle S_z \rangle$')
