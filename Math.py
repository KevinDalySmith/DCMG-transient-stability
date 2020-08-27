
import networkx as nx
import numpy as np


def graph_matrices(G: nx.Graph):
    """
    Compute the incidence matrix and cycle matrix of an undirected graph.

    Parameters
    ----------
    G : NetworkX Graph
        Undirected graph with n nodes and m edges. Weights are ignored.
        Assumes that nodes are {1, 2, ..., n}.

    Returns
    -------
    B : (n, m) NumPy array
        Incidence matrix, assigning an arbitrary orientation to each edge.
    C : (s, m) NumPy array
        Cycle basis matrix. Each row corresponds to one of the s basis cycles.
    """

    # Compute the incidence matrix
    n, m = G.number_of_nodes(), G.number_of_edges()
    B = np.zeros((n, m), dtype=int)
    for e, (u, v) in enumerate(G.edges):
        B[u, e], B[v, e] = 1, -1

    # Compute cycle basis matrix
    basis = nx.cycle_basis(G)
    C = np.zeros((m, len(basis)), dtype=int)
    for cycle_idx, cycle in enumerate(basis):

        # Frustratingly, minimum_cycle_basis does not list the cycle nodes in order, so we have to
        # re-construct the order of these nodes here.
        subgraph = G.subgraph(cycle)
        source = cycle[0]
        target = next(nx.neighbors(subgraph, source))
        all_paths = nx.all_simple_paths(subgraph, source, target)
        true_cycle = max(all_paths, key=len) + [source]

        # Get edges between these nodes
        for i in range(len(true_cycle) - 1):
            source, target = true_cycle[i], true_cycle[i + 1]
            edge_vector = np.zeros(n, dtype=np.int)
            edge_vector[source], edge_vector[target] = 1, -1
            edge_score = np.matmul(B.T, edge_vector)
            edge_idx = np.argmax(np.abs(edge_score))
            C[edge_idx, cycle_idx] = np.sign(edge_score[edge_idx])

    return B, C


def get_branch_params(r, x, ratio, Ei, Ej):
    """
    Parameterizes the sine model for active power flow across a transmission line.

    Parameters
    ----------
    r : float
        Series resistance.
    x : float
        Series reactance.
    ratio : float
        Tap ratio (zero if there is no transformer).
    Ei : float
        Voltage magnitude at the source.
    Ej : float
        Voltage magnitude at the sink.

    Returns
    -------
    aij : float
    aji : float
    bij : float
    bji : float
    phiij : float
    phiji : float
    """
    is_transformer = (0 < ratio) & (ratio < 999)
    y_mag = 1.0 / np.sqrt(np.square(r) + np.square(x))
    y_angle = -np.arctan(x / r) if r > 0 else -np.pi / 2
    aij = Ei * Ej * y_mag
    bij = np.square(Ei) * y_mag * np.cos(y_angle)
    bji = np.square(Ej) * y_mag * np.cos(y_angle)
    if is_transformer:
        aij *= ratio
        bij *= np.square(ratio)
    phiij = y_angle + np.pi / 2
    return aij, aij, bij, bji, phiij, phiij


def merge_branches(aij_1, bij_1, phiij_1, aij_2, bij_2, phiij_2):
    """
    Find parameters of a single branch with identical active power flow to the sum of two branches.
    I.e., find aij_3, bij_3, and phiij_3 such that:
        bij_3 + aij_3 * sin(y - phiij_1) = bij_1 + aij_1 * sin(y - phiij_1) + bij_2 + aij_2 * sin(y - phiij_2)

    Parameters
    ----------
    aij_1 : float
    bij_1 : float
    phiij_1 : float
    aij_2 : float
    bij_2 : float
    phiij_2 : float

    Returns
    -------
    aij_3 : float
    bij_3 : float
    phiij_3 : float
    """
    aij_3 = np.sqrt(np.square(aij_1) + np.square(aij_2) + 2 * aij_1 * aij_2 * np.cos(phiij_1 - phiij_2))
    bij_3 = bij_1 + bij_2
    num = aij_1 * np.sin(phiij_1) + aij_1 * np.sin(phiij_1)
    denom = aij_1 * np.cos(phiij_1) + aij_2 * np.cos(phiij_2)
    phiij_3 = np.arctan2(num, denom)
    return aij_3, bij_3, phiij_3


def latin_hypercube_sample(n_pts, lower, upper):
    """
    Randomly chooses points from a hypercube, using Latin hypercube sampling.

    Parameters
    ----------
    n_pts : int
        Number of points to sample.
    lower : (d,) NumPy array
        Array of lower bounds on each dimension.
    upper : (d,) NumPy array
        Array of upper bounds on each dimension.

    Returns
    -------
    X : (d, n_pts) NumPy array
        Array of sample points.
    """

    d = len(upper)
    bin_widths = (upper - lower) / n_pts
    bin_left_edges = np.c_[lower] + np.c_[bin_widths] * np.r_[np.arange(0, n_pts)]
    bin_samples = bin_left_edges + np.c_[bin_widths] * np.random.rand(d, n_pts)
    bin_samples_tr = np.transpose(bin_samples)
    np.random.shuffle(bin_samples_tr)
    return bin_samples_tr.transpose()


def sine_bounds(gamma, phi):
    """
    Compute a parallelogram to bound the graph of sin(y - phi) on the interval [-gamma, gamma].
    Returns parameters for the constraint a * y + b * eta <= c.

    Parameters
    ----------
    gamma : float
        Radius of the domain, in the range (0, pi/2 - |phi|).
    phi : float
        Argument offset, in the range (-pi/2, pi/2).

    Returns
    -------
    a : (2,) NumPy array
    b : (2,) NumPy array
    c : (2,) NumPy array
    """

    # Transform sine function to have zero-valued endpoints
    m = (np.sin(gamma - phi) + np.sin(gamma + phi)) / (2 * gamma)
    f = lambda y: np.sin(y - phi) - m * (y + gamma) + np.sin(gamma + phi)

    # Find min / max of transformed function
    crit_pts = phi + np.array([np.arccos(m), -np.arccos(m)])
    crit_pts = crit_pts[np.abs(crit_pts) < gamma]
    fvals = f(crit_pts)
    fmin = np.minimum(0, np.min(fvals))
    fmax = np.maximum(0, np.max(fvals))

    # Compute the hyperplanes
    a = np.array([m, -m])
    b = np.array([-1, 1])
    c = np.array([-fmin - m * gamma + np.sin(gamma + phi),
                  fmax + m * gamma - np.sin(gamma + phi)])

    return a, b, c
