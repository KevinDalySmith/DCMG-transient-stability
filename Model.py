
import numpy as np
import requests
from os.path import join
import re
import networkx as nx
from Math import get_branch_params, merge_branches, graph_matrices

MATPOWER_REPO = 'https://raw.githubusercontent.com/MATPOWER/matpower/master/'
MATPOWER_DATA = './data/'
BASE_MVA = 100


class Model:

    def __init__(self, B, C, aij, aji, bij, bji, phiij, phiji, d, p_nom, w_nom):
        """
        Initialize a new instance of the model.
        Active power flows from i into the {i, j} branch are modeled by:
            pij = bij + aij sin(xi - xj - phiij)

        Parameters
        ----------
        B : (n, m) NumPy array
            Signed incidence matrix for the underlying (undirected) graph, with n nodes and m branches.
            This matrix specifies which nodes are the "source" and "sink" of each branch.
        C : (m, s) NumPy array
            Cycle basis matrix for the underlying (undirected) graph.
        aij : (m,) NumPy array
            Sine coefficients for active power flow of each source node into the branch.
        aji : (m,) NumPy array
            Sine coefficients for active power flow of each sink node into the branch.
        bij : (m,) NumPy array
            Intercepts for active power flow of each source node into the branch.
        bji : (m,) NumPy array
            Intercepts for active power flow of each sink node into the branch.
        phiij : (m,) NumPy array
            Sine argument offsets for active power flow of each source node into the branch.
        phiji : (m,) NumPy array
            Sine argument offsets for active power flow of each sink node into the branch.
        d : (n,) NumPy array
            Damping / droop coefficients for each node.
        p_nom : (n,) NumPy array
            Nominal active power injections at each node.
        w_nom : float
            Nominal frequency.
        """

        # Save fundamental parameters
        self.B, self.C = B, C
        self.aij, self.aji, self.bij, self.bji, self.phiij, self.phiji = aij, aji, bij, bji, phiij, phiji
        self.d, self.p_nom = d, p_nom
        self.w_nom = w_nom

        # Compute derived parameters
        self.n, self.m = self.B.shape
        self.source_incidence, self.sink_incidence = np.zeros((self.n, self.m)), np.zeros((self.n, self.m))
        for i in range(self.n):
            self.source_incidence[i, :] = (self.B[i, :] == 1)
            self.sink_incidence[i, :] = (self.B[i, :] == -1)

    def branch_status(self, angles):
        """
        Computes several important properties of each branch, given the angle state:
            * Counterclockwise angle differences
            * Active power inputs / outputs

        Parameters
        ----------
        angles : (n,) NumPy array
            Array of phase angles at each node (in radians).

        Returns
        -------
        y : (m,) NumPy array
            Counterclockwise difference between source node and sink node angles.
        pij : (m,) NumPy array
            Active power flow from source nodes into each branch.
        pji : (m,) NumPy array
            Active power flow from sink nodes into each branch.
        """
        diffs = self.B.T @ angles
        y = np.mod(diffs + np.pi, 2 * np.pi) - np.pi
        pij = self.bij + self.aij * np.sin(y - self.phiij)
        pji = self.bji - self.aji * np.sin(y + self.phiji)
        return y, pij, pji

    def bus_status(self, angles):
        """
        Computes several important properties of each bus (node), given the angle state:
            * Active power injections
            * Frequency

        Parameters
        ----------
        angles : (n,) NumPy array
            Array of phase angles at each node (in radians).

        Returns
        -------
        pi : (n,) NumPy array
            Active power injections at each bus.
        freq : (n,) NumPy array
            Frequencies at each bus.
        """
        _, pij, pji = self.branch_status(angles)
        pi = (self.source_incidence @ pij) + (self.sink_incidence @ pji)
        freq = self.w_nom - (pi - self.p_nom) / self.d
        return pi, freq

    def state_status(self, angles):
        """
        Computes several important properties of an initial condition:
            * Initial max frequency deviation (delta0)
            * Initial angle difference magnitudes (gamma0)
            * Initial winding vector (u0)

        Parameters
        ----------
        angles : (n,) NumPy array
            Array of phase angles at each node (in radians).

        Returns
        -------
        delta0 : float
            Max frequency deviation.
        gamma0 : (m,) NumPy array
            Angle difference magnitudes.
        u0 : (c,) NumPy array
            Winding vector.
        """
        y, _, _ = self.branch_status(angles)
        pi, freqs = self.bus_status(angles)
        delta0 = np.max(np.abs(freqs - self.w_nom))
        gamma0 = np.abs(y)
        u0 = np.rint(self.C.T @ y)
        return delta0, gamma0, u0

    def cut_branches(self, branch_idx):
        """
        Create a new model in which the specified branches are removed.
        Raises a ValueError if the branch removal results in a disconnected topology.
        Note: this method does not modify self.

        Parameters
        ----------
        branch_idx : sequence of ints
            Indices of branches to remove.

        Returns
        -------
        new_model : Model
            Model that is identical to self, except the specified branches are removed.
        """

        # Create the new model
        cycle_idx = np.where(np.any(self.C[branch_idx, :] != 0, axis=0))
        B = np.delete(self.B, obj=branch_idx, axis=1)
        C = np.delete(np.delete(self.C, obj=cycle_idx, axis=1), obj=branch_idx, axis=0)
        aij = np.delete(self.aij, obj=branch_idx)
        aji = np.delete(self.aji, obj=branch_idx)
        bij = np.delete(self.bij, obj=branch_idx)
        bji = np.delete(self.bji, obj=branch_idx)
        phiij = np.delete(self.phiij, obj=branch_idx)
        phiji = np.delete(self.phiji, obj=branch_idx)
        new_model = Model(B=B, C=C, aij=aij, aji=aji, bij=bij, bji=bji, phiij=phiij, phiji=phiji,
                          d=np.copy(self.d), p_nom=np.copy(self.p_nom), w_nom=self.w_nom)

        # Check if the network is disconnected
        L = B @ B.T
        A = -L + np.diag(np.diag(L))
        G = nx.from_numpy_matrix(A)
        if nx.number_connected_components(G) > 1:
            raise ValueError('Network is disconnected')

        return new_model

    @staticmethod
    def from_test_case(name, d, w_nom=60):
        """
        Retrieve a test case graph from the MATPOWER GitHub.
        Note: This method is in beta. It does not work on all cases in the MATPOWER repo.

        Paramters
        ---------
        name : str
            Name of the test case. E.g., 'case39' for the New England system.
        d : (n,) NumPy array
            Droop coefficeints at each node.
        w_nom : float
            Nominal frequency.

        Returns
        -------
        Model with parameters based on the nominal values of the test case.
        """

        # Download content
        url = join(MATPOWER_REPO, MATPOWER_DATA, name + '.m')
        res = requests.get(url)

        # Extract bus data
        bus_regex = 'bus = \[(?s)(.*?)\];'
        bus_table = re.findall(bus_regex, res.text)
        bus_table = re.sub(';.*?\n', ';', bus_table[0])
        bus_data = np.array([list(map(float, s.split())) for s in bus_table.split(';')[:-1]])
        p_demands, volt_mag = bus_data[:, 2], bus_data[:, 7]

        # Extract generator data
        gen_regex = 'gen = \[(?s)(.*?)\];'
        gen_table = re.findall(gen_regex, res.text)
        gen_table = re.sub(';.*?\n', ';', gen_table[0])
        gen_data = np.array([list(map(float, s.split())) for s in gen_table.split(';')[:-1]])
        gen_bus, gen_inj, gen_vm = np.array(gen_data[:, 0], dtype=int), gen_data[:, 1], gen_data[:, 5]

        # Create nominal injection vector and override bus voltages with generator voltages
        p_nom = -p_demands
        for i in range(len(gen_bus)):
            p_nom[gen_bus[i] - 1] += gen_inj[i]
            volt_mag[gen_bus[i] - 1] = gen_vm[i]
        p_nom /= BASE_MVA

        # Extract branch data
        branch_regex = 'branch = \[(?s)(.*?)\];'
        branch_table = re.findall(branch_regex, res.text)
        branch_table = re.sub(';.*?\n', ';', branch_table[0])
        branch_data = np.array([list(map(float, s.split())) for s in branch_table.split(';')[:-1]])
        sources, sinks = branch_data[:, 0].astype(dtype=np.int) - 1, branch_data[:, 1].astype(dtype=np.int) - 1
        r, x, ratio = branch_data[:, 2], branch_data[:, 3], branch_data[:, 8]

        # Process branches, merging multiple branches between the same pairs of nodes
        G = nx.DiGraph()
        for e in range(len(r)):
            i, j = sources[e], sinks[e]
            aij, aji, bij, bji, phiij, phiji = get_branch_params(r[e], x[e], ratio[e], volt_mag[i], volt_mag[j])
            if G.has_edge(i, j):
                G[i][j]['aij'], G[i][j]['bij'], G[i][j]['phiij'] = merge_branches(
                    G[i][j]['aij'], G[i][j]['bij'], G[i][j]['phiij'], aij, bij, phiij)
                G[i][j]['aji'], G[i][j]['bji'], G[i][j]['phiji'] = merge_branches(
                    G[i][j]['aji'], G[i][j]['bji'], G[i][j]['phiji'], aji, bji, phiji)
            elif G.has_edge(j, i):
                G[j][i]['aij'], G[j][i]['bij'], G[j][i]['phiij'] = merge_branches(
                    G[j][i]['aij'], G[j][i]['bij'], G[j][i]['phiij'], aji, bji, phiji)
                G[j][i]['aji'], G[j][i]['bji'], G[j][i]['phiji'] = merge_branches(
                    G[j][i]['aji'], G[j][i]['bji'], G[j][i]['phiji'], aij, bij, phiij)
            else:
                G.add_edge(i, j, aij=aij, aji=aji, bij=bij, bji=bji, phiij=phiij, phiji=phiji)

        # Compute graph matrices
        G_symm = nx.Graph()
        for u, v, data in G.edges.data():
            G_symm.add_edge(u, v, **data)
        B, C = graph_matrices(G_symm)

        # Get edge parameter vectors (with edges in the same order as the indexing of B, C)
        # In Python 3.6, the order in which edges are iterated is always the same.
        m = B.shape[1]
        aij, aji, bij, bji = np.empty(m, ), np.empty(m, ), np.empty(m, ), np.empty(m, )
        phiij, phiji = np.empty(m, ), np.empty(m, )
        arrs = [aij, aji, bij, bji, phiij, phiji]
        keys = ['aij', 'aji', 'bij', 'bji', 'phiij', 'phiji']
        nodes = []
        for e, (u, v) in enumerate(G_symm.edges):
            nodes.append((u + 1, v + 1))
            for k in range(len(arrs)):
                arrs[k][e] = G_symm[u][v][keys[k]]

        # Create model
        model = Model(B, C, aij, aji, bij, bji, phiij, phiji, d, p_nom, w_nom)
        model.nodes = nodes
        return model
