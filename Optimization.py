
import numpy as np
import gurobipy as gb
from gurobipy import GRB
from Math import sine_bounds

# Gurobi settings
EPSILON = 1e-5
gb.setParam('OutputFlag', False)


class FreqDevOptimizer:

    def __init__(self, model, n_pieces=1, ignore_outward=False):
        """
        Create a new optimizer for a particular model instance.

        Parameters
        ----------
        model : Model
        n_pieces : int
            Number of pieces to use to approximate the sine function.
        ignore_outward : bool
            Flag for whether to ignore the "outward-pointing" constraint for the boundary.
        """
        self.model = model
        self.n_pieces = n_pieces
        self.ignore_outward = ignore_outward
        self.temp_obj = []
        self._setup_gurobi_model()

    def __call__(self, gamma, u=None):
        """
        Compute a lower bound on the min-max frequency deviation.

        Parameters
        ----------
        gamma : (m,) NumPy array
            Vector of arc lengths defining the phase-cohesive set.
            Must be in the range 0 < gamma[e] < pi/2 - max(|phiij|, |phiji|).
        u : (optional) (c,) NumPy array
            Winding vector.

        Returns
        -------
        V : float
            Lower bound on the min-max frequency deviation.
        """

        # Clear previous temporary constraints
        for obj in self.temp_obj:
            self.gb_model.remove(obj)
        self.temp_obj = []

        # Set gamma bounds
        for e in range(self.model.m):
            self.y[e].lb = -gamma[e]
            self.y[e].ub = gamma[e]

        # Enforce active boundary
        for e in range(self.model.m):
            self.temp_obj.append(
                self.gb_model.addGenConstrIndicator(self.z_plus[e], True, self.y[e], '=', gamma[e]))
            self.temp_obj.append(
                self.gb_model.addGenConstrIndicator(self.z_minus[e], True, self.y[e], '=', -gamma[e]))

        # Constrain eta
        if self.n_pieces == 1:
            self._eta_polytope_approx(gamma)

        # Add winding cell constraint
        if u is not None:
            winding_violation = np.dot(self.model.C.T, self.y) - 2 * np.pi * u
            for c in range(self.model.C.shape[1]):
                self.temp_obj.append(
                    self.gb_model.addConstr(winding_violation[c] == 0))

        # Optimize
        self.gb_model.optimize()
        code = self.gb_model.getAttr('Status')
        if code in [2, 15]:
            return self.gb_model.getAttr('ObjVal')
        else:
            return np.inf

    def get_variables(self):
        """
        Get the current values of all the decision variables.
        Note: This will fail unless the optimizer has been called at least once.

        Returns
        -------
        f : (n,) NumPy array
            Estimate of velocity vectors at each bus.
        y : (m,) NumPy array
            Vector of ccw angle differences across each branch.
        etaij: (m,) NumPy array
            Estimates of sin(y[e] - phiij[e]) for each branch.
        etaji: (m,) NumPy array
            Estimates of -sin(y[e] + phiji[e]) for each branch.
        pij : (m,) NumPy array
            Active power injections from source buses into each branch.
        pji : (m,) NumPy array
            Active power injections from target buses into each branch.
        z_plus : (m,) NumPy array
            Binary indicators for the m "positive" faces of the phase-cohesive set.
        z_minus : (m,) NumPy array
            Binary indicators for the m "negative" faces of the phase-cohesive set.
        """
        f_vals = np.array([f.X for f in self.f])
        y_vals = np.array([y.X for y in self.y])
        etaij_vals = np.array([e.X for e in self.etaij])
        etaji_vals = np.array([e.X for e in self.etaji])
        pij_vals = np.array([p.X for p in self.pij])
        pji_vals = np.array([p.X for p in self.pji])
        z_plus_vals = np.array([z.X for z in self.z_plus])
        z_minus_vals = np.array([z.X for z in self.z_minus])
        return f_vals, y_vals, etaij_vals, etaji_vals, pij_vals, pji_vals, z_plus_vals, z_minus_vals

    def _setup_gurobi_model(self):
        """
        Set up the internal Gurobi model.
        This should only be called by the constructor.
        """

        # Create the Gurobi model
        self.gb_model = gb.Model()
        self.gb_model.setParam('IntFeasTol', EPSILON)
        self.gb_model.setParam('FeasibilityTol', EPSILON)
        self.gb_model.setParam('BestObjStop', EPSILON)

        self.max_freq_dev = self.gb_model.addVar(vtype=GRB.CONTINUOUS, lb=-np.inf, ub=np.inf)
        self.gb_model.setObjective(self.max_freq_dev, GRB.MINIMIZE)

        # Add branch variables
        self.y = np.empty(self.model.m, dtype=gb.Var)        # Counterclockwise angle differences
        self.etaij = np.empty(self.model.m, dtype=gb.Var)    # Approximates sin(yij - phiij)
        self.etaji = np.empty(self.model.m, dtype=gb.Var)    # Approximates -sin(yij + phiji)
        self.pij = np.empty(self.model.m, dtype=gb.Var)      # Active power flow from source to target
        self.pji = np.empty(self.model.m, dtype=gb.Var)      # Active power flow from target to source
        self.z_plus = np.empty(self.model.m, dtype=gb.Var)   # Indicators for faces of phase cohesive set
        self.z_minus = np.empty(self.model.m, dtype=gb.Var)  # Indicators for faces of phase cohesive set
        for e in range(self.model.m):
            self.y[e] = self.gb_model.addVar(lb=-np.inf, ub=np.inf, vtype=GRB.CONTINUOUS)
            self.etaij[e] = self.gb_model.addVar(lb=-np.inf, ub=np.inf, vtype=GRB.CONTINUOUS)
            self.etaji[e] = self.gb_model.addVar(lb=-np.inf, ub=np.inf, vtype=GRB.CONTINUOUS)
            self.pij[e] = self.gb_model.addVar(lb=-np.inf, ub=np.inf, vtype=GRB.CONTINUOUS)
            self.pji[e] = self.gb_model.addVar(lb=-np.inf, ub=np.inf, vtype=GRB.CONTINUOUS)
            self.z_plus[e] = self.gb_model.addVar(vtype=GRB.BINARY)
            self.z_minus[e] = self.gb_model.addVar(vtype=GRB.BINARY)

        # Add bus variables
        self.f = np.empty(self.model.n, dtype=gb.Var)  # Velocity vector f(\theta)
        for i in range(self.model.n):
            self.f[i] = self.gb_model.addVar(lb=-np.inf, ub=np.inf, vtype=GRB.CONTINUOUS)

        # Define max frequency deviation
        for i in range(self.model.n):
            self.gb_model.addConstr(self.max_freq_dev >= (self.f[i] / self.model.d[i]) - self.model.w_nom)
            self.gb_model.addConstr(self.max_freq_dev >= -(self.f[i] / self.model.d[i]) + self.model.w_nom)

        # Define active power flows
        for e in range(self.model.m):
            self.gb_model.addConstr(self.pij[e] == self.model.bij[e] + self.model.aij[e] * self.etaij[e])
            self.gb_model.addConstr(self.pji[e] == self.model.bji[e] + self.model.aji[e] * self.etaji[e])

        # Define f
        p = self.model.p_nom + self.model.d * self.model.w_nom
        source_outflows = np.dot(self.model.source_incidence, self.pij)
        sink_outflows = np.dot(self.model.sink_incidence, self.pji)
        for i in range(self.model.n):
            self.gb_model.addConstr(self.f[i] == p[i] - source_outflows[i] - sink_outflows[i])

        # Constrain eta
        if self.n_pieces >= 2:
            self._eta_pwl_approx(self.n_pieces)

        # Define v
        if not self.ignore_outward:
            for e in range(self.model.m):
                source = np.flatnonzero(self.model.B[:, e] == 1)[0]
                target = np.flatnonzero(self.model.B[:, e] == -1)[0]
                direction = self.f[source] / self.model.d[source] - self.f[target] / self.model.d[target]
                self.gb_model.addGenConstrIndicator(self.z_plus[e], True, direction, '>', 0)
                self.gb_model.addGenConstrIndicator(self.z_minus[e], True, direction, '<', 0)

        # Activate boundary
        self.gb_model.addConstr(gb.quicksum(self.z_plus) + gb.quicksum(self.z_minus) == 1)

    def _eta_polytope_approx(self, gamma):
        """
        Approximate the constraints etaij[e] = sin(y[e] - phiij[e]) and etaji[e] = -sin(y[e] + phiji[e])
        using a bounding parallelogram.
        This should only be called by __call__.

        Parameters
        ----------
        gamma : (m,) NumPy array
            Vector of arc lengths defining the phase-cohesive set.
            Must be in the range 0 < gamma[e] < pi/2 - max(|phiij|, |phiji|).
        """

        for e in range(self.model.m):

            # Special case: sines are anti-symmetric
            if np.abs(self.model.phiij[e]) < EPSILON and np.abs(self.model.phiji[e]) < EPSILON:
                self.temp_obj.append(
                    self.gb_model.addConstr(self.etaij[e] + self.etaji[e] == 0))

            # Define etaij
            a, b, c = sine_bounds(gamma[e], self.model.phiij[e])
            for k in range(len(a)):
                self.temp_obj.append(
                    self.gb_model.addConstr(a[k] * self.y[e] + b[k] * self.etaij[e] <= c[k]))

            # Define etaji
            a, b, c = sine_bounds(gamma[e], -self.model.phiji[e])
            for k in range(len(a)):
                self.temp_obj.append(
                    self.gb_model.addConstr(a[k] * self.y[e] - b[k] * self.etaji[e] <= c[k]))

    def _eta_pwl_approx(self, n_pieces=2):
        """
        Approximate the constraints etaij[e] = sin(y[e] - phiij[e]) and etaji[e] = -sin(y[e] + phiji[e])
        using Gurobi's built-in PWL estimate to establish a lower and upper bound.
        This should only be called by _setup_gurobi_model.

        Parameters
        ----------
        n_pieces : int
            Number of pieces to approximate the sine lower and upper bounds, at least 2.
        """

        for e in range(self.model.m):

            # Define y - phiij and -y - phiji arguments
            argij = self.gb_model.addVar(lb=-np.inf, ub=np.inf, vtype=GRB.CONTINUOUS)
            argji = self.gb_model.addVar(lb=-np.inf, ub=np.inf, vtype=GRB.CONTINUOUS)
            argij_constr = self.gb_model.addConstr(argij == self.y[e] - self.model.phiij[e])
            argji_constr = self.gb_model.addConstr(argji == -self.y[e] - self.model.phiji[e])

            # Constrain etaij
            etaij_ubound = self.gb_model.addVar(lb=-np.inf, ub=np.inf, vtype=GRB.CONTINUOUS)
            etaij_lbound = self.gb_model.addVar(lb=-np.inf, ub=np.inf, vtype=GRB.CONTINUOUS)
            etaij_u_constr = self.gb_model.addGenConstrSin(argij, etaij_ubound,
                                                           options='FuncPieceRatio=1 FuncPieces={}'.format(n_pieces))
            etaij_l_constr = self.gb_model.addGenConstrSin(argij, etaij_lbound,
                                                           options='FuncPieceRatio=0 FuncPieces={}'.format(n_pieces))
            etaij_ub = self.gb_model.addConstr(etaij_lbound <= self.etaij[e])
            etaij_lb = self.gb_model.addConstr(self.etaij[e] <= etaij_ubound)

            # Constrain etaji
            etaji_ubound = self.gb_model.addVar(lb=-np.inf, ub=np.inf, vtype=GRB.CONTINUOUS)
            etaji_lbound = self.gb_model.addVar(lb=-np.inf, ub=np.inf, vtype=GRB.CONTINUOUS)
            etaji_u_constr = self.gb_model.addGenConstrSin(argji, etaji_ubound,
                                                           options='FuncPieceRatio=1 FuncPieces={}'.format(n_pieces))
            etaji_l_constr = self.gb_model.addGenConstrSin(argji, etaji_lbound,
                                                           options='FuncPieceRatio=0 FuncPieces={}'.format(n_pieces))
            etaji_ub = self.gb_model.addConstr(etaji_lbound <= self.etaji[e])
            etaji_lb = self.gb_model.addConstr(self.etaji[e] <= etaji_ubound)

            # # Save temporary vars / constraints
            # self.temp_obj.extend([
            #     argij, argji, argij_constr, argji_constr,
            #     etaij_ubound, etaij_lbound, etaij_u_constr, etaij_l_constr, etaij_ub, etaij_lb,
            #     etaji_ubound, etaji_lbound, etaji_u_constr, etaji_l_constr, etaji_ub, etaji_lb
            # ])

    # def check_consistency(self):
    #
    #     f, y, etaij, etaji, pij, pji, z_plus, z_minus = self.get_variables()
    #
    #     # Check error of sine bound
    #     etaij_real = np.sin(etaij - self.model.phiij)
    #     etaij_err = etaij_real - etaij
    #     print('etaij approx. error: avg {}, max {}'.format(
    #         np.linalg.norm(etaij_err, 1) / self.model.m, np.linalg.norm(etaij_err, np.inf)))
    #     etaji_real = np.sin(etaji - self.model.phiji)
    #     etaji_err = etaji_real - etaji
    #     print('etaji approx. error: avg {}, max {}'.format(
    #         np.linalg.norm(etaji_err, 1) / self.model.m, np.linalg.norm(etaji_err, np.inf)))
    #
    #     # Check the error of active power injections
    #     pij_real = self.model.bij + self.model.aij * etaij_real
    #     pij_err = pij_real - pij
    #     # print('pij approx. error: avg {}, max {}'.format(
    #     #     np.linalg.norm(pij_err, 1) / self.model.m, np.linalg.norm(pij_err, np.inf)))
    #     pji_real = self.model.bji + self.model.aji * etaji_real
    #     pji_err = pji_real - pji
    #     # print('pji approx. error: avg {}, max {}'.format(
    #     #     np.linalg.norm(pji_err, 1) / self.model.m, np.linalg.norm(pji_err, np.inf)))
    #
    #     # Check the error of frequencies
    #     pi = self.model.source_incidence @ pij_real + self.model.sink_incidence @ pji_real
    #     freqs = f / self.model.d
    #     freqs_real = self.model.w_nom - (self.model.p_nom - pi) / self.model.d
    #     freqs_err = freqs_real - freqs
    #     print(f)
    #     print(freqs)
    #     print(freqs_real)
    #     print('frequency approx. error: avg {}, max {}'.format(
    #         np.linalg.norm(freqs_err, 1) / self.model.m, np.linalg.norm(freqs_err, np.inf)))


class VoltLossOptimizer:

    def __init__(self, model, theta0):

        self.model = model
        self.theta0 = theta0
        self.temp_obj = []

        y, _, _ = model.branch_status(theta0)
        self.etaij = np.sin(y - model.phiij)
        self.etaji = -np.sin(y + model.phiji)

        self._setup_gurobi_model()

    def __call__(self, gamma, bus_idx, tol):

        # Clear previous temporary constraints
        for obj in self.temp_obj:
            self.gb_model.remove(obj)
        self.temp_obj = []

        # Bound sine functions
        etaij_bound = np.maximum(np.sin(gamma - self.model.phiij), np.sin(gamma + self.model.phiij))
        etaji_bound = np.maximum(np.sin(gamma - self.model.phiji), np.sin(gamma + self.model.phiji))

        # Define t2
        c2 = np.dot(self.model.source_incidence, self.delta_bij + etaij_bound * self.delta_aij) + \
             np.dot(self.model.sink_incidence, self.delta_bji + etaji_bound * self.delta_aji)
        c2 = c2 / self.model.d
        for i in range(self.model.n):
            self.temp_obj.append(self.gb_model.addConstr(self.t2 >= -c2[i]))

        # Define voltage perturbations
        for e in range(self.model.m):

            bus_is_source = self.model.B[bus_idx, e] == 1
            bus_is_sink = self.model.B[bus_idx, e] == -1

            # Set aij perturbations
            if bus_is_source or bus_is_sink:
                self.temp_obj.append(
                    self.gb_model.addConstr(self.delta_aij[e] == -self.model.aij[e] * self.alpha))
                self.temp_obj.append(
                    self.gb_model.addConstr(self.delta_aji[e] == -self.model.aji[e] * self.alpha))
            else:
                self.temp_obj.append(self.gb_model.addConstr(self.delta_aij[e] == 0))
                self.temp_obj.append(self.gb_model.addConstr(self.delta_aji[e] == 0))

            # Set bij perturbations
            if bus_is_source:
                lhs = self.delta_bij[e] + 2 * self.model.bij[e] * self.alpha
                rhs = self.model.bij[e] * self.alpha * self.alpha
                self.temp_obj.append(self.gb_model.addQConstr(lhs <= rhs))
                self.temp_obj.append(self.gb_model.addConstr(self.delta_bji[e] == 0))
            elif bus_is_sink:
                lhs = self.delta_bji[e] + 2 * self.model.bji[e] * self.alpha
                rhs = self.model.bji[e] * self.alpha * self.alpha
                self.temp_obj.append(self.gb_model.addQConstr(lhs <= rhs))
                self.temp_obj.append(self.gb_model.addConstr(self.delta_bij[e] == 0))
            else:
                self.temp_obj.append(self.gb_model.addConstr(self.delta_bij[e] == 0))
                self.temp_obj.append(self.gb_model.addConstr(self.delta_bji[e] == 0))

        self.temp_obj.append(self.gb_model.addConstr(self.t1 + self.t2 <= tol))

        # Optimize
        self.gb_model.optimize()
        code = self.gb_model.getAttr('Status')
        if code in [2, 15]:
            return self.gb_model.getAttr('ObjVal'), self.t1.X, self.t2.X
        else:
            raise ValueError('Optimization failed!')

    def _setup_gurobi_model(self):

        # Create the Gurobi model
        self.gb_model = gb.Model()
        self.gb_model.setParam('NonConvex', 2)
        self.alpha = self.gb_model.addVar(vtype=GRB.CONTINUOUS, lb=-np.inf, ub=np.inf)
        self.gb_model.setObjective(self.alpha, GRB.MAXIMIZE)

        # Add perturbation variables
        self.delta_aij = np.empty(self.model.m, dtype=gb.Var)
        self.delta_aji = np.empty(self.model.m, dtype=gb.Var)
        self.delta_bij = np.empty(self.model.m, dtype=gb.Var)
        self.delta_bji = np.empty(self.model.m, dtype=gb.Var)
        for i in range(self.model.m):
            self.delta_aij[i] = self.gb_model.addVar(lb=-np.inf, ub=np.inf, vtype=GRB.CONTINUOUS)
            self.delta_aji[i] = self.gb_model.addVar(lb=-np.inf, ub=np.inf, vtype=GRB.CONTINUOUS)
            self.delta_bij[i] = self.gb_model.addVar(lb=-np.inf, ub=np.inf, vtype=GRB.CONTINUOUS)
            self.delta_bji[i] = self.gb_model.addVar(lb=-np.inf, ub=np.inf, vtype=GRB.CONTINUOUS)

        # Define optimum
        self.t1 = self.gb_model.addVar(vtype=GRB.CONTINUOUS, lb=-np.inf, ub=np.inf)
        self.t2 = self.gb_model.addVar(vtype=GRB.CONTINUOUS, lb=-np.inf, ub=np.inf)
        c1 = np.dot(self.model.source_incidence, self.delta_bij + self.etaij * self.delta_aij) + \
             np.dot(self.model.source_incidence, self.delta_bji + self.etaji * self.delta_aji)
        c1 = c1 / self.model.d
        for i in range(self.model.n):
            self.gb_model.addConstr(self.t1 >= c1[i])
            self.gb_model.addConstr(self.t1 >= -c1[i])
