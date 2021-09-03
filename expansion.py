"""
Implements first-order and second-order recursive expansion.

"""
import numpy as np
import autograd.numpy as anp
from lin_quad import LinQuadVar
from lin_quad_util import E, cal_E_ww, matmul, concat, simulate,\
                                next_period, kron_prod
from elasticity import exposure_elasticity, price_elasticity
from utilities import mat, vec, sym, gschur
from derivatives import compute_derivatives


def recursive_expansion(eq_cond, ss, var_shape, γ,
                        second_order=True, args=(), tol=1e-12, max_iter=100):
    r"""
    This function solves a system with recursive utility via small-noise
        expansion, given a set of equilibrium conditions, steady states,
        and other model configurations. In particular, it treats the risk
        aversion γ as a function of the perturbation parameter.

    The solver returns a class storing solved variables and diagnostic tools.
    In particular, it stores the solved variables represented by a linear or
    linear-quadratic function of the first- and/or second-order derivatives
    of states. It also stores laws of motion for these state derivatives.

    Parameters
    ----------
    eq_cond : callable
        Returns [log_M0,f1,f2] where log_M0 is the logarithm of the change
        of measure M evaluated at γ = 0, f1 satisfy the forward-looking
        equations E[Mf1]=0, and f2 satisfy the state equations f2=0.

        ``eq_cond(X_t, X_tp1, W_tp1, q, *args) -> (n, ) ndarray``

        where X_t and X_tp1 are variables at time t and t+1 respectively,
        W_tp1 are time t+1 shocks, and q is perturbation parameter.
        Note that in X, state variables must follow endogenous variables.
    ss : (n, ) ndarray or callable
        Steady states or the funciton for calculating steady states.
    args : tuple of floats/ints
        Additional parameters to be passed to f, log_M0, log_V_growth
        (and ss if ss is callable).
    var_shape : tuple of ints
        (n_Y, n_Z, n_W). Number of endogenous variables, states and
        shocks respectively.
    γ : float
        Risk aversion parameter in recursive utility.
    second_order : bool
        If True it will perform second-order small-noise expansion; if False it
        will perform first-order small-noise expansion.
    tol : float
        The toleration for M1 iteration in the second-order expansion.
    max_iter : int
        The maximum of iterations for the M1 calculation in the second-order
        expansion.

    Returns
    -------
    res : ModelSolution
        The model solution represented as a OptimizeResult object. Important
        attributes are: X_t the approximated variables as a linear or linear-
        quadratic function of state derivatives; Z1_tp1 (and Z2_tp1) the laws of
        motion for the first-order (and second-order) derivatives of states.
        Important methods are: simulate() that implements simulation for all
        variables; elasticities() that computes exposure and price elasticities
        for all variables.

    """
    df, ss = _take_derivatives(eq_cond, ss, var_shape,
                                   second_order, args)
    res = _first_order_expansion(df, ss, var_shape, γ)
    if second_order:
        res = _second_order_expansion(res, df, ss, var_shape, γ, tol, max_iter)
    return res


def _take_derivatives(f, ss, var_shape, second_order,
                      args):
    """
    Take first- or second-order derivatives.

    """
    _, _, n_W = var_shape

    W_0 = np.zeros(n_W)
    q_0 = 0.

    if callable(ss):
        ss = ss(*args)

    df = compute_derivatives(f=lambda X_t, X_tp1, W_tp1, q:
                             anp.atleast_1d(f(X_t, X_tp1, W_tp1, q, *args)),
                             X=[ss, ss, W_0, q_0],
                             second_order=second_order)
    return df, ss


def _first_order_expansion(df, ss, var_shape, γ):
    """
    Implements first-order expansion.

    """
    n_Y, n_Z, n_W = var_shape
    n_X = n_Y + n_Z
    # Step 1: Set γ = 1 and solve for N
    Λp, Λ, a, b, Q, Z = gschur(-df['xtp1'], df['xt'])

    Z21 = Z.T[-n_Y:, :n_Y]
    Z22 = Z.T[-n_Y:, n_Y:]

    N = -np.linalg.solve(Z21, Z22)
    N_block = np.block([[N], [np.eye(n_Z)]])

    # Step 2: Reset γ and solve for ψ_tilde_x, ψ_tilde_w, D
    f_1_xtp1 = df['xtp1'][:n_Y]
    f_1_xt = df['xt'][:n_Y]
    f_1_wtp1 = df['wtp1'][:n_Y]
    f_2_xtp1 = df['xtp1'][n_Y:]
    f_2_xt = df['xt'][n_Y:]
    f_2_wtp1 = df['wtp1'][n_Y:]
    temp = -f_2_xtp1@N_block
    ψ_tilde_x = np.linalg.solve(temp, f_2_xt@N_block)
    ψ_tilde_w = np.linalg.solve(temp, f_2_wtp1)
    σ_v = (df['xtp1'][[0], :]@N_block@ψ_tilde_w + df['wtp1'][[0], :]).T
    μ_0 = (1 - γ) * σ_v
    # Adjustment to E_tilde[v^1_{t+1} - r^1_t]=0
    adj = np.zeros((n_Y, 1))
    σ_v = σ_v.reshape(-1,)
    adj[0, 0] = - 0.5 * (1 - γ)*σ_v.dot(σ_v)
    RHS = - np.block([[(f_1_xtp1@N_block@ψ_tilde_w + f_1_wtp1)@μ_0+adj],
                      [np.zeros((n_Z, 1))]])
    LHS = df['xtp1'] + df['xt']
#    D = np.linalg.solve(LHS, RHS)
    
    ### The following steps are special for the permanent income model
    if np.linalg.matrix_rank(LHS) == len(LHS):
        D = np.linalg.solve(LHS, RHS)
    else:
        new_column = np.zeros((LHS.shape[0],1))
        new_column[3,0] = 1
        new_row = np.zeros((1, LHS.shape[1]+1))
        new_row[0,n_Y+1] = 1
        LHS = np.vstack((np.hstack((LHS, new_column)),new_row))
        RHS = np.vstack((RHS, np.zeros(1)))
        
        D = np.linalg.solve(LHS, RHS)
        discount_adj_1 = np.float(D[-1])
        D = D[:-1]
    ### Permanent income model special treatment ends here
    
    
    # Step 3: Compute C, ψ_tilde_q, laws of motion for Z1
    C = D[:n_Y] - N@D[n_Y:]
    ψ_tilde_q = D[n_Y:] - ψ_tilde_x@D[n_Y:]
    Z1_tp1 = LinQuadVar({'x': ψ_tilde_x, 'w': ψ_tilde_w, 'c': ψ_tilde_q},
                        (n_Z, n_Z, n_W), False)

    # Step 4: Approximate states and endogenous variables
    Z0_t = LinQuadVar({'c': ss[n_Y:].reshape(-1, 1)}, (n_Z, n_Z, n_W), False)
    Z1_t = LinQuadVar({'x': np.eye(n_Z)}, (n_Z, n_Z, n_W), False)
    Y0_t = LinQuadVar({'c': ss[:n_Y].reshape(-1, 1)}, (n_Y, n_Z, n_W), False)
    Y1_t = LinQuadVar({'x': N, 'c': C}, (n_Y, n_Z, n_W), False)
    X1_t = concat([Y1_t, Z1_t])
    Y_t = Y0_t + Y1_t
    Z_t = Z0_t + Z1_t
    X_t = concat([Y_t, Z_t])
        
    X1_tp1 = next_period(X1_t, Z1_tp1)
    X_tp1 = next_period(X_t, Z1_tp1)

    # Step 5: Approximate log M
    log_M = LinQuadVar({'w': μ_0.T, 'c': -0.5 * μ_0.T @ μ_0}, (1, n_Z, n_W),
                       False)

    schur_decomposition = {'Λp': Λp, 'Λ': Λ, 'a': a, 'b': b, 'Q': Q, 'Z': Z}
    res = ModelSolution({'X_t': X_t,
                         'X_tp1': X_tp1,
                         'X1_t': X1_t,
                         'X1_tp1': X1_tp1,
                         'Z1_tp1': Z1_tp1,
                         'log_M': log_M,
                         'var_shape': var_shape,
                         'ss': ss,
                         'schur_decomposition': schur_decomposition,
                         'second_order': False,
                         'message': 'Model solved.',
                         'discount_adj':[discount_adj_1,None]})

    return res


def _second_order_expansion(res_first, df, ss, var_shape, γ, tol=1e-12, max_iter=100):
    """
    Implements second-order expansion.

    """
    n_Y, n_Z, n_W = var_shape
    n_X = n_Y + n_Z
    cov_w = np.eye(n_W)

    # First order expansion
    Z1_tp1 = res_first.X1_tp1[n_Y:n_X]
    X1_t = res_first.X1_t
    X1_tp1 = res_first.X1_tp1
    N = res_first.X_t['x'][:n_Y]
    C = res_first.X_t['c'][:n_Y] - ss[:n_Y].reshape(-1, 1)
    N_block = np.block([[N], [np.eye(n_Z)]])
    C_block = np.block([[C], [np.zeros((n_Z, 1))]])
    M0_E_w = res_first.log_M['w'].T
    M0_E_ww = cal_E_ww(M0_E_w, cov_w)
    Λp = res_first.schur_decomposition['Λp']
    Λ = res_first.schur_decomposition['Λ']
    Q = res_first.schur_decomposition['Q']
    Z = res_first.schur_decomposition['Z']
    Λp22 = Λp[-n_Y:, -n_Y:]
    Λ22 = Λ[-n_Y:, -n_Y:]
    Z21 = Z.T[-n_Y:, :n_Y]
    Z1Z1 = kron_prod(Z1_tp1, Z1_tp1)
    
    discount_adj = res_first.discount_adj

    # Second order expansion
    # Step 1: Combine terms
    D1 = _combine_first_order_terms(df, X1_t, X1_tp1)
    D1_coeff = np.block([[D1['c'], D1['x'], D1['w'],
                          D1['xx'], D1['xw'], D1['ww']]])
    D2 = _combine_second_order_terms(df, X1_t, X1_tp1)
    D2_coeff = np.block([[D2['c'], D2['x'], D2['w'],
                          D2['xx'], D2['xw'], D2['ww']]])
    # E[M1D1], where we set M1=0 in the first iteration
    E_D1_coeff = np.block([[np.zeros_like(D1['c']),
                            np.zeros_like(D1['x']),
                            np.zeros_like(D1['xx'])]])
    # E[D2] under M0 change of measure
    E_D2 = E(D2, M0_E_w, cov_w)
    E_D2_coeff = np.block([[E_D2['c'], E_D2['x'], E_D2['xx']]])

    # Step 2: Prepare coefficients for solving the forward-looking equation
    M0_mat = _form_M0(M0_E_w, M0_E_ww, Z1_tp1, Z1Z1)
    LHS = np.eye(n_Y*M0_mat.shape[0]) - np.kron(M0_mat.T,
                                                np.linalg.solve(Λ22, Λp22))

    # Y_{2,t+1} = Gp_hat[1,ztp1,ztp1ztp1] + N_hatY_{2,t} + G_hat[1,zt,ztzt]
    #             + C_hat[1,zt,w,ztzt,ztw,ww]'
    Y2_coeff = -df['xtp1'][n_Y:]@N_block
    C_hat = np.linalg.solve(Y2_coeff, D2_coeff[n_Y:])
    # Coeffs from C_hat
    C_hat_coeff = np.split(C_hat, np.cumsum([1, n_Z, n_W, n_Z**2, n_Z*n_W]),
                           axis=1)

    # Iteration for M1
    count = 0
    error = 1
    M1_μ_D0_pre = M0_E_w
    M1_μ_D1_pre = 0.

    while error > tol and count < max_iter:
        M1_μ_D0, M1_μ_D1, E_D1_coeff, Z2_tp1, G\
            = _inner_loop(D1, E_D1_coeff, E_D2_coeff, Λ22, Q, df, N,
                          C_hat_coeff, Y2_coeff, LHS, Z21, Z1_tp1,
                          Z1Z1, M0_E_w, M0_E_ww, n_Y, n_Z, n_W, γ, discount_adj)
        # Step 7: Update iteration information
        error_1 = np.max(np.abs(M1_μ_D0 - M1_μ_D0_pre))
        error_2 = np.max(np.abs(M1_μ_D1 - M1_μ_D1_pre))
        error = np.max([error_1, error_2])
        M1_μ_D0_pre = M1_μ_D0
        M1_μ_D1_pre = M1_μ_D1
        count += 1

    if count == max_iter:
        message = 'Maximum iteration reached. M1 not converged.'
    else:
        message = 'M1 converged. Model solved.'

    # Approximate states and endogenous variables
    Z2_t = LinQuadVar({'x2': np.eye(n_Z)}, (n_Z, n_Z, n_W), False)
    Y2_t = LinQuadVar({'x2': N,
                       'xx': G[:, 1+n_Z:1+n_Z+n_Z**2],
                       'x': G[:, 1:1+n_Z],
                       'c': G[:, :1]}, (n_Y, n_Z, n_W), False)
    X2_t = concat([Y2_t, Z2_t])
    X_t = res_first.X_t + X2_t*0.5

    X_tp1 = next_period(X_t, Z1_tp1, Z2_tp1, Z1Z1)
    X2_tp1 = next_period(X2_t, Z1_tp1, Z2_tp1, Z1Z1)

    # Approximate log M
    μ_0 = M1_μ_D0 + M0_E_w
    μ_1 = M1_μ_D1
    log_M = LinQuadVar({'xw': vec(μ_1).T,
                        'w': μ_0.T,
                        'xx': -0.5*vec(μ_1.T@μ_1).T,
                        'x': -μ_0.T@μ_1,
                        'c': -0.5 * μ_0.T @ μ_0}, (1, n_Z, n_W), False)

    res = ModelSolution({'X_t': X_t,
                         'X_tp1': X_tp1,
                         'X1_t': X1_t,
                         'X1_tp1': X1_tp1,
                         'X2_t': X2_t,
                         'X2_tp1': X2_tp1,
                         'Z1_tp1':Z1_tp1,
                         'Z2_tp1':Z2_tp1,
                         'log_M': log_M,
                         'var_shape': var_shape,
                         'ss': res_first.ss,
                         'schur_decomposition': res_first.schur_decomposition,
                         'nit': count,
                         'second_order': True,
                         'message': message,
                         'discount_adj': discount_adj})
    return res


class ModelSolution(dict):
    """
    Represents the model solution.

    Attributes
    ----------
    X_t : LinQuadVar
        Approximation for :math:`X_{t}` in terms of :math:`Z_{1,t}`.
    X_tp1 : LinQuadVar
        Approximation for :math:`X_{t+1}` in terms of :math:`Z_{1,t}`
        and :math:`W_{t+1}`.
    X1_t : LinQuadVar
        Representation of :math:`X_{1, t}`.
    X2_t : LinQuadVar
        Representation of :math:`X_{2, t}`.
    X1_tp1 : LinQuadVar
        Representation of :math:`X_{1, t+1}`.
    X2_tp1 : LinQuadVar
        Representation of :math:`X_{2, t+1}`.
    log_M : LinQuadVar
        Approximation for log change of measure in terms of :math:`X_{1,t}`
        and :math:`X_{2,t}`.
    nit : int
        Number of iterations performed to get M1. For second-order expansion
        only.
    second_order : bool
        If True, the solution is in second-order.
    var_shape : tuple of ints
        (n_Y, n_Z, n_W). Number of endogenous variables, states and shocks
        respectively.
    ss : (n_X, ) ndarray
        Steady states.
    message : str
        Message from the solver.

    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())    

    def approximate_fun(self, fun, args=()):
        """
        Approximates a scalar variable as a function of X and W.

        Parameters
        ----------
        fun : callable
            Function to be approximated.
            ``var_fun(X_t, X_tp1, W_tp1, q, *args) -> scalar``

            X_t, X_tp1, W_tp1, q, and args are the same as inputs to eq_cond().
        args : tuple of floats/ints
            Additional parameters to be passed to fun.

        Returns
        -------
        fun_approx : LinQuadVar
            Approximation for the input.

        """
        _, n_Z, n_W = self.var_shape

        W_0 = np.zeros(n_W)
        q_0 = 0.
        
        dfun = compute_derivatives(f=lambda X_t, X_tp1, W_tp1, q:
                                   anp.atleast_1d(fun(X_t, X_tp1, W_tp1, q, *args)),
                                   X=[self.ss, self.ss, W_0, q_0],
                                   second_order=self.second_order)

        fun_zero_order = fun(self.ss, self.ss, W_0, q_0, *args)
        fun_first_order = matmul(dfun['xtp1'], self.X1_tp1)\
            + matmul(dfun['xt'], self.X1_t)\
            + LinQuadVar({'w': dfun['wtp1'], 'c': dfun['q'].reshape(-1, 1)},
                         (1, n_Z, n_W), False)
        fun_approx = fun_zero_order + fun_first_order
        if self.second_order:
            temp1 = _combine_second_order_terms(dfun, self.X1_t, self.X1_tp1)
            temp2 = matmul(dfun['xt'], self.X2_t)\
                + matmul(dfun['xtp1'], self.X2_tp1)
            fun_second_order = temp1 + temp2
            fun_approx = fun_approx + fun_second_order*0.5

        return fun_approx

    def simulate(self, Ws):
        """
        Simulates stochastic path for X by generating iid normal shocks,
        or deterministic path for X by generating zero-valued shocks.

        Parameters
        ----------
        Ws : (T+burn_in, n_W) ndarray
            n_W dimensional shocks for (T+burn_in) periods to be fed into the system.        
        T : int
            Time horizon.
        burn_in: int
            Throwing away some iterations at the beginning of the simulation.

        Returns
        -------
        sim_result : (T, n_Y) ndarray
            Simulated Ys.

        """
        n_Y, n_Z, n_W = self.var_shape
        Z2_tp1 = self.X2_tp1[n_Y: n_Y+n_Z] if self.second_order else None
        sim_result = simulate(self.X_t,
                              self.X1_tp1[n_Y: n_Y+n_Z],
                              Z2_tp1,
                              Ws)
        return sim_result
        
    def IRF(self, T, shock):
        r"""
        Computes impulse response functions for each component in X to each shock.

        Parameters
        ----------
        T : int
            Time horizon.
        shock : int
            Position of the initial shock, starting from 0.

        Returns
        -------
        states : (T, n_Z) ndarray
            IRF of all state variables to the designated shock.
        controls : (T, n_Y) ndarray
            IRF of all control variables to the designated shock.

        """
    
        n_Y, n_Z, n_W = self.var_shape
        # Build the first order impulse response for each of the shocks in the system
        states1 = np.zeros((T, n_Z))
        controls1 = np.zeros((T, n_Y))
        
        W_0 = np.zeros(n_W)
        W_0[shock] = 1
        B = self.X1_tp1['w'][n_Y:,:]
        F = self.X1_tp1['w'][:n_Y,:]
        A = self.X1_tp1['x'][n_Y:,:]
        D = self.X1_tp1['x'][:n_Y,:]
        N = self.X1_t['x'][:n_Y,:]
        states1[0, :] = B@W_0
        controls1[0, :] = F@W_0
        for i in range(1,T):
            states1[i, :] = A@states1[i-1, :]
            controls1[i, :] = D@states1[i-1, :]
        if not self.second_order:
            states = states1
            controls = controls1
        else:
            # Define the evolutions of the states in second order
            # X_{t+1}^2 = Ψ_0 + Ψ_1 @ X_t^1 + Ψ_2 @ W_{t+1} + Ψ_3 @ X_t^2 +
            # Ψ_4 @ (X_t^1 ⊗ X_t^1) + Ψ_5 @ (X_t^1 ⊗ W_{t+1}) + Ψ_6 @ (W_{t+1} ⊗ W_{t+1})
            Ψ_0 = self.X2_tp1['c'][n_Y:,:]
            Ψ_1 = self.X2_tp1['x'][n_Y:,:]
            Ψ_2 = self.X2_tp1['w'][n_Y:,:]
            Ψ_3 = self.X2_tp1['x2'][n_Y:,:]
            Ψ_4 = self.X2_tp1['xx'][n_Y:,:]
            Ψ_5 = self.X2_tp1['xw'][n_Y:,:]
            Ψ_6 = self.X2_tp1['ww'][n_Y:,:]
            
            Φ_0 = self.X2_tp1['c'][:n_Y,:]
            Φ_1 = self.X2_tp1['x'][:n_Y,:]
            Φ_2 = self.X2_tp1['w'][:n_Y,:]
            Φ_3 = self.X2_tp1['x2'][:n_Y,:]
            Φ_4 = self.X2_tp1['xx'][:n_Y,:]
            Φ_5 = self.X2_tp1['xw'][:n_Y,:]
            Φ_6 = self.X2_tp1['ww'][:n_Y,:]
            
            states2 = np.zeros((T, n_Z))
            controls2 = np.zeros((T, n_Y))
            X_1_0 = np.zeros(n_Z)

            # Build the second order impulse response for each shock
            W_0 = np.zeros(n_W)
            W_0[shock] = 1
            states2[0, :] = Ψ_2 @ W_0 + Ψ_5 @ np.kron(X_1_0, W_0) + Ψ_6 @ np.kron(W_0, W_0)
            controls2[0, :] = Φ_2 @ W_0 + Φ_5 @ np.kron(X_1_0, W_0) + Φ_6 @ np.kron(W_0, W_0)
            for i in range(1,T):
                states2[i, :] = Ψ_1 @ states1[i-1, :] + Ψ_3 @ states2[i-1, :] + \
                    Ψ_4 @ np.kron(states1[i-1, :], states1[i-1, :])
                controls2[i, :] = Φ_1 @ states1[i-1, :] + Φ_3 @ states2[i-1, :] + \
                    Φ_4 @ np.kron(states1[i-1, :], states1[i-1, :])
            states = states1 + .5 * states2
            controls = controls1 + .5 * controls2
            
        return states, controls

    def elasticities(self, log_SDF_ex, args, locs=None, T=400, shock=0, percentile=0.5):
        r"""
        Computes shock exposure and price elasticities for X.

        Parameters
        ----------
        log_SDF_ex : callable
            Log stochastic discount factor exclusive of the
            change of measure M.

            ``log_SDF_ex(X_t, X_tp1, W_tp1, q, *args) -> scalar``
        args : tuple of floats/ints
            Additional parameters passed to log_SDF_ex.
        locs : None or tuple of ints
            Positions of variables of interest.
            If None, all variables will be selected.
        T : int
            Time horizon.
        shock : int
            Position of the initial shock, starting from 0.
        percentile : float
            Specifies the percentile of the elasticities.

        Returns
        -------
        elasticities : (T, n_Y) ndarray
            Elasticities for M.

        References
        ---------
        Borovicka, Hansen (2014). See http://larspeterhansen.org/.

        """
        n_Y, n_Z, n_W = self.var_shape
        log_SDF_ex = self.approximate_fun(log_SDF_ex, args)
        self.log_SDF = log_SDF_ex + self.log_M
        Z2_tp1 = self.X2_tp1[n_Y: n_Y+n_Z] if self.second_order else None
        X_growth = self.X_tp1 - self.X_t
        X_growth_list = X_growth.split()
        if locs is not None:
            X_growth_list = [X_growth_list[i] for i in locs]
        exposure_all = np.zeros((T, len(X_growth_list)))
        price_all = np.zeros((T, len(X_growth_list)))
        for i, x in enumerate(X_growth_list):
            exposure = exposure_elasticity(x,
                                           self.X1_tp1[n_Y: n_Y+n_Z],
                                           Z2_tp1,
                                           T,
                                           shock,
                                           percentile)
            price = price_elasticity(x,
                                     self.log_SDF,
                                     self.X1_tp1[n_Y: n_Y+n_Z],
                                     Z2_tp1,
                                     T,
                                     shock,
                                     percentile)
            exposure_all[:, i] = exposure.reshape(-1)
            price_all[:, i] = price.reshape(-1)
        return exposure_all, price_all


def _combine_first_order_terms(df, X1_t, X1_tp1):
    r"""
    This function combines terms from first-order f.

    Parameters
    ----------
    df : dict
        Partial derivatives of f.
    X1_t : LinQuadVar
        Expression of :math:`X_{1,t}`.
    X1_tp1 : LinQuadVar
        Expression of :math:`X_{1,t+1}`.

    Returns
    -------
    res : LinQuadVar
        Combined first-order terms from df.

    """
    _, n_Z, n_W = X1_t.shape
    res = matmul(df['xtp1'], X1_tp1)\
        + matmul(df['xt'], X1_t)\
        + LinQuadVar({'w': df['wtp1'], 'c': df['q'].reshape(-1, 1)},
                     (df['xt'].shape[0], n_Z, n_W), False)
    return res


def _combine_second_order_terms(df, X1_t, X1_tp1):
    r"""
    This function combines terms from second-order f, except those for X_{2,t}
    and X_{2,t+1}. Suppose [[Y_{1,t}],[Z_{1,t}]] = N_block Z_{1,t} + C_block,
    where Y_t are enndogenous variables and Z_t are states.

    Parameters
    ----------
    df : dict
        Partial derivatives of f.
    X1_t : LinQuadVar
        Expression of :math:`X_{1,t}`.
    X1_tp1 : LinQuadVar
        Expression of :math:`X_{1,t+1}`

    Returns
    -------
    res : LinQuadVar
        Combined second-order terms (except X_{2,t} and X_{2,t+1}) from df.

    """
    _, n_Z, n_W = X1_tp1.shape
    wtp1 = LinQuadVar({'w': np.eye(n_W)}, (n_W, n_Z, n_W), False)
    xtxt = kron_prod(X1_t, X1_t)
    xtxtp1 = kron_prod(X1_t, X1_tp1)
    xtwtp1 = kron_prod(X1_t, wtp1)
    xtp1xtp1 = kron_prod(X1_tp1, X1_tp1)
    xtp1wtp1 = kron_prod(X1_tp1, wtp1)
    wtp1wtp1 = LinQuadVar({'ww': np.eye(n_W**2)}, (n_W**2, n_Z, n_W), False)

    res = matmul(df['xtxt'], xtxt)\
        + matmul(2*df['xtxtp1'], xtxtp1)\
        + matmul(2*df['xtwtp1'], xtwtp1)\
        + matmul(2*df['xtq'], X1_t)\
        + matmul(df['xtp1xtp1'], xtp1xtp1)\
        + matmul(2*df['xtp1wtp1'], xtp1wtp1)\
        + matmul(2*df['xtp1q'], X1_tp1)\
        + matmul(df['wtp1wtp1'], wtp1wtp1)\
        + matmul(2*df['wtp1q'], wtp1)\
        + LinQuadVar({'c': df['qq']}, (df['qq'].shape[0], n_Z, n_W), False)

    return res


def _form_M0(M0_E_w, M0_E_ww, Z1_tp1, Z1Z1):
    """
    Get M0_mat, which satisfies E[M0 A [1 ztp1 ztp1ztp1]] = A M0_mat[1 zt ztzt]

    """
    _, n_Z, n_W = Z1_tp1.shape
    M0_mat_11 = np.eye(1)
    M0_mat_12 = np.zeros((1, n_Z))
    M0_mat_13 = np.zeros((1, n_Z**2))
    M0_mat_21 = Z1_tp1['w']@M0_E_w + Z1_tp1['c']
    M0_mat_22 = Z1_tp1['x']
    M0_mat_23 = np.zeros((n_Z, n_Z**2))
    M0_mat_31 = Z1Z1['ww']@M0_E_ww + Z1Z1['w']@M0_E_w + Z1Z1['c']
    temp = np.vstack([M0_E_w.T@mat(Z1Z1['xw'][row: row+1, :], (n_W, n_Z))
                      for row in range(Z1Z1.shape[0])])    
    M0_mat_32 = temp + Z1Z1['x']
    M0_mat_33 = Z1Z1['xx']
    M0_mat = np.block([[M0_mat_11, M0_mat_12, M0_mat_13],
                       [M0_mat_21, M0_mat_22, M0_mat_23],
                       [M0_mat_31, M0_mat_32, M0_mat_33]])
    return M0_mat


def _inner_loop(D1, E_D1_coeff, E_D2_coeff, Λ22, Q, df, N,
                C_hat_coeff, Y2_coeff, LHS, Z21, Z1_tp1,
                Z1Z1, M0_E_w, M0_E_ww, n_Y, n_Z, n_W, γ, discount_adj):
    # Step 3: Solve the forward-looking equation
    D_coeff = E_D2_coeff + 2*E_D1_coeff
    RHS = vec(-np.linalg.solve(Λ22, (Q.T@D_coeff)[-n_Y:]))
    D_tilde_vec = np.linalg.solve(LHS, RHS)
    D_tilde = mat(D_tilde_vec, (n_Y, 1+n_Z+n_Z**2))
    
    # Step 4: Compute ψ_tilde derivatives
    # Y_{2,t} = NZ_{2,t} + G[1,zt,ztzt]'
    G = np.linalg.solve(Z21, D_tilde)
    G_block = np.block([[G], [np.zeros((n_Z, 1+n_Z+n_Z**2))]])

    # Z_{2,t+1} = Gp_hat[1,ztp1,ztp1ztp1] + N_hatZ_{2,t} + G_hat[1,zt,ztzt]
    #           + C_hat[1,zt,w,ztzt,ztw,ww]'
    Gp_hat = np.linalg.solve(Y2_coeff, df['xtp1'][n_Y:]@G_block)
    G_hat = np.linalg.solve(Y2_coeff, df['xt'][n_Y:]@G_block)

    # Combine coefficients
    # 1) Coeffs from C_hat
    c_1, x_1, w_1, xx_1, xw_1, ww_1 = C_hat_coeff
    # 2) Coeffs from G_hat
    c_2, x_2, xx_2 = np.split(G_hat, np.cumsum([1, n_Z]), axis=1)
    # 3) Coeffs from Gp_hat
    var = LinQuadVar({'c': Gp_hat[:, :1], 'x': Gp_hat[:, 1:1+n_Z],
                      'xx': Gp_hat[:, 1+n_Z:1+n_Z+n_Z**2]},
                     (n_Z, n_Z, n_W), False)
    var_next = next_period(var, Z1_tp1, None, Z1Z1)

    Z2_tp1 = LinQuadVar({'x2': Z1_tp1['x'],
                         'xx': var_next['xx'] + xx_1 + xx_2,
                         'xw': var_next['xw'] + xw_1,
                         'ww': var_next['ww'] + ww_1,
                         'x': var_next['x'] + x_1 + x_2,
                         'w': var_next['w'] + w_1,
                         'c': var_next['c'] + c_1 + c_2},
                        (n_Z, n_Z, n_W), False)
    ### The following steps are special for the permanent income model
    # Step 4.5 Include an additional constant term
    
    f_1_xtp1 = df['xtp1'][:n_Y]
    f_1_xt = df['xt'][:n_Y]
#    f_1_wtp1 = df['wtp1'][:n_Y]
    f_2_xtp1 = df['xtp1'][n_Y:]
    f_2_xt = df['xt'][n_Y:]
#    f_2_wtp1 = df['wtp1'][n_Y:]
    
    LHS_const_term = np.block([[f_1_xtp1+f_1_xt], [f_2_xtp1+f_2_xt]])
    
    # Solve for R_1
    G[:,0] = 0 # times Y_2,t:=[1,xt,xtxt], predetermined variables only
    var_G = LinQuadVar({'c':G[:,:1],
             'x':G[:,1:1+n_Z],
             'xx':G[:,1+n_Z:1+n_Z+n_Z**2]}, (n_Y, n_Z, n_W), False)
    res_G_next  = next_period(var_G, Z1_tp1, None, Z1Z1)
    c_G_next    = res_G_next['c']
    w_G_next    = res_G_next['w']
    ww_G_next   = res_G_next['ww']
    cons_G_next = c_G_next+w_G_next@M0_E_w+ww_G_next@M0_E_ww
    
    c_J_next    = np.zeros_like(Z2_tp1['c'])
    c_J_next    = np.zeros((n_Z,1))
    w_J_next    = Z2_tp1['w']
    ww_J_next   = var_next['ww'] + ww_1
    cons_J_next = c_J_next+w_J_next@M0_E_w+ww_J_next@M0_E_ww
    
    cons_1 = f_1_xtp1@np.block([[N@cons_J_next+cons_G_next],[cons_J_next]]) # Constant from the first term
    cons_2 = E_D2_coeff[:n_Y,:1] # Constant from L_t
    R_1 = cons_1 + cons_2
    
    
    # Solve for R_2
    cons_1 = f_2_xtp1@np.block([[N@c_J_next+c_G_next],[c_J_next]])
    cons_2 = df['qq'][n_Y:] + df['q'][n_Y:].reshape(-1,1)
    R_2 = cons_1 + cons_2
    
    
    RHS_const_term = np.block([[-R_1],[-R_2]])
    
    if np.linalg.matrix_rank(LHS_const_term) == len(LHS_const_term):
        D = np.linalg.solve(LHS, RHS)
    else:
        new_column = np.zeros((LHS_const_term.shape[0],1))
        new_column[3,0] = 1
        new_row = np.zeros((1, LHS_const_term.shape[1]+1))
        new_row[0,n_Y+1] = 1
        LHS_const_term = np.vstack((np.hstack((LHS_const_term, new_column)),new_row))
        RHS_const_term = np.vstack((RHS_const_term, np.zeros(1)))
        
        D = np.linalg.solve(LHS_const_term, RHS_const_term)
        discount_adj_2 = np.float(D[-1])
        discount_adj[1] = discount_adj_2
        D = D[:-1]
    
    G[:,:1] = D[:n_Y] - N @D[n_Y:]
    
    Z2_tp1 = LinQuadVar({'x2': Z1_tp1['x'],
                     'xx': var_next['xx'] + xx_1 + xx_2,
                     'xw': var_next['xw'] + xw_1,
                     'ww': var_next['ww'] + ww_1,
                     'x': var_next['x'] + x_1 + x_2,
                     'w': var_next['w'] + w_1,
                     'c': D[n_Y:] - Z1_tp1['x']@D[n_Y:]},
                    (n_Z, n_Z, n_W), False)
    ### Permanent income model special treatment ends here
                        
    
    # Step 5: Update M1
    # v^2_t+1 = N[0:1,:]Z^2_2,t+1 + G[0:1,:][1,ztp1,ztp1ztp1]
    # := B1 ww + B2 zw + B3 w
    _, coeff_xtp1, coeff_xtp1xtp1 = np.split(G[0:1, :], np.cumsum([1, n_Z]),
                                             axis=1)
    B1 = N[0:1, :]@Z2_tp1['ww'] + coeff_xtp1xtp1@Z1Z1['ww']
    B2 = N[0:1, :]@Z2_tp1['xw'] + coeff_xtp1xtp1@Z1Z1['xw']
    B3 = N[0:1, :]@Z2_tp1['w'] + coeff_xtp1@Z1_tp1['w']\
        + coeff_xtp1xtp1@Z1Z1['w']
    M1_μ_D0 = 0.5 * (1 - γ) * ((-B1@M0_E_ww)*M0_E_w + B3.T)
    M1_μ_D1 = 0.5 * (1 - γ) * mat(B2.T, (n_W, n_Z))

    # Step 6: Update D1
    # Notice E[M1 Y_t] = 0
    E_D1_c = D1['w']@M1_μ_D0
    E_D1_x = D1['w']@M1_μ_D1
    # Adjusment for 2nd order relationship between v and r
    # E[M0(v^2_{t+1}-r^2_t)]=0
    E_D1_c[0, :] = np.zeros_like(E_D1_c[0, :])
    E_D1_x[0, :] = np.zeros_like(E_D1_x[0, :])
    E_D1_coeff = np.block([[E_D1_c, E_D1_x,
                            np.zeros((E_D1_x.shape[0], n_Z**2))]])

    return M1_μ_D0, M1_μ_D1, E_D1_coeff, Z2_tp1, G
