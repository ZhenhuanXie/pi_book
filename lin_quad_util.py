"""
Tools for operations on LinQuadVar.

"""
import numpy as np
from scipy.stats import norm
from utilities import vec, mat, sym, cal_E_ww
from lin_quad import LinQuadVar
from numba import njit
import time


def next_period(Y, X1_tp1, X2_tp1=None, X1X1=None):
    """
    Gets representation for Y_{t+1} when Y only contains time t variables.

    Parameters
    ----------
    Y : LinQuadVar
        Stores the coefficient of Y.
    X1_tp1 : LinQuadVar
        Stores the coefficients of laws of motion for X1.
    X2_tp2 : LinQuadVar or None
        Stores the coefficients of laws of motion for X2. Does not need
        to be specified when Y['x2'] is zero ndarray.
    X1X1 : LinQuadVar or None
        Stores the coefficients of :math:`X_{1,t+1}\otimes X_{1,t+1}`.
        If None, the function will recalculate it.

    Returns
    -------
    Y_next : LinQuadVar

    """
    if not Y.deterministic:
        raise ValueError("Y should only contain time t variables.")
    n_Y, n_X, n_W = Y.shape
    Y_next = LinQuadVar({'c': Y['c']}, (n_Y, n_X, n_W))\
            + matmul(Y['x'], X1_tp1)
    # if np.any(Y['xx'] != 0)
    if Y['xx'].any():
        if X1X1 is None:
            X1X1 = kron_prod(X1_tp1, X1_tp1)
        Y_next = Y_next + matmul(Y['xx'], X1X1)
    if Y['x2'].any():
        Y_next = Y_next + matmul(Y['x2'], X2_tp1)

    return Y_next


def kron_prod(Y1, Y2):
    """
    Computes the Kronecker product of Y1 and Y2, where Y1 and Y2
    do not have second-order terms.
    
    Parameters
    ----------
    Y1 : LinQuadVar
        Y1.second_order should be False.
    Y2 : LinQuadVar
        Y2.second_order should be False.

    Returns
    -------
    Y_kron : LinQuadVar
        Stores coefficients for the Kronecker product of Y1 and Y2.

    """
    if Y1.second_order or Y2.second_order:
        raise ValueError('Y1.second_order and Y2.second_order should be False.')  
    n_Y1, n_X, n_W = Y1.shape
    n_Y2, _, _ = Y2.shape
    kron_prod = {}
    terms = ['x', 'w', 'c']
    for key_left in terms:
        for key_right in terms:
            # if np.any(Y1[key_left] != 0) and np.any(Y2[key_right] != 0)
            if Y1[key_left].any() and Y2[key_right].any():
                kron_prod[key_left+key_right] = np.kron(Y1[key_left], Y2[key_right])
            else:
                _, m1 = Y1[key_left].shape
                _, m2 = Y2[key_right].shape
                kron_prod[key_left+key_right] = np.zeros((n_Y1*n_Y2, m1*m2))
    # Combine terms
    xx = kron_prod['xx']
    wx = kron_prod['wx']
    wx_reshape = np.vstack([vec(mat(wx[row:row+1, :].T, (n_X, n_W)).T).T for row in range(wx.shape[0])])
    xw = kron_prod['xw'] + wx_reshape
    ww = kron_prod['ww']
    x = kron_prod['xc'] + kron_prod['cx']
    w = kron_prod['wc'] + kron_prod['cw']
    c = kron_prod['cc']
    
    Y_kron = LinQuadVar({'xx': xx, 'xw': xw, 'ww': ww, 'x': x, 'w': w, 'c': c},
                        (n_Y1*n_Y2, n_X, n_W))
    
    return Y_kron


def matmul(matrix, Y):
    """
    Computes matrix@Y[key] for each key in Y.
    
    Parameters
    ----------
    matrix : (n1, n2) ndarray
    Y : (n2, n_X, n_W) LinQuadVar
    
    Returns
    Y_new : (n1, n_X, n_W) LinQuadVar

    """
    Y_new_coeffs = {}
    n_Y, n_X, n_W = Y.shape
    for key in Y.coeffs:
        Y_new_coeffs[key] = matrix @ Y.coeffs[key]
    Y_new = LinQuadVar(Y_new_coeffs, (matrix.shape[0], n_X, n_W), False)
    return Y_new


def concat(Y_list):
    """
    Concatenates a list of LinQuadVar.

    Parameters
    ----------
    Y_list : list of (n_Yi, n_X, n_W) LinQuadVar

    Returns
    -------
    Y_cat : (n_Y1+n_Y2..., n_X, n_W) LinQuadVar
    
    See Also
    --------
    LinQuadVar.split : Splits the N-dimensional Y into N 1-dimensional Ys.

    """
    terms = []
    for Y in Y_list:
        terms = set(terms) | set(Y.coeffs.keys())
    Y_cat = {}
    for key in terms:
        Y_coeff_list = [Y[key] for Y in Y_list]
        Y_cat[key] = np.concatenate(Y_coeff_list, axis=0)
    temp = list(Y_cat.keys())[0]
    n_Y_cat = Y_cat[temp].shape[0]
    n_X = Y_list[0].shape[1]
    n_W = Y_list[0].shape[2]
    Y_cat = LinQuadVar(Y_cat, (n_Y_cat, n_X, n_W), False)

    return Y_cat
    

def E(Y, E_w, Cov_w=None):
    r"""
    Computes :math:`E[Y_{t+1} \mid \mathfrak{F}_t]`,
    Parameters
    ----------
    Y : LinQuadVar
    E_w : (n_W, 1) ndarray
    Cov_w : (n_W, n_W) ndarray
        Used when the Y has non-zero coefficient on 'ww' term.
    Returns
    -------
    E_Y : LinQuadVar
    """
    n_Y, n_X, n_W = Y.shape
    if Y.deterministic:
        return LinQuadVar(Y.coeffs, Y.shape)
    else:
        E_Y = {}
        E_Y['x2'] = Y['x2']
        E_Y['xx'] = Y['xx']
        temp = np.vstack([E_w.T@mat(Y['xw'][row: row+1, :], (n_W, n_X))
                          for row in range(n_Y)])
        E_Y['x'] = temp + Y['x']
        E_Y['c'] = Y['c'] + Y['w'] @ E_w
        if Y['ww'].any():
            E_ww = cal_E_ww(E_w, Cov_w)
            E_Y['c'] += Y['ww'] @ E_ww
        E_Y = LinQuadVar(E_Y, Y.shape, False)
        return E_Y


def log_E_exp(Y):
    r"""
    Computes :math:`\log E[\exp(Y_{t+1}) \mid \mathfrak{F}_t]`,
    assuming shocks follow iid normal distribution.

    Parameters
    ----------
    Y : LinQuadVar

    Returns
    -------
    Y_sol : LinQuadVar

    References
    ---------
    Borovicka, Hansen (2014). See http://larspeterhansen.org/.

    """
    n_Y, n_X, n_W = Y.shape
    if n_Y != 1:
        raise ValueError('Y should be scalar-valued.')
    if Y.deterministic:
        return LinQuadVar(Y.coeffs, Y.shape)
    else:
        x2, xx, x, c = _log_E_exp_jit(Y['x2'], Y['x'], Y['w'],
                                      Y['c'], Y['xx'], Y['xw'],
                                      Y['ww'], n_X, n_W)
        Y_sol = LinQuadVar({'x2': x2, 'xx': xx, 'x': x, 'c':c}, Y.shape, False)
        return Y_sol


def simulate(Y, X1_tp1, X2_tp1, Ws):
    """
    Simulate a time path for `Y` given shocks `Ws`.

    Parameters
    ----------
    Y : LinQuadVar
        Variable to be simulated.
    X1_tp1 : LinQuadVar
        Stores the coefficients of laws of motion for X1.
    X2_tp2 : LinQuadVar or None
        Stores the coefficients of laws of motion for X2. Does not need to be
        specified when Y only has first-order terms.
    Ws : (T, n_W) ndarray
        n_W dimensional shocks for T periods to be fed into the system.

    Returns
    -------
    sim_result : (T, n_Y) ndarray
        Simulated Ys.

    """
    n_Y, n_X, n_W = Y.shape
    T = Ws.shape[0]
    Ws = Ws.reshape(T, n_W, 1)
    x1 = np.zeros((T, n_X, 1))
    x2 = np.zeros((T, n_X, 1))

    for i in range(1, T):
        x1[i] = X1_tp1(x1[i-1], np.zeros((n_X, 1)), Ws[i])

    if Y.second_order:
        for i in range(1, T):
            x2[i] = X2_tp1(x1[i-1], x2[i-1], Ws[i])         
    sim_result = np.vstack([Y(x1[i], x2[i], Ws[i]).ravel() for i in range(T)])

    return sim_result


@njit
def _log_E_exp_jit(x2, x, w, c, xx, xw, ww, n_X, n_W):
    Σ = np.eye(n_W) - sym(mat(2 * ww, (n_W, n_W)))
    Σ_xw_solved = np.linalg.solve(Σ, mat(xw, (n_W, n_X)))
    new_x2 = x2
    new_xx = xx + 0.5 * vec(mat(xw, (n_W, n_X)).T
                                      @ Σ_xw_solved).T
    new_x = x + w @ Σ_xw_solved
    new_c = c - 1. / 2 * np.log(np.linalg.det(Σ))\
        + 1. / 2 * w @ np.linalg.solve(Σ, w.T)

    return new_x2, new_xx, new_x, new_c
