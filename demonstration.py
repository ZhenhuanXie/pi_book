"""
Functions to facilitate the computation in expansion solvers. 

"""
import numpy as np
import matplotlib.pyplot as plt


# Notation is consistent with Knight chapter 11 (Jan 27 version on Canvas).

def kalman_filter_find_steady(A, B, H, D, F, Σ0, max_iter, tol, verbose=False):
    """
    This function computes steady state covariance matrix and associated Kalman gain.
    
    Parameters
    ----------
    A : (n_x, n_x) ndarray
    B : (n_x, n_w) ndarray
    H : (n_z, 1) ndarray
    D : (n_z, n_x) ndarray
    F : (n_z, n_w) ndarray
    Σ0 : (n_x, n_x) ndarray
        initial covariance matrix of hidden states
    max_iter : int
        maximum number of iterations for computing the steady state
    tol : float
        threshold to decide whether convergence is attained
    
    Returns
    ----------
    Σ_ss : (n_x, n_x) ndarray
        steady state covariance matrix
    K_ss : (n_x, n_z) ndarray
        steady state Kalman gain
    
    """

    Σ_t = Σ0
    count = 0
    diff = 1
    
    while diff > tol and count < max_iter:
        Ω = D@Σ_t@D.T + F@F.T
        invΩ = np.linalg.inv(Ω)
        K = (A@Σ_t@D.T + B@F.T)@invΩ
        Σ_t_next = A@Σ_t@A.T + B@B.T - (A@Σ_t@D.T + B@F.T)@invΩ@(D@Σ_t@A.T + F@B.T)
        diff = np.linalg.norm(Σ_t_next - Σ_t)
        Σ_t = Σ_t_next.copy() #save a copy for next iteration
        count += 1
    if count == max_iter and verbose:
        print("Failed to converge within {} iterations.".format(max_T))
    elif verbose:
        print("Converged within {} iterations.".format(count))
        
    Σ_ss = Σ_t
    K_ss = K

    if verbose:
        print("Steady state covariance matrix is: \n {}.".format(Σ_ss))
        print("Steady state Kalman gain is: \n {}.".format(K_ss))
    
    return Σ_ss, K_ss

def Innovations_rep(Σ, K, D, F):
    """
    This function computes coefficient matrices in the Innovations representation system,
    where a Gram-schmidt process is used to construct orthogonal shocks.
    
    Parameters
    ----------
    Σ : (n_x, n_x) ndarray
    K : (n_x, n_z) ndarray
    D : (n_z, n_x) ndarray
    F : (n_z, n_w) ndarray
    
    Returns
    ----------
    B_bar : (n_x, n_z) ndarray
    F_bar : (n_z, n_z) ndarray
    F_bar_inv: (n_z, n_z) ndarray
    
    """
    
    Omega =  D @ Σ @ D.T + F @ F.T
    F_bar = np.linalg.cholesky(Omega) #Cholesky decomposition
    F_bar_inv = np.linalg.inv(F_bar)
    B_bar = K @ F_bar
    
    return B_bar, F_bar, F_bar_inv

def compute_IRF(A, B, D, F, T):
    """
    This function computes impulse response of state variables as well as signal variables to a unit of each shock.
    
    Parameters
    ----------
    A : (n_x, n_x) ndarray
    B : (n_x, n_w) ndarray
    D : (n_z, n_x) ndarray
    F : (n_z, n_w) ndarray
    T : int
        Time horizon for impulse response calculation.
    
    Returns
    ----------
    state_IRF : (n_w, T, n_x) ndarray
        Impulse response function of each hidden state variable to each shock.
    signal_IRF : (n_w,T, n_z) ndarray
        Impulse response function of each signal variable to each shock.
    """
    
    # check matrix dimensions are compatible
    if len(set([A.shape[0],A.shape[1], B.shape[0], D.shape[1]])) != 1:
        raise ValueError("Matrix dimensions not compatible. Please double check the number of states.")
    if B.shape[1]!= F.shape[1]:
        raise ValueError("Matrix dimensions not compatible. Please double check the number of shocks.")
    if D.shape[0]!= F.shape[0]:
        raise ValueError("Matrix dimensions not compatible. Please double check the number of signals.")
        
    n_shocks = B.shape[1]
    n_states = A.shape[0]
    n_signals = D.shape[0]
    
    state_IRF = np.zeros((n_shocks, T, n_states))
    signal_IRF = np.zeros((n_shocks, T, n_signals))
    for k in range(n_shocks): # loop over shock numbers, since we want to look at the IRF to each shock separately
        W_0 = np.zeros(n_shocks)
        W_0[k] = 1    # generate a unit shock only at the beginning
        state_IRF[k, 0] = B@W_0
        signal_IRF[k, 0] = F@W_0
        for i in range(1,T):
            state_IRF[k, i] = A@state_IRF[k, i-1]
            signal_IRF[k, i] = D@state_IRF[k, i-1]
    return state_IRF, signal_IRF

def plot_figure_1():
    # matrices in the original system
    A = np.array([[.704, 0, 0],[0, 1, -.154],[0,1,0]])
    B = np.array([[.144,0],[0,.206],[0,0]])
    H = np.zeros((1,1))
    D = .01 * np.array([.704, 0, -.154]).reshape(1,-1)
    F = .01 * np.array([.144,.206]).reshape(1,-1)

    # initial distribution
    Σ0 = np.eye(3)

    # parameters for iteration
    max_iter = 1000
    tol = 1e-15

    Σ_ss, K_ss = kalman_filter_find_steady(A, B, H, D, F, Σ0, max_iter, tol)

    B_bar_ss, F_bar_ss, F_bar_inv_ss = Innovations_rep(Σ_ss, K_ss, D, F)

    T = 100
    state_IRF_original, signal_IRF_original = compute_IRF(A, B, D, F, T)
    income_IRF_original_1 = np.cumsum(signal_IRF_original[0,:,0]) * 100 # IRF to the first shock
    income_IRF_original_2 = np.cumsum(signal_IRF_original[1,:,0]) * 100 # IRF to the second shock

    state_IRF_KF, signal_IRF_KF = compute_IRF(A, B_bar_ss, D, F_bar_ss, T)
    income_IRF_KF = np.cumsum(signal_IRF_KF[0,:,0]) * 100

    plt.plot(income_IRF_original_1,color='b', lw=0.8, alpha=0.8, label = "$W_1$")
    plt.plot(income_IRF_original_2,color='r', lw=0.8, alpha=0.8, label = "$W_2$")
    plt.plot(income_IRF_KF,color='g', lw=0.8, alpha=0.8, label = "$\overline{W}$")
    plt.legend()
    plt.xlabel("Quarters")
    plt.title("Impulse Responses of $\log(Y_t)$")
#    plt.savefig("IncomeIRF", dpi = 300)
    plt.show()

def plot_figure_2():

    a = .00663 #risk free capital growth rate
    g = .00373 # mean of y_growth in 0th order
    # γ = 10
    # ξ = 1/(γ - 1)
    σ_c = np.array([0.0048197574, 0.0000379813])
    σ_c_norm = np.linalg.norm(σ_c)

    T = 11
    γ = np.arange(1,T+1)

    delta_0 = np.zeros(T)
    delta_1 = np.zeros(T)

    for i in range(T):
        delta_0[i] = a - g
        delta_1[i] = a - g + σ_c_norm**2 * (γ[i]-1)

    plt.plot(γ-1, delta_0, color='b', lw=0.8, alpha=0.8, label = "Order 0")
    plt.plot(γ-1, delta_1, color='r', lw=0.8, alpha=0.8, label = "Order 1")
    plt.fill_between(γ-1, delta_1, delta_0, color = 'yellow', alpha = 0.2, label = 'Adjustment for precaution')
    plt.legend()
    plt.xlabel("$γ-1$")
    plt.ylabel("$\delta$")
    plt.tight_layout()
#    plt.savefig("delta", dpi = 300)
    plt.show()
