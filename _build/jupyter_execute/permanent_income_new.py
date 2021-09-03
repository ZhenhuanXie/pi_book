#!/usr/bin/env python
# coding: utf-8

# # Stochastic Growth and Long-Run Risk
# 
# ## Overview
# 
# __This notebook displays computational results from solving a permanent income model using small noise expansion method.This model is a version of Friedman's permanent income model, and habit persistence preferences, motivated by the work of Hansen, Sargent and Tallarini, is considered in computation.__
# 
# __This notebook is mainly for illustration of model specific results. If interested in more general computation details, users can refer to the Computational Details notebook accompanying this notebook.__

# ## 1 The Model
# ### 1.1 Preference: Recursive Utility with Habit Persistence
# We introduce $U_t$ via a CES aggregator of current consumption $C_t$ and a household stock variable $H_t$. $H_t$, which is a geometrically weighted average of current and past consumptions and the initial $H_0$, can be interpreted either as habits or as durable goods.
# 
# A representative household ranks $\{U_t: t\geq 0\}$ processes with a utility functional $\{V_t: t\geq 0\}$ processes. We also introduce an $\{R_t: t\geq 0\}$ process as a risk adjusted version of  $\{V_t: t\geq 0\}$, called a certainty equivalent. $V_t$, $R_t$, $U_t$ and $H_t$ are defined via the following recursion:
# 
# \begin{equation}
# V_t = \left[(1-\beta)U_t^{1-\rho}+\beta R_t^{1-\rho}\right]^{\frac{1}{1-\rho}} \tag{1'}
# \end{equation}
# 
# In the special case of $\rho = 1$, (1') becomes:
# 
# \begin{equation}
# V_t = U_t^{1-\beta}R_t^{\beta}
# \end{equation}
# 
# \begin{equation}
# R_t = \mathbb{E}\left[V_{t+1}^{1-\gamma} \mid {\mathfrak F}_t\right]^{\frac{1}{1-\gamma}} \tag{2'}
# \end{equation}
# 
# \begin{equation}
# U_t = \left[(1-\alpha)C_t^{1-\epsilon}+\alpha H_t^{1-\epsilon}\right]^{\frac{1}{1-\epsilon}} \tag{3'}
# \end{equation}
# 
# In the special case of $\epsilon = 1$, (3') becomes:
# \begin{equation}
# U_t = C_t^{1-\alpha} H_t^\alpha
# \end{equation}
# 
# \begin{equation}
# H_{t+1}  = \chi H_t + (1-\chi) C_t \tag{4'}
# \end{equation}
# 
# Stochastic growth in the income process (will be introduced shortly) makes it natural to divide every variable in equations (1)-(4) by $Y_t$ to form a **balanced growth version**: 
# 
# \begin{equation}
# \frac{V_t}{Y_t} = \left[(1-\beta)\left(\frac{U_t}{Y_t}\right)^{1-\rho}+\beta\left(\frac{R_t}{Y_t}\right)^{1-\rho}\right]^{\frac{1}{1-\rho}} \tag{1}
# \end{equation}
# 
# In the special case of $\rho = 1$, (1) becomes:
# 
# \begin{equation}
# \frac{V_t}{Y_t} = \left(\frac{U_t}{Y_t}\right)^{1-\beta} \left(\frac{R_t}{Y_t}\right)^{\beta}
# \end{equation}
# 
# \begin{equation}
# \frac{R_t}{Y_t} = \mathbb{E}\left[\left(\frac{V_{t+1}}{Y_{t+1}}\frac{Y_{t+1}}{Y_t}\right)^{1-\gamma} | {\mathfrak F}_t\right]^{\frac{1}{1-\gamma}} \tag{2}
# \end{equation}
# 
# \begin{equation}
# \frac{U_t}{Y_t} = \left[(1-\alpha)\left(\frac{C_t}{Y_t}\right)^{1-\epsilon}+\alpha \left(\frac{H_t}{Y_t}\right)^{1-\epsilon}\right]^{\frac{1}{1-\epsilon}} \tag{3}
# \end{equation}
# 
# In the special case of $\epsilon = 1$, (3) becomes:
# \begin{equation}
# \frac{U_t}{Y_t} = \left(\frac{C_t}{Y_t}\right)^{1-\alpha} \left(\frac{H_t}{Y_t}\right)^{\alpha} 
# \end{equation}
# 
# \begin{equation}
# \frac{H_{t+1}}{Y_{t+1}}\frac{Y_{t+1}}{Y_t}  = \chi \frac{H_t}{Y_t} + (1-\chi) \frac{C_t}{Y_t} \tag{4}
# \end{equation}
# 
# 
# 
# The reciprocal of the parameter $\rho$ describes the consumer's attitudes toward intertemporal substitution, while the parameter $\gamma$ describes the consumer's attitudes toward risk.

# ### 1.2 Technology: AK with Non-Financial Income
# 
# We construct a nonlinear version of a permanent income technology in the spirit of Hansen et al. (1999) and Hansen and Sargent (2013, ch. 11) that assumes the consumer's non-financial income process $\left\{Y_t\right\}$ is an exogenous multiplicative functional.
# 
# \begin{equation}
# K_{t+1} - K_t +C_t = {\sf a} K_t + Y_t \tag{5'}
# \end{equation}
# 
# balanced growth version of (5'):
# 
# \begin{equation}
# \frac{K_{t+1}}{Y_{t+1}} \frac{Y_{t+1}}{Y_t} -\frac{K_t}{Y_t} + \frac{C_t}{Y_t} = {\sf a} \frac{K_t}{Y_t} + 1 \tag{5}
# \end{equation}
# 
# We use this to define a model variable "scaled gross investment":
# 
# \begin{equation*}
# \frac{I_{t}}{Y_t} = \frac{K_{t+1} - K_t}{Y_t}
# \end{equation*}
# 
# 
# $\left\{Y_t\right\}$ has two components $Z_{1,t}$ and $Z_{2,t}$, and they follow the recursion
# 
# \begin{equation}
# \log Y_{t+1} - \log Y_t = D{Z_t} + FW_{t+1} + {\sf{g}} \tag{6}
# \end{equation}
# 
# \begin{equation}
# Z_{t+1} = AZ_t + BW_{t+1} \tag{7}
# \end{equation}
# 
# 
# 
# 
# $Z_t$ is a $3 \times 1$ vector 
# \begin{equation*}
# Z_t = \left[Z_{1,t}, Z_{2,t}, Z_{2,t-1}\right]^{\prime}
# \end{equation*}
# 
# and $W_{t+1}$ is a $2 \times 1$ vector 
# \begin{equation*}
# W_{t+1} = \left[W_{1,t+1}, W_{2,t+1}\right]^{\prime}
# \end{equation*}
# 
# 
# where $W_{1,t+1}$ and $W_{2,t+1}$ are shocks to $Z_{1,t+1}$ and $Z_{2,t+1}$, respectively, and $W_{t+1}$  follows a standardized multivariate  normal distribution.
# 
# We assume the following parameter values originally estimated by Hansen et al. (1999)
# 
# \begin{equation}
# \log(\frac{Y_{t+1}}{Y_t}) = .01(Z_{1,t+1}+Z_{2,t+1}-Z_{2,t}) = .01\left(\begin{bmatrix} .704 & 0 & -.154\end{bmatrix} \begin{bmatrix} Z_{1,t}\\Z_{2,t}\\Z_{2,t-1}\end{bmatrix} + \begin{bmatrix}.144 & .206\end{bmatrix} \begin{bmatrix}W_{1,t+1}\\W_{2,t+1} \end{bmatrix}\right) + .00373
# \end{equation}
# 
# \begin{equation}
# \begin{bmatrix}Z_{1,t+1}\\Z_{2,t+1}\\Z_{2,t}\end{bmatrix} = \begin{bmatrix}.704 & 0 & 0 \\0 & 1 & -.154\\ 0 & 1 & 0\end{bmatrix} \begin{bmatrix}Z_{1,t}\\Z_{2,t}\\Z_{2,t-1}\end{bmatrix} + \begin{bmatrix}.144 & 0 \\0 & .206 \\ 0 & 0\end{bmatrix} \begin{bmatrix} W_{1,t+1}\\ W_{2,t+1}\end{bmatrix}
# \end{equation}
# 
# 
# 
# $W_{1,t+1}$ and $W_{2,t+1}$ play different roles in the non-financial income process. $W_{1,t+1}$ has a permanent effect on income, while $W_{2,t+1}$ only has a transient effect. This can be seen by computing and plotting the **impulse response function** of $\log Y_t$ to each shock. Note that the responses are multiplied by 100 to reflect percentage responses.

# In[1]:


# Income IRF graph
from demonstration import plot_figure_1
plot_figure_1()


# ### 1.3 Stochastic Discount Factor; FOC on investment
# <font color='blue'> New marginal utilituy notations in PI model notes: We only has the special case $\rho = 1$ so far, so I haven't officially adapted to the new version.
# 
# \begin{equation*}
#     MS_{t+1} = (1-\beta)\alpha U_{t+1}^{\epsilon-1} H_{t+1}^{-\epsilon}
# \end{equation*}
#     
# \begin{equation*}
#     MH_t = \beta \chi \mathbb{E}\left[\left(\frac{V_{t+1}}{R_t}\right)^{1-\gamma} MH_{t+1} \Biggl| {\mathfrak F}_t \right] + \beta  \mathbb{E}\left[\left(\frac{V_{t+1}}{R_t}\right)^{1-\gamma} MS_{t+1} \Biggl| {\mathfrak F}_t \right]
# \end{equation*}
#     
# \begin{equation*}
#     MC_t =  (1-\beta)\alpha U_t^{\epsilon-1} C_t^{-\epsilon} + (1-\chi)MH_t
# \end{equation*}
#     
# NB: The "new" "$MH_t$" in this new notation is the discounted time t conditional expectation of the $MH_{t+1}$ in the previous notation. I have proved that **they are algebraically equivalent** in the $\rho = 1$ special case.
# </font>
# 
# 
# SDF increment in units of $U_t$:
# \begin{equation*}
# \widetilde{\frac{S_{t+1}}{S_t}} = \beta \left(\frac{V_{t+1}}{R_t}\right)^{1-\gamma} \left(\frac{V_{t+1}}{R_t}\right)^{\rho-1} \left(\frac{U_{t+1}}{U_t}\right)^{-\rho}
# \end{equation*}  
# 
# Note that the $ \left(\frac{V_{t+1}}{R_t}\right)^{1-\gamma}$ term is written separately. This is because it is a random variable with mean 1 conditioned on time $t$ information. Therefore, it represents a **change of probability measure** whenever we take a mathematical expectation. Specifically, $W_{t+1}$  follows a standardized multivariate  normal distribution under the "original" probability measure, but after taking into account of the change of measure, the distribution of $W_{t+1}$ is different. <font color='red'>What differences? How many details shall we provide.</font> Therefore, the change of measure alters the structure of terms in which $W_{t+1}$ is involved.
# 
# We are more interested in viewing $C_t$ rather than $U_t$ as the numeraire. This leads us to introduce two additional equations in which enduring effects of consumption at $t$ come into play. These equations in effect pin down two marginal rates of substitution, $\frac{MC_t}{MU_t}$ and $\frac{MH_t}{MU_t}$. 
# 
# $\frac{MC_t}{MU_t}$ has different specificiations depending on whether habit is "external (externality is ignored by the consumer)" or "internal (externality is internalized by the consumer)".
# 
# 1. External:
# \begin{equation}
# \frac{MC_t}{MU_t} = (1-\alpha)\left( \frac{U_t}{C_t} \right)^\epsilon \tag{8}
# \end{equation}
# 
# 2. Internal:
# \begin{equation}
# \frac{MC_t}{MU_t} = (1-\alpha)\left( \frac{U_t}{C_t} \right)^\epsilon + (1-\chi) \mathbb{E}\left[\widetilde{\frac{S_{t+1}}{S_t}}\frac{MH_{t+1}}{MU_{t+1}} \Biggl| {\mathfrak F}_t\right] \tag{8-1}
# \end{equation}
# where $\frac{MH_t}{MU_t}$ satisfies:
# \begin{equation}
# \frac{MH_t}{MU_t} = \alpha\left( \frac{U_t}{H_t} \right)^\epsilon + \chi \mathbb{E}\left[\widetilde{\frac{S_{t+1}}{S_t}}\frac{MH_{t+1}}{MU_{t+1}} \Biggl| {\mathfrak F}_t \right] \tag{8-2}
# \end{equation}
# 
# Then we have SDF increment in units of $C_t$:
# \begin{equation*}
# \frac{S_{t+1}}{S_t} = \widetilde{\left(\frac{S_{t+1}}{S_t}\right)}\left(\frac{{MC_{t+1}}/{MU_{t+1}}}{{MC_t}/{MU_t}}\right)
# \end{equation*}
# 
# FOC on investment:
# 
# \begin{equation*}
# \log\mathbb{E}\left[\left(1+ {\sf a}\right)\frac{S_{t+1}}{S_t}\Biggl| {\mathfrak F}_t\right] = 0 \label{foc} \tag{9}
# \end{equation*}
# 
# ### 1.4 Summary
# 
# The permanent income model that we study here consists of equations (1) to (4) that describe the representative consumer’s **preferences**, equation (8) or (8'-a&b) that describes **habit persistence**, equation (9) that describes the **FOC on investment**, equation (5) that restricts **feasibility**, (6) and (7) that describe the evolution of the consumer’s **non-financial income process** $\left\{Y_t\right\}$.

# Some HTML code
# <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.css" integrity="sha512-5A8nwdMOWrSz20fDsjczgUidUBR8liPYU+WymTZP1lmY9G6Oc7HlZv156XqnsgNUzTyMefFTcsFH/tnJE/+xBg==" crossorigin="anonymous" />
# <script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"></script>
# 
# <script type="text/x-thebe-config">
#   {
#     requestKernel: true,
#     binderOptions: {
#       repo: "matplotlib/ipympl",
#       ref: "0.6.1",
#       repoProvider: "github",
#     },
#   }
# </script>
# <script src="https://unpkg.com/thebe@latest/lib/index.js"></script>
# 
# <button id="activateButton" style="width: 120px; height: 40px; font-size: 1.5em;">
#   Activate
# </button>
# <script>
# var bootstrapThebe = function() {
#     thebelab.bootstrap();
# }
# document.querySelector("#activateButton").addEventListener('click', bootstrapThebe)
# </script>

# <pre data-executable="true" data-language="python">

# In[2]:


import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,10)

def sine_func(x, w, amp):
    return amp*np.sin(w*x)

@widgets.interact(w=(0, 4, 0.25), amp=(0, 4, .1))
def update(w = 1, amp = 1):
    plt.clf()
    plt.ylim(-4, 4)
    plt.plot(x, sine_func(x, w, amp))


# </pre>

# ## 2 Computation using ExpansionSuite Code
# This section serves as a demonstration on how to use ExpansionSuite code to solve recursive models. Users are given the flexibility of using parameters different from the default setting. Users simply have to enter the parameters in the user interface in section 2.1 below. Following the user interface, we also provide a detailed instruction on using the ExpansionSuite Code in section 2.2, in case a user may want to use the code elsewhere.
# 
# ### 2.1 User Interface

# In[3]:


from jupyterWidgets_pi import *
from plotting_module import *

import warnings
warnings.filterwarnings('ignore')

display(fixed_params_Panel)


# ### 2.1.1 Simulate paths using parameters given above
# 
# Users are able to select multiple variables of interest to simulate, by holding shift and/or ctrl (or command) pressed and then mouse click variable names. Please always press `Update parameters` first, and then press `Run simulation`.
# 
# The dashed curves indicate deterministic growth path (with both shocks being 0 at every period), and the solid curves indicate stochastic growth path (with randomly generated standard normal shock at every period).

# In[4]:


# Read key variables
γ = gamma.value
χ = chi.value
α = alpha.value
ϵ = epsilon.value
T = timeHorizon.value
    
a = ta.value
g = gy.value

A_1 = tA_1.value
A_21 = tA_21.value
A_22 = tA_22.value
B_11 = tB_11.value
B_22 = tB_22.value
A_x = np.array([[A_1, 0, 0],[0,A_21,A_22],[0,1,0]])
B_x = np.array([[B_11,0],[0,B_22],[0,0]])

KoverY_ss = K_Y_ss.value


# In[5]:


if habit.value == 1:
    display(simulate_box_external_habit_run)
elif habit.value == 2:
    display(simulate_box_internal_habit_run)


# In[6]:


if habit.value == 1:
    all_var_names = ['V/Y', 'R/Y', 'U/Y', 'C/Y', 'MC/MU', 'I/Y',                            'H/Y', 'K/Y', 'Y_{t+1}/Y_t', 'Z1', 'Z2', 'Z2_lag']
    selected_index = [all_var_names.index(element) for element in simulate_external_habit.value]
elif habit.value == 2:
    all_var_names = ['V/Y', 'R/Y', 'U/Y', 'C/Y', 'MH/MU', 'MC/MU', 'I/Y',                            'H/Y', 'K/Y', 'Y_{t+1}/Y_t', 'Z1', 'Z2', 'Z2_lag']
    selected_index = [all_var_names.index(element) for element in simulate_internal_habit.value]
    
fig, ax = plot_simulation(int(order.value), T, [np.float(r) for r in rhos.value.split(',')],                          γ, χ, α, ϵ, a, g, A_x, B_x, habit.value, KoverY_ss, selected_index)
plt.tight_layout()
plt.show()


# ### 2.1.2 Slide over a specific parameter of interest

# In[7]:


display(slider_box_run)


# In[8]:


# Generate slider variables
#slider_vars = np.arange(slider_min.value,slider_max.value,slider_step.value)
slider_vars = list(np.arange(slider_min.value,slider_max.value,slider_step.value)) + [slider_max.value]

if slider_var.value == 'γ':
    γ = slider_vars
elif slider_var.value == 'χ':
    χ = slider_vars
elif slider_var.value == 'α':
    α = slider_vars
elif slider_var.value == 'ϵ':
    ϵ = slider_vars


if conf_int.value == 0:
    conf_interval = None
else:
    conf_interval = conf_int.value

    
fig1, solved_models = plot_impulse_pi([np.float(r) for r in rhos.value.split(',')], T, order.value, γ, slider_var.value, habit.value, χ, α, ϵ,a,g,A_x,B_x,
                 KoverY_ss, shock = int(shock.value), title = plotName.value+", Shock {}")

fig1 = go.FigureWidget(fig1)
iplot(fig1)


# ### 2.2 Examples of Computation using ExpansionSuite Code
# 
# This section serves as a demonstration on how to use ExpansionSuite code to solve recursive models.
# 
# We start by assigning values to model parameters and defining equilibrium conditions. We decide to include the following variables as $X_t$ and $X_{t+1}$ in expansion.
# 
# For the **internal habit** specification, the first **6** variables in the table below are considered as **jump variables**, and the last **6** are considered as **state variables**. 
# 
# For the **external habit** specification, we don't have $\frac{MH_t}{MU_t}$ in the model, and all the other variables are still there, so there will be **5 controls and 6 states**. <font color='red'>LPH indicated that he didn't like the naming of "control" and "state". There will be a new terminology in the future.</font>
# 
# To avoid any confusion, the table below also provides the name of variables in the code to connect them with their math expressions.
# 
# | Python Index | Code name | Corresponding $X_t$ variable| Code name | Corresponding $X_{t+1}$ Variable| Category |
# |:-:|:---------:|:---------:|:---------:|:---------:|:---------:|
# |0|`vmy_t`|$\log\left(\frac{V_t}{Y_t}\right)$|`vmy_tp1`|$\log\left(\frac{V_{t+1}}{Y_{t+1}}\right)$ | jump
# |1|`rmy_t`|$\log\left(\frac{R_t}{Y_t}\right)$|`rmy_tp1`|$\log\left(\frac{R_{t+1}}{Y_{t+1}}\right)$ | jump
# |2|`umy_t`|$\log\left(\frac{U_t}{Y_t}\right)$|`umy_tp1`|$\log\left(\frac{U_{t+1}}{Y_{t+1}}\right)$ | jump
# |3|`cmy_t`|$\log\left(\frac{C_t}{Y_t}\right)$|`cmy_tp1`|$\log\left(\frac{C_{t+1}}{Y_{t+1}}\right)$ | jump
# |4|`mhmu_t`|$\log\left(\frac{MH_t}{MU_t}\right)$|`mhmu_tp1`|$\log\left(\frac{MH_{t+1}}{MU_{t+1}}\right)$| jump
# |5|`mcmu_t`|$\log\left(\frac{MC_t}{MU_t}\right)$|`mcmu_tp1`|$\log\left(\frac{MC_{t+1}}{MU_{t+1}}\right)$| jump
# |6|`IoverY_t`|$\frac{K_{t+1}-K_{t}}{Y_{t}}$|`IoverY_tp1`|$\frac{K_{t+2}-K_{t+1}}{Y_{t+1}}$| jump
# |7|`hmy_t`|$\log\left(\frac{H_t}{Y_t}\right)$|`hmy_tp1`|$\log\left(\frac{H_{t+1}}{Y_{t+1}}\right)$| state
# |8|`KoverY_t`|$\frac{K_t}{Y_t}$|`KoverY_tp1`|$\frac{K_{t+1}}{Y_{t+1}}$| state
# |9|`gy_t`|$\log\left(\frac{Y_t}{Y_{t-1}}\right)$|`gy_tp1`| $\log\left(\frac{Y_{t+1}}{Y_t}\right)$| state
# |10|`z1_t`|$Z_{1,t}$|`z1_tp1`|$Z_{1,t+1}$| state
# |11|`z2_t`|$Z_{2,t}$|`z2_tp1`|$Z_{2,t+1}$| state
# |12|`z2l_t`|$Z_{2,t-1}$|`z2l_tp1`|$Z_{2,t}$| state
# 
# Note that $KoverY_t \equiv \frac{K_t}{Y_t}$ is approximated as its level rather than logarithm in this model, because we intend to allow for negative capital (borrowing) in the evolution of $\frac{K_t}{Y_t}$, as actually happens in so-called open economy models.

# ### User Input 1: Equilibrium Conditions

# In[9]:


import numpy as np
import autograd.numpy as anp

# Model parameters
γ = 10.
ρ = 2./3

χ = 0.9
α = 0.9
ϵ = 10.

a = .00663 # risk free growth rate of capital
g = .00373 # deterministic growth trend of income
A = np.array([[.704, 0, 0],[0, 1, -.154],[0,1,0]])
B = np.array([[0.144,0],[0,0.206],[0,0]])

internal = True #True: internal habit; False: external habit.

KoverY_ss = 0.

args = (γ, ρ, χ, α, ϵ, a, g, A, B, internal, KoverY_ss)

'''
f=[f1,f2] which satisfy E[Mf1] = 0 and f2=0.

'''
def eq_cond(X_t, X_tp1, W_tp1, q, *args):
    # Parameters for the model
    γ, ρ, χ, α, ϵ, a, g, A, B, internal, _ = args

    # Variables:
    # log V_t/Y_t, log R_t/Y_t, log U_t/Y_t, log C_t/Y_t,log MH_t/MU_t, log MC_t/MU_t, I_t/Y_t - JUMP VARIABLES
    # log H_t/Y_t, K_t/Y_t, log(Y_t/Y_{t-1}), Z_{1,t}, Z_{2,t}, Z_{2,t-1} - STATE VARIABLES
    if internal:
        vmy_t, rmy_t, umy_t, cmy_t, mhmu_t,mcmu_t, IoverY_t,             hmy_t, KoverY_t, gy_t, z1_t, z2_t, z2l_t= X_t.ravel()
        vmy_tp1, rmy_tp1, umy_tp1, cmy_tp1, mhmu_tp1, mcmu_tp1, IoverY_tp1,             hmy_tp1, KoverY_tp1, gy_tp1, z1_tp1, z2_tp1, z2l_tp1= X_tp1.ravel()
    else:
        vmy_t, rmy_t, umy_t, cmy_t, mcmu_t, IoverY_t,             hmy_t, KoverY_t, gy_t, z1_t, z2_t, z2l_t= X_t.ravel()
        vmy_tp1, rmy_tp1, umy_tp1, cmy_tp1, mcmu_tp1, IoverY_tp1,             hmy_tp1, KoverY_tp1, gy_tp1, z1_tp1, z2_tp1, z2l_tp1= X_tp1.ravel()

    # Exogenous states (stacked together)
    Z_t = anp.array([z1_t, z2_t, z2l_t])
    Z_tp1 = anp.array([z1_tp1, z2_tp1, z2l_tp1])

    # log SDF in units of U, EXCLUDING the change of measure term
    β = anp.exp(g * ρ - anp.log(1 + a))
    sdf_u = anp.log(β) + (ρ - 1) * (vmy_tp1 + gy_tp1 - rmy_t) - ρ * (umy_tp1 + gy_tp1 - umy_t)
    sdf_ex = sdf_u + mcmu_tp1 - mcmu_t

    # Eq0: Change of measure evaluated at γ=0. --- (2) in Section 1
    # THIS CHANGE OF MEASURE EQUATION MUST BE THE FIRST EQUATION. Order of others doesn't matter.
    m = vmy_tp1 + gy_tp1 - rmy_t
    # Eq1: Recursive utility --- (1) in Section 1
    if ρ == 1.:
        res_1 = anp.exp((1-β)*umy_t) * anp.exp(β*rmy_t) - anp.exp(vmy_t)
    else:
        res_1 = (1-β)*anp.exp((1-ρ)*(umy_t)) + β*anp.exp((1-ρ)*(rmy_t)) - anp.exp((1-ρ)*(vmy_t))
    # Eq2: Utility function --- (3) in Section 1
    if ϵ == 1.:
        res_2 = anp.exp((1-α)*cmy_t) * anp.exp(α*hmy_t) - anp.exp(umy_t)
    else:
        res_2 = anp.log((1-α)*anp.exp((1-ϵ)*cmy_t)+α*anp.exp((1-ϵ)*hmy_t))/(1-ϵ) - (umy_t)
    # Eq3: FOC on investment --- (9) in Section 1
    res_3 = anp.log(1+a) + sdf_ex
    if internal:
        # Eq4: MC/MU --- (8-a) in Section 1
        res_4 = (1-α)*anp.exp(ϵ*(umy_t-cmy_t)) + (1-χ)*anp.exp(sdf_u+mhmu_tp1) - anp.exp(mcmu_t)
        # Eq5: MH/MU --- (8-b) in Section 1
        res_5 = α*anp.exp(ϵ*(umy_t-hmy_t)) + χ*anp.exp(sdf_u+mhmu_tp1) - anp.exp(mhmu_t)
    else:
        # Eq4: MC/MU --- (8) in Section 1. No MH/MU when external.
        res_4 = (1-α)*anp.exp(ϵ*(umy_t-cmy_t)) - anp.exp(mcmu_t)
    # Eq6: Capital growth --- (5-b) in Section 1
    res_6 = KoverY_tp1 * anp.exp(gy_tp1) + anp.exp(cmy_t) - ((1+a)*KoverY_t + 1)
    # Eq7: Habit evolution --- (4) in Section 1
    res_7 = anp.exp(hmy_tp1+gy_tp1) - χ*anp.exp(hmy_t) - (1-χ)*anp.exp(cmy_t)
    # Eq8: Nonfinancial income process --- (6) in Section 1
    res_8 = gy_tp1 - .01 * (z1_tp1 + z2_tp1 - z2_t) - g
    # Eq9-11: State process  --- (7) in Section 1
    res_9 = (A@Z_t + B@W_tp1 - Z_tp1)[0]
    res_10 = (A@Z_t + B@W_tp1 - Z_tp1)[1]
    res_11 = (A@Z_t + B@W_tp1 - Z_tp1)[2]
    
    res_12 = KoverY_tp1 * anp.exp(gy_tp1) - KoverY_t - IoverY_t
    
    if internal:
        return anp.array([m, res_1,res_2,res_3,res_4,res_5,res_6,res_7,res_12, res_8,res_9, res_10, res_11])
    else:
        return anp.array([m, res_1,res_2,res_3,res_4,res_6,res_7,res_12, res_8,res_9, res_10, res_11])


# ### User Input 2: Steady States
# After having parameters and equilibrium conditions defined, we also need to define a function that returns the deterministic steady state (0th order expansion) value of model variables.
# 
# For this model, we have a free initial condition that can be imposed on $\frac{K_0^0}{Y_0^0}$, i.e. there is a family of steady states indexed by $\frac{K_0^0}{Y_0^0}$ subject to the restriction $\frac{C_0^0}{Y_0^0} > 0$ (otherwise we cannot take the logarithm of it).
# 
# In the following example, we impose $\frac{K_0^0}{Y_0^0} = 0$ as its deterministic steady state value. Once this is imposed, we can solve for the steady state value of others with some effort in algebra.

# In[10]:


def ss_func(*args):

    '''
    User-defined function to calculate steady states.

    Alternatively, users can simply provide a hard-coded np.array for steady states.

    '''
    
    # Extra parameters for the model
    γ, ρ, χ, α, ϵ, a, g, A, B, internal, KoverY_ss = args
    
    KoverY = KoverY_ss # This is a condition that we can freely impose
    β = np.exp(g * ρ - np.log(1+a)) # From FOC (9)
    gy = g # From (6). This is just the deterministic growth trend of income

    # Starting from the steady state of Y_{t+1}/Y_{t} and K/Y,
    # we can solve the steady state of other variables by hand:
    cmy = np.log(((1+a) - np.exp(g)) * KoverY + 1) # From (5-b)
    hmy = np.log(1-χ) + cmy - np.log((np.exp(g) - χ)) # From (4)
    if ϵ == 1.:
        umy = (1-α)*cmy + α*hmy # From (3) special case
    else:
        umy = np.log((1-α)*np.exp((1-ϵ)*cmy) + α * np.exp((1-ϵ)*hmy)) / (1-ϵ) # From (3)

    λ = β * np.exp((1-ρ)*g)
    
    if ρ == 1.:
        vmy = (1-β)*umy + β*rmy # From (1) special case
    else:
        vmy = (np.log((1-β)*np.exp((1-ρ)*umy)) - np.log(1-λ)) / (1-ρ) # From (1)
    rmy = vmy + gy # From (2)
    
    IoverY = KoverY*(np.exp(gy)-1)
    
    # steady state of Z components (=0, by stationarity of the two AR process)
    z1 = 0.
    z2 = 0.
    z2l = 0.
    
    if internal:
        mhmu = np.log(α) + ϵ * (umy - hmy) - np.log(1 - χ * β * np.exp(-g * ρ)) # From (8'-b)
        mcmu = np.log((1-α) * np.exp(ϵ * (umy - cmy)) + (1 - χ) * β * np.exp(-g * ρ) * np.exp(mhmu)) # From (8'-a)
        X_0 = np.array([vmy, rmy, umy, cmy, mhmu, mcmu, IoverY, hmy, KoverY, gy, z1, z2, z2l])
        
    else:
        mcmu = np.log((1-α) * np.exp(ϵ * (umy - cmy))) # From (8)
        X_0 = np.array([vmy, rmy, umy, cmy, mcmu, IoverY, hmy, KoverY, gy, z1, z2, z2l])
        
    return X_0


# ### Solving (Approximate) Equilibrium Law of Motion
# Once we have the above two pieces of user inputs at hand, we are ready to apply small-noise expansion method to solve for (approximate) the equilibrium law of motion of every variable in the model. Let's first try a 1st order approximation.

# In[11]:


from expansion import recursive_expansion

if internal:
    var_shape = (7,6,2) # (number of controls, number of states, number of shocks)
else:
    var_shape = (6,6,2)

modelSol = recursive_expansion(eq_cond=eq_cond,
                               ss=ss_func,
                               var_shape=var_shape, 
                               γ=args[0],
                               second_order=False,# False: only 1st order expansion; True: include 2nd order expansion
                               args=args)


# The approximate law of motion given by small-noise expansion method takes the form of a familiar **linear state-space system**: It represents $X_{t+1}$, **all variables** (including both control variables and state variables) at time $t+1$, only using **state variables** at time $t$ and **shock** at time $t+1$. 
# 
# If we partition $X_{t} = \begin{bmatrix} Y_t\\ Z_t\end{bmatrix}$ where $Y_t$ and $Z_t$ stand for time $t$ control and state variables respectively, the first order approximation of $X_{t+1} $ will take the form
# 
# \begin{align*}
#     Y_{t+1}^1 &= D Z_{t}^1 + F W_{t+1} + H\\
#     Z_{t+1}^1 &= A Z_{t}^1 + B W_{t+1} + C
# \end{align*}
# or
# \begin{equation*}
#     X_{t+1}^1 = \begin{bmatrix} D\\A \end{bmatrix} Z_t^1 + \begin{bmatrix} F\\B \end{bmatrix} W_{t+1} + \begin{bmatrix} H\\C \end{bmatrix}
# \end{equation*}
# 
# <!-- 
# The second order approximation of $X_{t+1}$ will have a bit complicated structure:
# 
# \begin{align*}
#     Y_{t+1}^2 &= D Z_{t}^1 + F W_{t+1} + H\\
#     Z_{t+1}^2 &= A Z_{t}^1 + B W_{t+1} + C
# \end{align*} -->
# 
# Let's print the first order coefficients out in the approximation that we just ran. You can easily tell what are the corresponding matrices $D$, $F$, $A$, $B$, $H$ and $C$. These coefficients are represented in a dictionary, so if you need the value of a specific coefficient matrix, you can conveniently extract it from the dictionary using the corresponding key 'c', 'w' or 'x'.
# 
# <font color='red'>Second order approximation takes a bit more complicated structure. I haven't written it down.</font>

# In[12]:


np.set_printoptions(precision=6,suppress=True)
modelSol.X1_tp1.coeffs # First order LOM coefficients


# With the law of motion at hand, it would be a piece of cake to perform tasks such as simulation, or compute important quantities that reflect the dynamics in the model, for example, impulse response functions. The ExpansionSuite code provides such functionalities.
# 
# ### Simulation
# Suppose we are interested in knowing how the model variable $\log(\frac{C_t}{Y_t})$ evolves over time if it follows the (approximate) equilibrium law of motion. We just need to generate shocks throughout the horizon that we are interested in (for example, 100 periods), and feed the shocks to the `simulate` method of `modelSol`.
# 
# In the cell below, we first try 0 shocks (i.e. examine the deterministic path) and then try bivariate standard normal shocks.

# In[13]:


T = 100 # time horizon for simulation
_, _, n_W = modelSol.var_shape # the number of shocks within this model
Ws_1 = np.zeros((T,n_W)) # "generate" zero shocks throughout the horizon T=100
sim_result_1 = modelSol.simulate(Ws_1) # feed the generated shocks to the simulate method

Ws_2 = np.random.multivariate_normal(np.zeros(n_W), np.eye(n_W), size = T) # generate N(0,I) shocks
sim_result_2 = modelSol.simulate(Ws_2) # feed the generated shocks to the simulate method

import matplotlib.pyplot as plt
# sim_result contains the simulated value of all 12 model variables over the specified horizon
# recall that we arranged log(C_t/Y_t) as the 4th control variable, and python index starts from 0
plt.plot(sim_result_1[:,3], 'r', lw = .8, alpha = .8, label = 'deterministic') 
plt.plot(sim_result_2[:,3], 'b', lw = .8, alpha = .8, label = 'stochastic')
plt.legend()
plt.title(r'$\log(\frac{C_t}{Y_t})$ simulation')
plt.show()


# It turns out that in equilibrium, the deterministic path of $\log(\frac{C_t}{Y_t})$ is $\log(\frac{C_t}{Y_t}) \equiv 0$, $\forall t$, and the stochastic path oscillate around the deterministic path. This implies that consumption and income have the same deterministic growth rate in equilibrium, which doesn't come as a surprise (as expected from the FOC on investment).

# ### Computing Impulse Response Functions
# You may remember this is not the first time that you see impusle response functions in this notebook: Previously, we computed and plotted the IRFs of income process to two shocks, without knowing small-noise expansion at all! At that time, we only made use of the law of motion that $Y_t$ and the two components follow to construct IRFs.
# 
# Now that we're using small-noise expasion method, we have included a bunch of more variables in the linear space-system, we are able to use the approximate law of motion to construct **"new" IRFs** of income, again to the two shocks. How will the "new" IRFs look like? We can directly use the `IRF` method of `modelSol` to obtain the results.

# In[14]:


states_IRF = []
controls_IRF = []
T = 100 # time horizon for IRF computation
for shock_index in [0,1]: # we have two shocks for this model
    states, controls = modelSol.IRF(T, shock_index)
    states_IRF.append(states)
    controls_IRF.append(controls)

plt.plot(np.cumsum(states_IRF[0][:,2])*100, color='b', lw=0.8, alpha=0.8, label = "$W_1$")
plt.plot(np.cumsum(states_IRF[1][:,2])*100, color='r', lw=0.8, alpha=0.8, label = "$W_2$")
plt.legend()
plt.xlabel("Quarters")
plt.title("Impulse Responses of $\log(Y_t)$")
# plt.savefig("IncomeIRF", dpi = 300)
plt.show()


# It turns out that we actually get the same IRFs as we did previously. This is because the approximate law of motion of $\log(\frac{Y_{t+1}}{Y_t})$ obtained using small-noise expansion is equivalent to its "true" law of motion that we have described in Section 1.2. 
# 
# We can verify this by simply inspecting the coefficients of $\log(\frac{Y_{t+1}}{Y_t})$ on **each time $t$ state variable** printed below. Remember, we have 6 control variables and 6 state variables in the model, and $\log(\frac{Y_{t+1}}{Y_t})$ is the 3rd state variable. Therefore, the relevant coefficients of $\log(\frac{Y_{t+1}}{Y_t})$ is on the 9th row.
# 
# The coefficients of $\log(\frac{Y_{t+1}}{Y_t})$ on $Z_{1,t}$, $Z_{2,t}$ and $Z_{2,t-1}$, $W_{1, t+1}$ and $W_{2, t+1}$ are .00704, 0, -.00154, .00144, .00206, respectively. These numbers are the same as in the "true" law of motion. The coefficients on other state variables are 0.

# In[15]:


modelSol.X1_tp1['x']


# In[16]:


modelSol.X1_tp1['w']


# ### 2.3 Special Issue: Endogenously Determined Subjective Discount Rate
# 
# Before moving forward, let's have another look at the FOC (9). For compactness, we define $\delta = -\log\beta$. Essentially, FOC says
# 
# \begin{equation*}
# \log(1+{\sf a}) -\delta + \log\tilde{\mathbb{E}}\left[\left(\frac{V_{t+1}}{R_t}\right)^{\rho-1} \left(\frac{U_{t+1}}{U_t}\right)^{-\rho} \left(\frac{{MC_{t+1}}/{MU_{t+1}}}{{MC_t}/{MU_t}}\right) \Biggl| {\mathfrak F}_t\right] = 0 \label{foc_new} \tag{9'}
# \end{equation*}
# 
# where $\tilde{\mathbb{E}}[\cdotp | \mathfrak F_t]$ is the conditional expectation operator under the **altered probability measure**, i.e. taking into account the $\left(\frac{V_{t+1}}{R_t}\right)^{1-\gamma}$ term. Here $\sf a$ is an exogenous one period risk-free capital growth rate, thus the subjective discount rate $\delta$ is not exogenously given, but rather endogenously pinned down by \eqref{foc_new}. Our small noise expansion method gives the law of motion of model variables $X_{t+1}({\sf q}) = \psi[X_t({\sf q}), {\sf q}W_{t+1}, {\sf q}]$ , and if we plug it into \eqref{foc_new}, there will surely be a bunch of terms that involve $W_{t+1}$. Since the structure of $W_{t+1}$ under the altered probability measure differs as the order of expansion differs, it's clear that $\beta$ pinned down by \eqref{foc_new} in order 0, 1, 2 differs. Let's write them as $\delta^0$, $\delta^1$ and $\delta^2$. 
# 
# In the deterministic steady state (order 0), $\delta^0$ can be easily solved from \eqref{foc_new} because change of measure and expectation operator don't matter (no $W_{t+1}$ terms).
# 
# In order 1, an **additional constant term** needs to be added to \eqref{foc_new}, which equals $\delta^0 - \delta^1$. Without this constant term, the LHS of \eqref{foc_new} is obviously non-zero, and the approximated law of motion coming from expansion that can make the wrong FOC "hold" will be wrong (not the equilibria that we want).
# 
# Similarly, in order 2, an **additional constant term** needs to be added to \eqref{foc_new}, which equals $\delta^1 - \delta^2$.
# 
# <font color='red'>Technical details:</font> When introducing the additional free constant term, we also impose an additional restriction, to set the constant component that applies to $\frac{K_t}{Y_t}$ to zero. We make this latter adjustment because we want to set $\frac{K_0}{Y_0}$ as an initial condition and don’t want to adjust it later when we include higher order terms in an expansion.
# 
# <font color='red'>As in LPH's *PI Model Notes*,</font> we provide a simplified example where there's no habit/durable goods. It can be viewed as a special case of the preference described in section 1.1, by setting $\alpha = 0$, $\rho = 1$. In this special case, <font color='red'>LPH showed that</font>
# 
# In order 0 approximation,
# 
# \begin{equation*}
#     \delta^0 = {\sf a} - {\sf g}
# \end{equation*}
# 
# In order 1 approximation,
# 
# \begin{equation*}
#     \delta^1 = {\sf a} - {\sf g} - (1-\gamma) \sigma_c^1 \cdot \sigma_c^1
# \end{equation*}
# 
# The last term is an adjustment for precaution, which is exactly the "additional constant term" that should be included when we move from order 0 to order 1 expansion in this special case.

# In[17]:


# discount rate adjustment graph
from demonstration import plot_figure_2
plot_figure_2()


# ## References
# 
# [[1] Hansen, Lars Peter, Thomas J. Sargent, and Thomas D. Tallarini Jr. "Robust permanent income and pricing." Review of Economic studies (1999): 873-907.](http://larspeterhansen.org/wp-content/uploads/2016/10/Robust-Permanent-Income-and-Pricing.pdf)
# 
# [[2] Hansen, Lars Peter, and Thomas J. Sargent. Recursive models of dynamic linear economies. Princeton University Press, 2013.](http://larspeterhansen.org/wp-content/uploads/2016/10/mbook2.pdf)
# 
# [[3] Lombardo, Giovanni, and Harald Uhlig. "A Theory Of Pruning." International Economic Review 59, no. 4 (2018): 1825-1836.](https://onlinelibrary.wiley.com/doi/abs/10.1111/iere.12321)
# 
# [[4] Borovička, Jaroslav, and Lars Peter Hansen. "Examining macroeconomic models through the lens of asset pricing." Journal of Econometrics 183, no. 1 (2014): 67-90.](http://larspeterhansen.org/wp-content/uploads/2016/10/Examining-Macroeconomic-Models-through-the-Lens-of-Asset-Pricing.pdf)
