#!/usr/bin/env python
# coding: utf-8

# (model_construction)=
# 
# # Model Construction
# 
# ## Preference: Recursive Utility with Habit Persistence
# 
# We introduce $U_t$ via a CES aggregator of current consumption $C_t$ and a household stock variable $H_t$. $H_t$, which is a geometrically weighted average of current and past consumptions and the initial $H_0$, can be interpreted either as habits or as durable goods.
# 
# A representative household ranks $\{U_t: t\geq 0\}$ processes with a utility functional $\{V_t: t\geq 0\}$ processes. We also introduce an $\{R_t: t\geq 0\}$ process as a risk adjusted version of  $\{V_t: t\geq 0\}$, called a certainty equivalent. $V_t$, $R_t$, $U_t$ and $H_t$ are defined via the following recursion:
# 
# ```{math}
#     V_t = \left[(1-\beta)U_t^{1-\rho}+\beta R_t^{1-\rho}\right]^{\frac{1}{1-\rho}} 
# ```
# 
# \begin{equation}
# 	
# \end{equation}
# 
# In the special case of $\rho = 1$,
# 
# ```{math}
#     V_t = U_t^{1-\beta}R_t^{\beta} 
# ```
# 
# ```{math}
#     R_t = \mathbb{E}\left[V_{t+1}^{1-\gamma} \mid {\mathfrak F}_t\right]^{\frac{1}{1-\gamma}} 
# ```
# 
# ```{math}
#     U_t = \left[(1-\alpha)C_t^{1-\epsilon}+\alpha H_t^{1-\epsilon}\right]^{\frac{1}{1-\epsilon}} 
# ```
# 
# In the special case of $\epsilon = 1$,
# 
# ```{math}
#     U_t = C_t^{1-\alpha} H_t^\alpha
# ```
# 
# ```{math}
#     H_{t+1}  = \chi H_t + (1-\chi) C_t 
# ```
# 
# Stochastic growth in the income process (will be introduced shortly) makes it natural to divide every variable in the above equations by $Y_t$ to form a **balanced growth version**: 
# 
# ```{math}
# :label: V_balanced
#     \frac{V_t}{Y_t} = \left[(1-\beta)\left(\frac{U_t}{Y_t}\right)^{1-\rho}+\beta\left(\frac{R_t}{Y_t}\right)^{1-\rho}\right]^{\frac{1}{1-\rho}}
# ```
# 
# In the special case of $\rho = 1$, {eq}`V_balanced` becomes:
# 
# ```{math}
#     \frac{V_t}{Y_t} = \left(\frac{U_t}{Y_t}\right)^{1-\beta} \left(\frac{R_t}{Y_t}\right)^{\beta}
# ```
# 
# ```{math}
# :label: R_balanced
#     \frac{R_t}{Y_t} = \mathbb{E}\left[\left(\frac{V_{t+1}}{Y_{t+1}}\frac{Y_{t+1}}{Y_t}\right)^{1-\gamma} | {\mathfrak F}_t\right]^{\frac{1}{1-\gamma}}
# ```
# 
# ```{math}
# :label: U_balanced
#     \frac{U_t}{Y_t} = \left[(1-\alpha)\left(\frac{C_t}{Y_t}\right)^{1-\epsilon}+\alpha \left(\frac{H_t}{Y_t}\right)^{1-\epsilon}\right]^{\frac{1}{1-\epsilon}}
# ```
# 
# In the special case of $\epsilon = 1$, {eq}`U_balanced` becomes:
# 
# ```{math}
#     \frac{U_t}{Y_t} = \left(\frac{C_t}{Y_t}\right)^{1-\alpha} \left(\frac{H_t}{Y_t}\right)^{\alpha}
# ```
# 
# ```{math}
# :label: H_balanced
#     \frac{H_{t+1}}{Y_{t+1}}\frac{Y_{t+1}}{Y_t}  = \chi \frac{H_t}{Y_t} + (1-\chi) \frac{C_t}{Y_t}
# ```
# 
# The reciprocal of the parameter $\rho$ describes the consumer's attitudes toward intertemporal substitution, while the parameter $\gamma$ describes the consumer's attitudes toward risk.
# 
# ##  Technology: AK with Non-Financial Income
# 
# We construct a nonlinear version of a permanent income technology in the spirit of {cite:t}`hansen1999robust` and {cite:t}`hansen2013recursive` Hansen and Sargent (2013, ch. 11) that assumes the consumer's non-financial income process $\left\{Y_t\right\}$ is an exogenous multiplicative functional.
# 
# ```{math}
#     K_{t+1} - K_t +C_t = {\sf a} K_t + Y_t
# ```
# 
# balanced growth version:
# 
# ```{math}
# :label: K_balanced
#     \frac{K_{t+1}}{Y_{t+1}} \frac{Y_{t+1}}{Y_t} -\frac{K_t}{Y_t} + \frac{C_t}{Y_t} = {\sf a} \frac{K_t}{Y_t} + 1
# ```
# 
# We use this to define a model variable "scaled gross investment":
# 
# ```{math}
# :label: Investment
#     \frac{I_{t}}{Y_t} = \frac{K_{t+1} - K_t}{Y_t}
# ```
# 
# $\left\{Y_t\right\}$ has two components $Z_{1,t}$ and $Z_{2,t}$, and they follow the recursion:
# 
# ```{math}
# :label: Y_log_growth
#     \log Y_{t+1} - \log Y_t = D{Z_t} + FW_{t+1} + {\sf{g}}
# ```
# 
# ```{math}
# :label: exogenous
#     Z_{t+1} = AZ_t + BW_{t+1}
# ```
# 
# 
# $Z_t$ is a $3 \times 1$ vector 
# ```{math}
#     Z_t = \left[Z_{1,t}, Z_{2,t}, Z_{2,t-1}\right]^{\prime}
# ```
# 
# and $W_{t+1}$ is a $2 \times 1$ vector
# ```{math}
#     W_{t+1} = \left[W_{1,t+1}, W_{2,t+1}\right]^{\prime}
# ``` 
# 
# where $W_{1,t+1}$ and $W_{2,t+1}$ are shocks to $Z_{1,t+1}$ and $Z_{2,t+1}$, respectively, and $W_{t+1}$  follows a standardized multivariate normal distribution.
# 
# We assume the following parameter values originally estimated by {cite:t}`hansen1999robust`:
# 
# ```{math}
#     \log\left(\frac{Y_{t+1}}{Y_t}\right) = .01(Z_{1,t+1}+Z_{2,t+1}-Z_{2,t}) = .01\left(\begin{bmatrix} .704 & 0 & -.154\end{bmatrix} \begin{bmatrix} Z_{1,t}\\Z_{2,t}\\Z_{2,t-1}\end{bmatrix} + \begin{bmatrix}.144 & .206\end{bmatrix} \begin{bmatrix}W_{1,t+1}\\W_{2,t+1} \end{bmatrix}\right) + .00373
# ```
# ```{math}
#     \begin{bmatrix}Z_{1,t+1}\\Z_{2,t+1}\\Z_{2,t}\end{bmatrix} = \begin{bmatrix}.704 & 0 & 0 \\0 & 1 & -.154\\ 0 & 1 & 0\end{bmatrix} \begin{bmatrix}Z_{1,t}\\Z_{2,t}\\Z_{2,t-1}\end{bmatrix} + \begin{bmatrix}.144 & 0 \\0 & .206 \\ 0 & 0\end{bmatrix} \begin{bmatrix} W_{1,t+1}\\ W_{2,t+1}\end{bmatrix}
# ```
# 
# $W_{1,t+1}$ and $W_{2,t+1}$ play different roles in the non-financial income process. $W_{1,t+1}$ has a permanent effect on income, while $W_{2,t+1}$ only has a transient effect. This can be seen by computing and plotting the **impulse response function** of $\log Y_t$ to each shock. Note that the responses are multiplied by 100 to reflect percentage responses.

# In[1]:


# Income IRF graph
from demonstration import plot_figure_1
plot_figure_1()


# ## Stochastic Discount Factor; FOC on investment
# 
# SDF increment in units of $U_t$:
# ```{math}
#     \widetilde{\frac{S_{t+1}}{S_t}} = \beta \left(\frac{V_{t+1}}{R_t}\right)^{1-\gamma} \left(\frac{V_{t+1}}{R_t}\right)^{\rho-1} \left(\frac{U_{t+1}}{U_t}\right)^{-\rho}
# ``` 
# 
# Note that the $ \left(\frac{V_{t+1}}{R_t}\right)^{1-\gamma}$ term is written separately. This is because it is a random variable with mean 1 conditioned on time $t$ information. Therefore, it represents a **change of probability measure** whenever we take a mathematical expectation. Specifically, $W_{t+1}$  follows a standardized multivariate  normal distribution under the "original" probability measure, but after taking into account of the change of measure, the distribution of $W_{t+1}$ is different. <font color='red'>What differences? How many details shall we provide.</font> Therefore, the change of measure alters the structure of terms in which $W_{t+1}$ is involved.
# 
# We are more interested in viewing $C_t$ rather than $U_t$ as the numeraire. This leads us to introduce two additional equations in which enduring effects of consumption at $t$ come into play. These equations in effect pin down two marginal rates of substitution, $\frac{MC_t}{MU_t}$ and $\frac{MH_t}{MU_t}$. 
# 
# $\frac{MC_t}{MU_t}$ has different specificiations depending on whether habit is "external (externality is ignored by the consumer)" or "internal (externality is internalized by the consumer)".
# 
# - External:
# ```{math}
# :label: MCMU_external
#     \frac{MC_t}{MU_t} = (1-\alpha)\left( \frac{U_t}{C_t} \right)^\epsilon
# ``` 
# - Internal:
# ```{math}
# :label: MCMU_internal
#     \frac{MC_t}{MU_t} = (1-\alpha)\left( \frac{U_t}{C_t} \right)^\epsilon + (1-\chi) \mathbb{E}\left[\widetilde{\frac{S_{t+1}}{S_t}}\frac{MH_{t+1}}{MU_{t+1}} \Biggl| {\mathfrak F}_t\right]
# ``` 
# where $\frac{MH_t}{MU_t}$ satisfies:
# ```{math}
# :label: MHMU
#     \frac{MH_t}{MU_t} = \alpha\left( \frac{U_t}{H_t} \right)^\epsilon + \chi \mathbb{E}\left[\widetilde{\frac{S_{t+1}}{S_t}}\frac{MH_{t+1}}{MU_{t+1}} \Biggl| {\mathfrak F}_t \right]
# ``` 
# 
# Then we have SDF increment in units of $C_t$:
# 
# ```{math}
#     \frac{S_{t+1}}{S_t} = \widetilde{\left(\frac{S_{t+1}}{S_t}\right)}\left(\frac{{MC_{t+1}}/{MU_{t+1}}}{{MC_t}/{MU_t}}\right)
# ``` 
# 
# FOC on investment:
# 
# ```{math}
# :label: FOC
#     \log\mathbb{E}\left[\left(1+ {\sf a}\right)\frac{S_{t+1}}{S_t}\Biggl| {\mathfrak F}_t\right] = 0 \label{foc}
# ```
# 
# ## Summary
# 
# The permanent income model that we study here consists of equations {eq}`V_balanced` to {eq}`H_balanced` that describe the representative consumer’s **preferences**, equation {eq}`MCMU_external` (or {eq}`MCMU_internal` and {eq}`MHMU`) that describes **habit persistence**, equation {eq}`FOC` that describes the **FOC on investment**, equation {eq}`K_balanced` that restricts **feasibility**, {eq}`Y_log_growth` and {eq}`exogenous` that describe the evolution of the consumer’s **non-financial income process** $\left\{Y_t\right\}$.
