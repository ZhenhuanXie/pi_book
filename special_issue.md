---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(special_issue)=

# Special Issue
## Endogenously Determined Subjective Discount Rate

Before moving forward, let's have another look at the FOC {eq}`FOC`. For compactness, we define $\delta = -\log\beta$. Essentially, FOC says

```{math}
:label: FOC_detail
    \log(1+{\sf a}) -\delta + \log\tilde{\mathbb{E}}\left[\left(\frac{V_{t+1}}{R_t}\right)^{\rho-1} \left(\frac{U_{t+1}}{U_t}\right)^{-\rho} \left(\frac{{MC_{t+1}}/{MU_{t+1}}}{{MC_t}/{MU_t}}\right) \Biggl| {\mathfrak F}_t\right] = 0 \label{foc_new}
```

where $\tilde{\mathbb{E}}[\cdotp | \mathfrak F_t]$ is the conditional expectation operator under the **altered probability measure**, i.e. taking into account the $\left(\frac{V_{t+1}}{R_t}\right)^{1-\gamma}$ term. Here $\sf a$ is an exogenous one period risk-free capital growth rate, thus the subjective discount rate $\delta$ is not exogenously given, but rather endogenously pinned down by {eq}`FOC_detail`. Our small noise expansion method gives the law of motion of model variables $X_{t+1}({\sf q}) = \psi[X_t({\sf q}), {\sf q}W_{t+1}, {\sf q}]$ , and if we plug it into {eq}`FOC_detail`, there will surely be a bunch of terms that involve $W_{t+1}$. Since the structure of $W_{t+1}$ under the altered probability measure differs as the order of expansion differs, it's clear that $\beta$ pinned down by {eq}`FOC_detail` in order 0, 1, 2 differs. Let's write them as $\delta^0$, $\delta^1$ and $\delta^2$. 

In the deterministic steady state (order 0), $\delta^0$ can be easily solved from {eq}`FOC_detail` because change of measure and expectation operator don't matter (no $W_{t+1}$ terms).

In order 1, an **additional constant term** needs to be added to {eq}`FOC_detail`, which equals $\delta^0 - \delta^1$. Without this constant term, the LHS of {eq}`FOC_detail` is obviously non-zero, and the approximated law of motion coming from expansion that can make the wrong FOC "hold" will be wrong (not the equilibria that we want).

Similarly, in order 2, an **additional constant term** needs to be added to {eq}`FOC_detail`, which equals $\delta^1 - \delta^2$.

<font color='red'>Technical details:</font> When introducing the additional free constant term, we also impose an additional restriction, to set the constant component that applies to $\frac{K_t}{Y_t}$ to zero. We make this latter adjustment because we want to set $\frac{K_0}{Y_0}$ as an initial condition and don’t want to adjust it later when we include higher order terms in an expansion.

<font color='red'>As in LPH's *PI Model Notes*,</font> we provide a simplified example where there's no habit/durable goods. It can be viewed as a special case of the preference described in section 1.1, by setting $\alpha = 0$, $\rho = 1$. In this special case, <font color='red'>LPH showed that</font>

In order 0 approximation,

```{math}
    \delta^0 = {\sf a} - {\sf g}
```

In order 1 approximation,

```{math}
    \delta^1 = {\sf a} - {\sf g} - (1-\gamma) \sigma_c^1 \cdot \sigma_c^1
```

The last term is an adjustment for precaution, which is exactly the "additional constant term" that should be included when we move from order 0 to order 1 expansion in this special case.

```{code-cell} python3
# discount rate adjustment graph
from demonstration import plot_figure_2
plot_figure_2()
```


## Bibliography

```{bibliography} ../_bibliography/references.bib
```
