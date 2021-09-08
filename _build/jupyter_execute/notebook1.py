#!/usr/bin/env python
# coding: utf-8

# (user_interface)=
# # User Interface
# 
# This section serves as a demonstration on how to use ExpansionSuite code to solve recursive models. Users are given the flexibility of using parameters different from the default setting. Users simply have to enter the parameters in the user interface below. Following the user interface, we also provide a detailed instruction on using the ExpansionSuite Code in {ref}`code_details`, in case a user may want to use the code elsewhere.
# 
# <font color='red'>I changed the section reference to hyperlink. Alternatively, "[Section 2.2](code_details)"</font>

# In[1]:


from jupyterWidgets_pi import *
from plotting_module import *

import warnings
warnings.filterwarnings('ignore')

display(fixed_params_Panel)


# ## Simulate paths using parameters given above
# 
# Users are able to select multiple variables of interest to simulate, by holding shift and/or ctrl (or command) pressed and then mouse click variable names. Please always press `Update parameters` first, and then press `Run simulation`.
# 
# The dashed curves indicate deterministic growth path (with both shocks being 0 at every period), and the solid curves indicate stochastic growth path (with randomly generated standard normal shock at every period).

# In[2]:


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


# In[3]:


if habit.value == 1:
    display(simulate_box_external_habit_run)
elif habit.value == 2:
    display(simulate_box_internal_habit_run)


# In[4]:


if habit.value == 1:
    all_var_names = ['V/Y', 'R/Y', 'U/Y', 'C/Y', 'MC/MU', 'I/Y',                            'H/Y', 'K/Y', 'Y_{t+1}/Y_t', 'Z1', 'Z2', 'Z2_lag']
    selected_index = [all_var_names.index(element) for element in simulate_external_habit.value]
elif habit.value == 2:
    all_var_names = ['V/Y', 'R/Y', 'U/Y', 'C/Y', 'MH/MU', 'MC/MU', 'I/Y',                            'H/Y', 'K/Y', 'Y_{t+1}/Y_t', 'Z1', 'Z2', 'Z2_lag']
    selected_index = [all_var_names.index(element) for element in simulate_internal_habit.value]
    
fig, ax = plot_simulation(int(order.value), T, [np.float(r) for r in rhos.value.split(',')],                          γ, χ, α, ϵ, a, g, A_x, B_x, habit.value, KoverY_ss, selected_index)
plt.tight_layout()
plt.show()


# ## Impulse responses with a slider parameter

# In[5]:


display(slider_box_run)


# In[6]:


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

