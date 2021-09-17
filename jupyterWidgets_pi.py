#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains the code for the Jupyter widgets. It is not required
for the model framework. The widgets are purely for decorative purposes.
"""

#######################################################
#                    Dependencies                     #
#######################################################

from ipywidgets import widgets, Layout, Button, HBox, VBox, interactive
from IPython.core.display import display
from IPython.display import clear_output, Markdown, Latex
from IPython.display import Javascript
import numpy as np
try:
    import plotly.graph_objs as go
    from plotly.tools import make_subplots
    from plotly.offline import init_notebook_mode, iplot
except ImportError:
    print("Installing plotly. This may take a while.")
    from pip._internal import main as pipmain
    pipmain(['install', 'plotly'])
    import plotly.graph_objs as go
    from plotly.tools import make_subplots
    from plotly.offline import init_notebook_mode, iplot

# Define global parameters for parameter checks
params_pass = False
model_solved = False

#######################################################
#          Jupyter widgets for user inputs            #
#######################################################

## This section creates the widgets that will be diplayed and used by the user
## to input parameter values.

style_mini = {'description_width': '5px'}
style_short = {'description_width': '100px'}
style_med = {'description_width': '180px'}
style_long = {'description_width': '200px'}

layout_mini =Layout(width='18.75%')
layout_50 =Layout(width='50%')
layout_med =Layout(width='70%')

widget_layout = Layout(width = '100%')

rhos = widgets.Text( ## death rate
    value="0.667, 1.0000001, 1.5",
    disabled=False,
    description = 'Inverse IES ρ',
    style = {'description_width': '180px'},
    layout = Layout(width='70%')
)

ta = widgets.BoundedFloatText( ## risk free rate
    value=0.00663,
    min = 0.001,
    max = 0.010,
    step=0.00001,
    disabled=False,
    description = 'a',
    style = style_med,
    layout = layout_med
)

tA_1 = widgets.BoundedFloatText( ## 
    value=0.704,
    min = 0.,
    max = 1.,
    step=0.001,
    disabled=False,
    description = 'A_1',
    style = style_med,
    layout = layout_med
)

tA_21 = widgets.BoundedFloatText( ## 
    value=1,
    min = 0.5,
    max = 1.5,
    step=0.001,
    disabled=False,
    description = 'A_21',
    style = style_med,
    layout = layout_med
)

tA_22 = widgets.BoundedFloatText( ## 
    value=-0.154,
    min = -1.,
    max = 0.,
    step=0.001,
    disabled=False,
    description = 'A_22',
    style = style_med,
    layout = layout_med
)

tB_11 = widgets.BoundedFloatText( ## 
    value=0.144,
    min = 0.1,
    max = 2.0,
    step=0.001,
    disabled=False,
    description = 'B_11',
    style = style_med,
    layout = layout_med
)

tB_22 = widgets.BoundedFloatText( ## 
    value=0.206,
    min = 0.1,
    max = 2.0,
    step=0.001,
    disabled=False,
    description = 'B_22',
    style = style_med,
    layout = layout_med
)


gy = widgets.BoundedFloatText( ## income growth rate
    value=0.00373,
    min = 0.001,
    max = 0.010,
    step=0.00001,
    disabled=False,
    description = 'g',
    style = style_med,
    layout = layout_med
)


shock = widgets.Dropdown(
    options = {'1', '2'},
    value = '1',
    description='Shock index:',
    disabled=False,
    style = style_med,
    layout = layout_med
)

order = widgets.Dropdown(
    options = {1, 2},
    value = 1,
    description='Solution order:',
    disabled=False,
    style = style_med,
    layout = layout_med
)

K_Y_ss = widgets.BoundedFloatText(
    value=0.,
    min = -200.,
    max = 200.,
    step=0.01,
    disabled=False,
    description = r'$\frac{K}{Y}$ steady state:',
    style = style_med,
    layout = layout_med
)

slider_var = widgets.Dropdown(
    options = {'γ','α','χ','ϵ'},
    value = 'γ',
    description='Parameter to slide over:',
    disabled=False,
    style = style_med,
    layout = layout_med
)

slider_min= widgets.BoundedFloatText(
    value=2.,
    disabled=False,
    description = 'Min',
    style=style_med,
    layout = layout_med
)

slider_max= widgets.BoundedFloatText(
    value=10.,
    disabled=False,
    description = 'Max',
    style=style_med,
    layout = layout_med
)

slider_step= widgets.BoundedFloatText(
    value=1.,
    disabled=False,
    description = 'Step',
    style=style_med,
    layout = layout_med
)



habit = widgets.Dropdown(
    options = [('Externality ignored',1), ('Habit internalized',2), ],
    value = 2,
    description='Habit:',
    disabled=False,
    style = style_med,
    layout = layout_med
)

gamma = widgets.BoundedFloatText(
    value=10.,
    min = 1.,
    max = 20,
    step=0.01,
    disabled=False,
    description = 'γ',
    style=style_med,
    layout = layout_med
)

chi = widgets.BoundedFloatText(
    value= 0.9,
    step= 0.05,
    min = 0,
    max = .9999,
    description=r'Habit Depreciation (χ)',
    disabled=False,
    style = style_med,
    layout = layout_med
)

alpha = widgets.BoundedFloatText(
    value= 0.9,
    step= 0.0001,
    min = 0,
    max = .9999,
    description=r'α',
    disabled=False,
    style = style_med,
    layout = layout_med
)

epsilon = widgets.BoundedFloatText(
    value= 10,
    step= 0.05,
    min = 0,
    max = 1000,
    description=r'ϵ',
    disabled=False,
    style = style_med,
    layout = layout_med
)

def displayHabit(habit):
    ## This function displays the box to input households productivity
    ## if hosueholds are allowed to hold capital.
    #if habit == 1:
    #    chi.layout.display = 'none'
    #    alpha.layout.display = 'none'
    #    epsilon.layout.display = 'none'
    #    chi.value = 0.9
    #    alpha.value = 0.9
    #    epsilon.value = 10
    #else:
    chi.layout.display = None
    alpha.layout.display = None
    epsilon.layout.display = None
    chi.value = 0.9
    alpha.value = 0.9
    epsilon.value = 10
    display(chi)
    display(alpha)
    display(epsilon)

habitOut = widgets.interactive_output(displayHabit, {'habit': habit})

timeHorizon = widgets.BoundedIntText(
    value=100,
    min = 10,
    max = 2000,
    step=10,
    disabled=False,
    description = 'Time Horizon (quarters)',
    style = style_med,
    layout = layout_med
)

plotName = widgets.Text(
    value='Stochastic Growth',
    placeholder='Stochastic Growth',
    description='Plot Title',
    disabled=False,
    style = style_med,
    layout = layout_med
)

conf_int = widgets.BoundedFloatText(
    value= .9,
    step= 0.0001,
    min = 0,
    max = .9999,
    description='Risk Price Confidence Interval',
    disabled=False,
    style = style_med,
    layout = layout_med
)

overwrite = widgets.Dropdown(
    options = {'Yes', 'No'},
    value = 'Yes',
    description='Overwrite if folder exists:',
    disabled=False,
    style = style_med,
    layout = layout_med
)

checkParams = widgets.Button(
    description='Update parameters',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
)

runSim = widgets.Button(
    description='Run simulation',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
)

checkParams2 = widgets.Button(
    description='Update parameters',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
)

runSlider = widgets.Button(
    description='Run models',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
)

#showSS = widgets.Button(
#    description='Show steady states',
#    disabled=False,
#    button_style='', # 'success', 'info', 'warning', 'danger' or ''
#)
#
#displayPlotPanel = widgets.Button(
#    description='Show panel chart',
#    disabled=False,
#    button_style='', # 'success', 'info', 'warning', 'danger' or ''
#)

#runRFR = widgets.Button(
#    description='Simulate states',
#    disabled=False,
#    button_style='', # 'success', 'info', 'warning', 'danger' or ''
#)
#
#displayRFRMoments = widgets.Button(
#    description='Show table',
#    disabled=False,
#    button_style='', # 'success', 'info', 'warning', 'danger' or ''
#)

#displayRFRPanel = widgets.Button(
#    description='Show panel chart',
#    disabled=False,
#    button_style='', # 'success', 'info', 'warning', 'danger' or ''
#)

box_layout       = Layout(width='100%', flex_flow = 'row')#, justify_content='space-between')
box_layout_wide  = Layout(width='100%', justify_content='space-between')
box_layout_small = Layout(width='10%')

Growth_box = VBox([widgets.Label(value="Growth rates"), ta, gy], layout = Layout(width='90%'))
Preferences_box = VBox([widgets.Label(value="Preference parameters"), rhos], layout = Layout(width='90%'))
A_matrix_box = VBox([widgets.Label(value="A Matrix"), tA_1, tA_21, tA_22], layout = Layout(width='90%'))
B_matrix_box = VBox([widgets.Label(value="B Matrix"), tB_11, tB_22], layout = Layout(width='90%'))

habit_box = VBox([widgets.Label(value="Habit"), habit, habitOut], layout = Layout(width='90%'))
order_box = VBox([widgets.Label(value="Solution details"), order, K_Y_ss], layout = Layout(width='90%'))
gamma_box = VBox([widgets.Label(value="Risk sensitivity"), gamma], layout = Layout(width='90%'))
slider_box = VBox([widgets.Label(value="Slider setting"), slider_var,slider_min,slider_max,slider_step], layout = Layout(width='90%'))

Selector_box = VBox([widgets.Label(value="Graph parameters"), shock, plotName, timeHorizon, conf_int], layout = Layout(width='90%'))


sim_var_names_external_habit = ['V/Y', 'R/Y', 'U/Y', 'C/Y', 'MC/MU', 'I/Y', \
                           'H/Y', 'K/Y', 'Y_{t+1}/Y_t', 'Z1', 'Z2', 'Z2_lag']
sim_var_names_internal_habit = ['V/Y', 'R/Y', 'U/Y', 'C/Y', 'MH/MU', 'MC/MU', 'I/Y', \
                           'H/Y', 'K/Y', 'Y_{t+1}/Y_t', 'Z1', 'Z2', 'Z2_lag']
                           
simulate_external_habit = widgets.SelectMultiple(options = sim_var_names_external_habit,
    value = ['C/Y'],
    rows = len(sim_var_names_external_habit),
    disabled = False
)
simulate_box_external_habit = VBox([widgets.Label(value="Select variables to simulate:"),simulate_external_habit], layout = Layout(width='100%'))

simulate_internal_habit = widgets.SelectMultiple(options = sim_var_names_internal_habit,
    value = ['C/Y'],
    rows = len(sim_var_names_internal_habit),
    disabled = False
)
simulate_box_internal_habit = VBox([widgets.Label(value="Select variables to simulate:"),simulate_internal_habit], layout = Layout(width='100%'))


habit_box_layout = Layout(width='56%', flex_flow = 'row')
line1      = HBox([Selector_box, order_box], layout = box_layout)
line2      = HBox([Growth_box, Preferences_box], layout = box_layout)
line3      = HBox([A_matrix_box, B_matrix_box], layout = box_layout)
line4      = HBox([habit_box, gamma_box], layout = box_layout)
#line5      = HBox([slider_box ], layout = box_layout)
fixed_params_Panel = VBox([line1, line2, line3, line4])
run_box_sim = VBox([widgets.Label(value="Run simulation"), checkParams, runSim], layout = Layout(width='100%'))
run_box_slider = VBox([widgets.Label(value="Run multiple models"), checkParams2, runSlider], layout = Layout(width='100%'))
#run_box_rfr = VBox([widgets.Label(value="Compute interest rate"), runRFR, displayRFRMoments])

simulate_box_external_habit_run = HBox([simulate_box_external_habit, run_box_sim], layout = Layout(width='100%'))
simulate_box_internal_habit_run = HBox([simulate_box_internal_habit, run_box_sim], layout = Layout(width='100%'))

slider_box_run = HBox([slider_box, run_box_slider], layout = Layout(width='100%'))

#######################################################
#                      Functions                      #
#######################################################

def checkParamsFn(b):
    ## This is the function triggered by the updateParams button. It will
    ## check dictionary params to ensure that adjustment costs are well-specified.
    clear_output() ## clear the output of the existing print-out
    display(Javascript("Jupyter.notebook.execute_cells([3])"))
    if habit.value == 1:
        display(simulate_box_external_habit_run) ## after clearing output, re-display buttons
    elif habit.value == 2:
        display(simulate_box_internal_habit_run) ## after clearing output, re-display buttons
    global params_pass
    global model_solved
    model_solved = False
    rho_vals = np.array([np.float(r) for r in rhos.value.split(',')])
    if gamma.value < 1:
        params_pass = False
        print("Gamma should be greater than {}.".format(1))
    elif chi.value <0 or chi.value > 1:
        params_pass = False
        print("Chi should be between 0 and 1.")
    elif alpha.value <=0 or alpha.value > 1:
        params_pass = False
        print("Alpha should be between 0 and 1.")
#    elif epsilon.value <=1:
#        params_pass = False
#        print("Epsilon should be greater than 1.")
    elif 1 in rho_vals:
        params_pass = False
        print("Rho should be different from 1.")
    elif max(rho_vals) >= 1.8:
        params_pass = False
        print("Rho should be smaller than 1.8.")
    else:
        params_pass = True
        print("Parameter check passed.")
        
def checkParams2Fn(b):
    ## This is the function triggered by the updateParams button. It will
    ## check dictionary params to ensure that adjustment costs are well-specified.
    clear_output() ## clear the output of the existing print-out
    display(Javascript("Jupyter.notebook.execute_cells([3])"))
    display(slider_box_run)
    global params_pass
    global model_solved
    model_solved = False
    rho_vals = np.array([np.float(r) for r in rhos.value.split(',')])
    if gamma.value < 1:
        params_pass = False
        print("Gamma should be greater than {}.".format(1))
    elif chi.value <0 or chi.value > 1:
        params_pass = False
        print("Chi should be between 0 and 1.")
    elif alpha.value <=0 or alpha.value > 1:
        params_pass = False
        print("Alpha should be between 0 and 1.")
#    elif epsilon.value <=1:
#        params_pass = False
#        print("Epsilon should be greater than 1.")
    elif 1 in rho_vals:
        params_pass = False
        print("Rho should be different from 1.")
    elif max(rho_vals) >= 1.8:
        params_pass = False
        print("Rho should be smaller than 1.8.")
    else:
        params_pass = True
        print("Parameter check passed.")

def runSimFn(b):
    ## This is the function triggered by the runSim button.
    global model_solved
    if params_pass:
        print("Running simulation...")
        display(Javascript("Jupyter.notebook.execute_cells([5])"))
        model_solved = True
    else:
        print("You must update the parameters first.")

def runSliderFn(b):
    ## This is the function triggered by the runModel button.
    global model_solved
    if params_pass:
        print("Running models...")
        display(Javascript("Jupyter.notebook.execute_cells([8])"))
        model_solved = True
    else:
        print("You must update the parameters first.")

#def showSSFn(b):
#    if model_solved:
#        print("Showing steady state values.")
#        display(Javascript("Jupyter.notebook.execute_cells([17])"))
#    else:
#        print("You must run the model first.")

#def displayPlotPanelFn(b):
#    if model_solved:
#        print("Showing plots.")
#        display(Javascript("Jupyter.notebook.execute_cells([18])"))
#    else:
#        print("You must run the model first.")
#     
#def runRFRFn(b):
#    clear_output() ## clear the output of the existing print-out
#    display(run_box_rfr) ## after clearing output, re-display buttons
#    if model_solved:
#        print("Calculating values.")
#        display(Javascript("Jupyter.notebook.execute_cells([21, 22])"))
#    else:
#        print("You must run the model first.")
#
#def displayRFRMomentsFn(b):
#    print("Showing moment table.")
#    display(Javascript("Jupyter.notebook.execute_cells([23,24])"))
#
#def displayRFRPanelFn(b):
#    print("Showing plots.")
#    display(Javascript("Jupyter.notebook.execute_cells([25])"))

#######################################################
#                 Configure buttons                   #
#######################################################

selectedMoments = []

checkParams.on_click(checkParamsFn)
checkParams2.on_click(checkParams2Fn)
runSim.on_click(runSimFn)
runSlider.on_click(runSliderFn)
#showSS.on_click(showSSFn)
#displayPlotPanel.on_click(displayPlotPanelFn)
#runRFR.on_click(runRFRFn)
#displayRFRMoments.on_click(displayRFRMomentsFn)
#displayRFRPanel.on_click(displayRFRPanelFn)
