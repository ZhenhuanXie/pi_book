import numpy as np
np.set_printoptions(precision=6, suppress=True)
import matplotlib.pyplot as plt
from model_specification import eq_cond, ss_func, log_SDF_ex
from expansion import recursive_expansion
from elasticity import price_elasticity
try:
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
    from plotly.offline import init_notebook_mode, iplot
except ImportError:
    print("Installing plotly. This may take a while.")
    from pip._internal import main as pipmain
    pipmain(['install', 'plotly'])
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
    from plotly.offline import init_notebook_mode, iplot
    
def plot_simulation(order, T, rhos, γ, χ, α, ϵ, a, g, A, B, habit_val, KoverY_ss, selected_index):
    colors = ['blue', 'green', 'red', 'gold', 'cyan', 'magenta', 'yellow', 'salmon', 'grey', 'black']
    titles_external_habit = [r'$\log\frac{V_t}{Y_t}$', r'$\log\frac{R_t}{Y_t}$', r'$\log\frac{U_t}{Y_t}$', r'$\log\frac{C_t}{Y_t}$', r'$\log\frac{MC_t}{MU_t}$', r'$\frac{I_t}{Y_{t}}$', r'$\log\frac{H_t}{Y_t}$', r'$\frac{K_t}{Y_t}$', r'$\log\frac{Y_{t+1}}{Y_{t}}$', r'$Z_{1,t}$', r'$Z_{2,t}$', r'$Z_{2,t-1}$']
    titles_internal_habit = [r'$\log\frac{V_t}{Y_t}$', r'$\log\frac{R_t}{Y_t}$', r'$\log\frac{U_t}{Y_t}$', r'$\log\frac{C_t}{Y_t}$', r'$\log\frac{MH_t}{MU_t}$', r'$\log\frac{MC_t}{MU_t}$', r'$\frac{I_t}{Y_{t}}$', r'$\log\frac{H_t}{Y_t}$', r'$\frac{K_t}{Y_t}$',r'$\log\frac{Y_{t+1}}{Y_{t}}$', r'$Z_{1,t}$', r'$Z_{2,t}$', r'$Z_{2,t-1}$']
    
    fig, axs = plt.subplots((len(selected_index)+1)//2,2, squeeze = False, figsize = (10,4*(len(selected_index)+1)//2), dpi = 200)
    
    second_order = False if order == 1 else True
    for i, ρ in enumerate(rhos):
        if habit_val == 1:
            internal = False
            var_shape = (6,6,2)
            args = (γ, ρ, χ, α, ϵ, a, g, A, B, internal, KoverY_ss)
            modelSol = recursive_expansion(eq_cond=eq_cond,
                           ss=ss_func,
                           var_shape=var_shape,
                           γ=args[0],
                           second_order=second_order,
                           args=args)
            titles = titles_external_habit
        elif habit_val  == 2:
            internal = True
            var_shape = (7,6,2)
            args = (γ, ρ, χ, α, ϵ, a, g, A, B, internal, KoverY_ss)
            modelSol = recursive_expansion(eq_cond=eq_cond,
                           ss=ss_func,
                           var_shape=var_shape,
                           γ=args[0],
                           second_order=second_order,
                           args=args)
            titles = titles_internal_habit
        _, _, n_W = modelSol.var_shape
        if i == 0:
            Ws_1 = np.zeros((T,n_W))
            Ws_2 = np.random.multivariate_normal(np.zeros(n_W), np.eye(n_W), size = T)
            
        sim_result_deterministic = modelSol.simulate(Ws_1)
        sim_result_stochastic = modelSol.simulate(Ws_2)
        
        for j, index in enumerate(selected_index):
            axs[j//2][j%2].plot(sim_result_deterministic[:,index], color = colors[i], linestyle = 'dashed')
            axs[j//2][j%2].plot(sim_result_stochastic[:,index], color = colors[i], alpha = 0.6, label = r'$\rho = {:.2f}$'.format(ρ))
            axs[j//2][j%2].set_ylabel(titles[index])
            axs[j//2][j%2].set_xlabel('Quarters')
            axs[j//2][j%2].legend(loc = 'lower right')
        if len(selected_index) % 2 != 0:
            axs[-1][-1].set_axis_off()
    return fig, axs
    


def plot_impulse_pi(rhos, T, order, gamma, slider_varname, habit_val, chi, alpha, epsilon, a, g, A, B,
                 KoverY_ss, shock = 1, title = None, confidence_interval = None):
    
    """
    Given a set of parameters, computes and displays the impulse responses of
    consumption, capital, the consumption-investment ratio, along with the
    shock price elacticities.

    Input
    ==========
    Note that the values of delta, phi, A, and a_k are specified within the code
    and are only used for the empirical_method = 0 or 0.5 specifications (see below).

    rhos:               The set of rho values for which to plot the IRFs.
    gamma:              The risk aversion of the model.
    betaz:              Shock persistence.
    T:                  Number of periods to plot.
    shock:              (1 or 2) Defines which of the two possible shocks to plot.
    empirical method:   Use 0 to use Eberly and Wang parameters and 0.5 for parameters
                        from a low adjustment cost setting. Further cases still under
                        development.
    transform_shocks:   True or False. True to make the rho = 1 response to
                        shock 2 be transitory.
    title:              Title for the image plotted.
    """
    colors = ['blue', 'green', 'red', 'black', 'cyan', 'magenta', 'yellow', 'black']
    mult_fac = len(rhos) // len(colors) + 1
    colors = colors * mult_fac
    
    Cmin = 0
    Cmax = 0
    Ymin = 0
    Ymax = 0
    CmYmin = 0
    CmYmax = 0
    Kmin = 0
    Kmax = 0
    Imin = 0
    Imax = 0
    ϵ_p_c_min = 0
    ϵ_p_c_max = 0

    fig = make_subplots(3, 2, print_grid = False, specs=[[{}, {}], [{}, {}], [{}, {}]])
    
    # Update slider information
    if slider_varname == 'γ':
        slider_vars = gamma
    elif slider_varname == 'χ':
        slider_vars = chi
    elif slider_varname == 'α':
        slider_vars = alpha
    elif slider_varname == 'ϵ':
        slider_vars = epsilon

        
    solved_models = []
    
    for i, r in enumerate(rhos):
        for j, var in enumerate(slider_vars):
            # Update variables
            if slider_varname == 'γ':
                gamma = var
            elif slider_varname == 'χ':
                chi = var
            elif slider_varname == 'α':
                alpha = var
            elif slider_varname == 'ϵ':
                epsilon = var

            # Specify models with/without habit
            if habit_val == 2:
                internal = True
                var_shape = (7,6,2)
                c_loc = 3
                i_loc = 6
                k_loc = 8
                gy_loc = 9
                args = (gamma, r, chi, alpha, epsilon, a, g, A, B, internal, KoverY_ss)

            elif habit_val == 1:
                internal = False
                var_shape = (6,6,2)
                c_loc = 3
                i_loc = 5
                k_loc = 7
                gy_loc = 8
                args = (gamma, r, chi, alpha, epsilon, a, g, A, B, internal, KoverY_ss)
                
            second_order = False if order == 1 else True
                
            modelSol = recursive_expansion(eq_cond=eq_cond,
                                   ss=ss_func,
                                   var_shape=var_shape,
                                   γ=args[0],
                                   second_order=second_order,
                                   args=args)
            n_Y, n_Z, n_W = modelSol.var_shape
                
            states, controls = modelSol.IRF(T, shock - 1)
                
            CmY_IRF = controls[:,c_loc] * 100
            Y_IRF = np.cumsum(states[:,gy_loc-n_Y]) * 100
            C_IRF = CmY_IRF + Y_IRF
            K_IRF = states[:, k_loc-n_Y] * 100
            I_IRF = controls[:, i_loc] * 100
                
            Z2_tp1 = modelSol.Z2_tp1 if second_order else None
            Y_growth = modelSol.X_tp1.split()[gy_loc]
            X_growth = modelSol.X_tp1 - modelSol.X_t
            X_growth_list = X_growth.split()
            CmY_growth = X_growth_list[c_loc]
            C_growth = CmY_growth + Y_growth

            log_SDF = modelSol.approximate_fun(log_SDF_ex, args) + modelSol.log_M
            
            ϵ_p_c = price_elasticity(C_growth, log_SDF, modelSol.Z1_tp1, Z2_tp1, T, shock-1, 0.5).flatten()
                
            if confidence_interval is not None and order == 2:
                ϵ_p_c_lower = price_elasticity(C_growth, log_SDF, modelSol.Z1_tp1, Z2_tp1, T, shock-1, 0.5-confidence_interval/2).flatten()
                ϵ_p_c_upper = price_elasticity(C_growth, log_SDF, modelSol.Z1_tp1, Z2_tp1, T, shock-1, 0.5+confidence_interval/2).flatten()
                
            solved_models.append(modelSol)

            Cmin = min(Cmin, np.min(C_IRF) * 1.2)
            Cmax = max(Cmax, np.max(C_IRF) * 1.2)
            Ymin = min(Ymin, np.min(Y_IRF) * 1.2)
            Ymax = max(Ymax, np.max(Y_IRF) * 1.2)
            CmYmin = min(CmYmin, np.min(CmY_IRF) * 1.2)
            CmYmax = max(CmYmax, np.max(CmY_IRF) * 1.2)
            Kmin = min(Kmin, np.min(K_IRF) * 1.2)
            Kmax = max(Kmax, np.max(K_IRF) * 1.2)
            Imin = min(Imin, np.min(I_IRF) * 1.2)
            Imax = max(Imax, np.max(I_IRF) * 1.2)
                
            if confidence_interval is None or order == 1:
                ϵ_p_c_min = min(ϵ_p_c_min, np.min(ϵ_p_c) * 1.2)
                ϵ_p_c_max = max(ϵ_p_c_max, np.max(ϵ_p_c) * 1.2)

            else:
                ϵ_p_c_min = min(ϵ_p_c_min, np.min(ϵ_p_c_lower) * 1.2)
                ϵ_p_c_max = max(ϵ_p_c_max, np.max(ϵ_p_c_upper) * 1.2)
                
                
            fig.add_scatter(y = C_IRF, row = 1, col = 1, visible = j == 0,
                            name = 'rho = {}'.format(r), line = dict(color = (colors[i])))
            fig.add_scatter(y = K_IRF, row = 1, col = 2, visible = j == 0,
                            name = 'rho = {}'.format(r), line = dict(color = (colors[i])))
            fig.add_scatter(y = CmY_IRF, row = 2, col = 1, visible = j == 0,
                            name = 'rho = {}'.format(r), line = dict(color = (colors[i])))
            fig.add_scatter(y = I_IRF, row = 2, col = 2, visible = j == 0,
                            name = 'rho = {}'.format(r), line = dict(color = (colors[i])))
            fig.add_scatter(y = Y_IRF, row = 3, col = 1, visible = j == 0,
                            name = 'rho = {}'.format(r), line = dict(color = (colors[i])))
            fig.add_scatter(y = ϵ_p_c, row = 3, col = 2, visible = j == 0,
                            name = 'rho = {}'.format(r), line = dict(color = (colors[i])))
                            
            if confidence_interval is not None and order == 2:

                fig.add_scatter(y = ϵ_p_c_lower, row = 3, col = 2, visible = j == 0,
                            name = 'rho = {}'.format(r), line = dict(color = (colors[i])))
                fig.add_scatter(y = ϵ_p_c_upper, row = 3, col = 2, visible = j == 0,
                            name = 'rho = {}'.format(r), line = dict(color = (colors[i])),
                               fill = 'tonexty')


    steps = []
    for i in range(len(slider_vars)):
        step = dict(
            method = 'restyle',
            args = ['visible', ['legendonly'] * len(fig.data)],
            label = slider_varname + ' = '+'{}'.format(round(slider_vars[i], 2))
        )
        if confidence_interval is None or order == 1:
            for j in range(6):
                for k in range(len(rhos)):
                    step['args'][1][i * 6 + j + k * len(slider_vars) * 6] = True
        else:
            for j in range(8):
                for k in range(len(rhos)):
                    step['args'][1][i * 8 + j + k * len(slider_vars) * 8] = True
        steps.append(step)


    sliders = [dict(
        steps = steps
    )]

    fig.layout.sliders = sliders
    fig['layout'].update(height=800, width=1000,
                     title=title.format(shock), showlegend = False)

    fig['layout']['xaxis1'].update(range = [0, T])
    fig['layout']['xaxis2'].update(range = [0, T])
    fig['layout']['xaxis3'].update(range = [0, T])
    fig['layout']['xaxis4'].update(range = [0, T])
    fig['layout']['xaxis5'].update(range = [0, T])
    fig['layout']['xaxis6'].update(range = [0, T])

    fig['layout']['yaxis1'].update(title=r'$\text{IRF: }\log C$', range = [Cmin, Cmax])
    fig['layout']['yaxis2'].update(title=r'$\text{IRF: }\frac{K}{Y}$', range=[Kmin, Kmax])
    fig['layout']['yaxis3'].update(title=r'$\text{IRF: }\frac{C}{Y}$', range = [CmYmin, CmYmax])#showgrid=False)
    fig['layout']['yaxis4'].update(title=r'$\text{IRF: }\frac{I}{Y}$', range = [Imin, Imax])
    fig['layout']['yaxis5'].update(title=r'$\text{IRF: }\log Y$', range = [Ymin, Ymax])
    fig['layout']['yaxis6'].update(title=r'$\text{Price Elasticity: }\log C$', range = [ϵ_p_c_min, ϵ_p_c_max])
    
    return fig, solved_models
