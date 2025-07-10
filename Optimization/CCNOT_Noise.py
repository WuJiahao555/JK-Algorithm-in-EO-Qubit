from CCNOT_Noise_Analysis import CCNOT_Noise
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

run = 0

if run:
    np.random.seed(0)

    N_Scale = 20
    mu_power_min = -6
    mu_power_max = -1
    mu_list = np.zeros(N_Scale+1)

    for i in range(N_Scale+1):
        power = mu_power_min + i*(mu_power_max-mu_power_min)/N_Scale
        mu_list[i] = 10**power

    inf_decomposition = np.zeros(N_Scale+1)
    inf_search = np.zeros(N_Scale+1)

    N_MC = 100

    NA = CCNOT_Noise()
    for i in range(N_Scale+1):
        mu = mu_list[i]
        sigma = 0.1*mu
        dJ_list = np.random.normal(mu,sigma,N_MC)
        for j in range(N_MC):
            NA.dJ = dJ_list[j]
            NA.load_H()
            NA.U_decomposition()
            NA.U_search()
            inf_decomposition[i] = (inf_decomposition[i]*j+NA.inf(mode=0))/(j+1)
            inf_search[i] = (inf_search[i]*j+NA.inf(mode=1))/(j+1)

    datapath = os.path.dirname(__file__) + os.sep + "CCNOT_Noise_MC100.pickle"
    data = {"mu_list":mu_list,"inf_decomposition":inf_decomposition,"inf_search":inf_search}
    with open(datapath,"wb") as f:
        pickle.dump(data,f)
else:
    datapath = os.path.dirname(__file__) + os.sep + "CCNOT_Noise_MC100.pickle"
    with open(datapath,"rb") as f:
        data = pickle.load(f)

    plt.figure(figsize=(8,6))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['text.usetex'] = True

    plt.loglog(data["mu_list"],data["inf_decomposition"],label='infidelity by decomposition',color='blue', linestyle='-', marker='o', markersize=5, linewidth=1,markerfacecolor='none')
    plt.loglog(data["mu_list"],data["inf_search"],label='infidelity by searching',color='red', linestyle='-', marker='>', markersize=5, linewidth=1,markerfacecolor='none')

    plt.xlabel(r'$\delta J/J$', fontsize=16, color='black')
    plt.ylabel(r'$1-F$', fontsize=16, color='black')

    plt.minorticks_on()

    plt.tick_params(axis='both', which='major', labelsize=18, direction='in', length=10)
    plt.tick_params(axis='both', which='minor', direction='in', length=5)

    plt.tick_params(axis='both', which='both', top=True, right=True, labeltop=False, labelright=False)

    plt.gca().xaxis.set_tick_params(pad=2)
    plt.gca().yaxis.set_tick_params(pad=2)

    legend = plt.legend(fontsize=18, loc='upper left', bbox_to_anchor=(0.05, 0.95), frameon=True, fancybox=True, shadow=True,labelspacing=1)

    legend.get_frame().set_edgecolor('#404040')
    legend.get_frame().set_linewidth(1)

    legend._legend_box.align = "center"

    plt.show()
