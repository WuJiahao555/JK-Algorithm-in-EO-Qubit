from QFT3_Optimizer import Krotov_optimizer_QFT3
import os
import scipy.io

FW = Krotov_optimizer_QFT3(N_Iteration=1000,N_T=80)

#seed 1360, 4500 already: 100-90% 250-95% 500-98% 1000-99% 4500-99.7%
#FW.J_78[10] = 0 restart: 300 return-99.64%-0.00365

#seed 1626, 6000 already: 100-0.209, 250-0.0263, 500-0.0138, 1000-0.00741, 1500-0.003325, 2000-0.001617,
#2500--0.0007-5052, 3000--0.0003-2574, 4000--0.0000-7897, 5000--0.0000-3060, 6000--0.0000-1415, 10000--0.0000-0324

FW.set_control(mode=1,seed=1626)

continue_optimize = 1
if continue_optimize:
    FW.optimize()
    FW.save()

FW.plot_C()

U = FW.test()

save_mat = 0
if save_mat:
    datapath = os.path.dirname(__file__) + os.sep + "U_55.mat"
    scipy.io.savemat(datapath,{"U_55":U})

print(U[0][0])
print(U[1][0])
print(U[2][0])
print(U[3][0])
print(U[4][0])
print(U[5][0])
print(U[6][0])
print(U[7][0])
