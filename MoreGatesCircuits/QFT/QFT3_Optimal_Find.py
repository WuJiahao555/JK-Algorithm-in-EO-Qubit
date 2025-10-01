from QFT3_Optimizer import Krotov_optimizer_QFT3
import os
import scipy.io

FW = Krotov_optimizer_QFT3(N_Iteration=1000,N_T=80)

FW.set_control(mode=1)

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
