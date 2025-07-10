from CCNOT_Optimizer import Krotov_optimizer_CCNOT
import os
import scipy.io

FW = Krotov_optimizer_CCNOT(N_Iteration=1,N_T=55)
FW.set_control(mode=1)

continue_optimize = 1

if continue_optimize:
    FW.optimize()
    #FW.save()

FW.plot_C()
U = FW.test()

save_mat = 0

if save_mat:
    datapath = os.path.dirname(__file__) + os.sep + "U_55.mat"
    scipy.io.savemat(datapath,{"U_55":U})

print(U[0][0])
print(U[1][1])
print(U[2][2])
print(U[3][3])
print(U[4][4])
print(U[5][5])
print(U[6][7])
print(U[7][6])
print("\n")
print(U[0+10][0+10])
print(U[1+10][1+10])
print(U[2+10][2+10])
print(U[3+10][3+10])
print(U[4+10][4+10])
print(U[5+10][5+10])
print(U[6+10][7+10])
print(U[7+10][6+10])
print("\n")
print(U[0+42][0+42])
print(U[1+42][1+42])
print(U[2+42][2+42])
print(U[3+42][3+42])
print(U[4+42][4+42])
print(U[5+42][5+42])
print(U[6+42][7+42])
print(U[7+42][6+42])
