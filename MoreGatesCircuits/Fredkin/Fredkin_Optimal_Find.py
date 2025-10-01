from Fredkin_Optimizer import Krotov_optimizer_Fredkin
import os
import scipy.io

FW = Krotov_optimizer_Fredkin(N_Iteration=1,N_T=104)
FW.set_control(mode=1,seed=0)

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
print(U[2][3])
print(U[3][2])
print(U[4][4])
print(U[5][5])
print(U[6][7])
print(U[7][6])
print("\n")

# print(U[0][0])
# print(U[1][1])
# print(U[2][2])
# print(U[3][3])
# print(U[4][4])
# print(U[5][6])
# print(U[6][5])
# print(U[7][7])
# print("\n")
