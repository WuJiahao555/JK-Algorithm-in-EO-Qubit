from CCNOT_Shorten import Krotov_optimizer_CCNOT_Shorten
import numpy as np
import os
import scipy.io

np.random.seed(0)
N_T = 55

FW = Krotov_optimizer_CCNOT_Shorten(N_Iteration=1)
FW.mute_initial(mode=1)
FW.set_control()

continue_optimize = 1

if continue_optimize:
    FW.optimize()
    FW.save()

FW.plot_C()
U = FW.test()

save_mat = 1

if save_mat:
    datapath = os.path.dirname(__file__) + os.sep + "U_55.mat"
    scipy.io.savemat(datapath,{"U_55":U})
