from Fredkin_Shorten import Krotov_optimizer_Fredkin_Shorten
import numpy as np

np.random.seed(0)

FW = Krotov_optimizer_Fredkin_Shorten(N_Iteration=3,N_T=104)
FW.mute_initial(mode=1)
FW.set_control()

mode = 0

if mode:
    FW.optimize()
    FW.plot_C()
else:
    while True:
        FW.set_control()

        while True:
            m = np.random.randint(4)
            n = np.random.randint(104)
            if FW.mute_memory[m][n] == 2:
                print("m,n=",m,n)
                FW.mute_control(time_index=n,control_index=m)
                break
        
        FW.optimize()

        if FW.JT_final[-1] < 10**(-4):
            FW.mute_memory[m][n] = 0
            #FW.save()
            print("saved!\n")
        else:
            FW.mute_memory[m][n] = 1
            FW.control_switch[m][n] = 1
            print("not save!\n")
        
        # print("mute memory now:",FW.mute_memory)
        # print("\n")
        #FW.save_memory()
