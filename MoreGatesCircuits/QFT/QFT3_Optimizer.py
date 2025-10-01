import os
import scipy.io
import pickle
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

class Krotov_optimizer_QFT3():
    def __init__(self,N_Iteration=100,N_T=55):
        self.dirpath = os.path.dirname(__file__)
        self.datapath = self.dirpath + os.sep + "data_QFT3_" + str(N_T) + ".pickle"

        self.N_k = 24

        self.initial_state = np.empty(self.N_k,dtype=object)
        for k in range(self.N_k):
            state = np.zeros((90,1))
            if k < 8:
                state[k][0] = 1
            elif k < 16:
                state[k+2][0] = 1
            else:
                state[k+26][0] = 1
            self.initial_state[k] = qt.Qobj(state)
        
        self.target_state = np.empty(self.N_k,dtype=object)
        omega = np.exp(1j*np.pi/4)
        for k in range(self.N_k):
            state = np.zeros(90,dtype=np.complex128) # row vector first

            # k/8=m...n; m=0,1,2; n=0,1,...,7; m=block, n=column
            k_block = k // 8
            k_column = k % 8

            omega = np.exp(1j*np.pi/4)
            multiple = omega**k_column
            local_state = np.array([multiple**i for i in range(8)])

            if k_block == 0:
                state[0:8] = local_state
            elif k_block == 1:
                state[10:18] = local_state
            else:
                state[42:50] = local_state
            
            state = state/np.sqrt(8)
            state = state.reshape(state.shape[0],1)
            self.target_state[k] = qt.Qobj(state)
        
        H_path = self.dirpath + os.sep + "H_3_qubit.mat"
        H = scipy.io.loadmat(H_path)

        self.H_12 = qt.Qobj(H["H_12"])
        self.H_23 = qt.Qobj(H["H_23"])
        self.H_34 = qt.Qobj(H["H_34"])
        self.H_45 = qt.Qobj(H["H_45"])
        self.H_56 = qt.Qobj(H["H_56"])
        self.H_67 = qt.Qobj(H["H_67"])
        self.H_78 = qt.Qobj(H["H_78"])
        self.H_89 = qt.Qobj(H["H_89"])

        self.dt = 1
        self.J0 = np.pi/self.dt
        self.N_T = N_T
        self.T = self.N_T*self.dt

        self.tlist = np.zeros(self.N_T)
        for i in range(self.N_T):
            self.tlist[i] = (i+0.5)*self.dt
        
        self.J_12 = np.zeros(self.N_T)
        self.J_23 = np.zeros(self.N_T)
        self.J_34 = np.zeros(self.N_T)
        self.J_45 = np.zeros(self.N_T)
        self.J_56 = np.zeros(self.N_T)
        self.J_67 = np.zeros(self.N_T)
        self.J_78 = np.zeros(self.N_T)
        self.J_89 = np.zeros(self.N_T)

        self.data = {"J_12":self.J_12,"J_23":self.J_23,"J_34":self.J_34,"J_45":self.J_45,"J_56":self.J_56,"J_67":self.J_67,"J_78":self.J_78,"J_89":self.J_89}

        step = 1.9
        self.S_12 = np.ones(self.N_T)
        self.step_12 = step
        self.S_23 = np.ones(self.N_T)
        self.step_23 = step
        self.S_34 = np.ones(self.N_T)
        self.step_34 = step
        self.S_45 = np.ones(self.N_T)
        self.step_45 = step
        self.S_56 = np.ones(self.N_T)
        self.step_56 = step
        self.S_67 = np.ones(self.N_T)
        self.step_67 = step
        self.S_78 = np.ones(self.N_T)
        self.step_78 = step
        self.S_89 = np.ones(self.N_T)
        self.step_89 = step

        self.N_Iteration = N_Iteration
        self.JT_final = np.zeros(self.N_Iteration)
    
    def clear_control(self):
        self.J_12 = np.zeros(self.N_T)
        self.J_23 = np.zeros(self.N_T)
        self.J_34 = np.zeros(self.N_T)
        self.J_45 = np.zeros(self.N_T)
        self.J_56 = np.zeros(self.N_T)
        self.J_67 = np.zeros(self.N_T)
        self.J_78 = np.zeros(self.N_T)
        self.J_89 = np.zeros(self.N_T)

    def set_control(self,mode=0,seed=0):
        if mode == 0:
            np.random.seed(seed)
            for n in range(self.N_T):
                if n % 2 == 0:
                    self.J_12[n] = np.random.rand()-0.5
                    self.J_23[n] = 0
                    self.J_34[n] = np.random.rand()-0.5
                    self.J_45[n] = 0
                    self.J_56[n] = np.random.rand()-0.5
                    self.J_67[n] = 0
                    self.J_78[n] = np.random.rand()-0.5
                    self.J_89[n] = 0
                else:
                    self.J_12[n] = 0
                    self.J_23[n] = np.random.rand()-0.5
                    self.J_34[n] = 0
                    self.J_45[n] = np.random.rand()-0.5
                    self.J_56[n] = 0
                    self.J_67[n] = np.random.rand()-0.5
                    self.J_78[n] = 0
                    self.J_89[n] = np.random.rand()-0.5
        else:
            with open(self.datapath,"rb") as f:
                self.data = pickle.load(f)
            self.J_12 = self.data["J_12"]
            self.J_23 = self.data["J_23"]
            self.J_34 = self.data["J_34"]
            self.J_45 = self.data["J_45"]
            self.J_56 = self.data["J_56"]
            self.J_67 = self.data["J_67"]
            self.J_78 = self.data["J_78"]
            self.J_89 = self.data["J_89"]

    def _Hamiltonian(self,J_12=0,J_23=0,J_34=0,J_45=0,J_56=0,J_67=0,J_78=0,J_89=0):
        return self.J0*(J_12*self.H_12+J_23*self.H_23+J_34*self.H_34+J_45*self.H_45+J_56*self.H_56+J_67*self.H_67+J_78*self.H_78+J_89*self.H_89)
    
    def _sum_bpf(self,backward,partial,forward):
        s = 0
        for k in range(self.N_k):
            s += np.imag(((backward[k].conj().trans())*(partial)*forward[k]))
        return s

    def optimize(self):
        forward_propagator = np.empty((self.N_k,self.N_T+1),dtype=object)
        for k in range(self.N_k):
            forward = self.initial_state[k]
            forward_propagator[k][0] = forward
            for n in range(self.N_T):
                Hr = self._Hamiltonian(J_12=self.J_12[n],J_23=self.J_23[n],J_34=self.J_34[n],J_45=self.J_45[n],J_56=self.J_56[n],J_67=self.J_67[n],J_78=self.J_78[n],J_89=self.J_89[n])
                Ur = (-1j*Hr*self.dt).expm()
                forward = Ur*forward
                forward_propagator[k][n+1] = forward

        for i in range(self.N_Iteration):
            backward_propagator = np.empty((self.N_k,self.N_T+1),dtype=object)
            for k in range(self.N_k):
                backward = self.target_state[k]/(2*self.N_k)
                backward_propagator[k][self.N_T] = backward
                for n in range(self.N_T,0,-1):
                    Hl = self._Hamiltonian(J_12=self.J_12[n-1],J_23=self.J_23[n-1],J_34=self.J_34[n-1],J_45=self.J_45[n-1],J_56=self.J_56[n-1],J_67=self.J_67[n-1],J_78=self.J_78[n-1],J_89=self.J_89[n-1])
                    Ul = (1j*Hl*self.dt).expm()
                    backward = Ul*backward
                    backward_propagator[k][n-1] = backward
            
            forward = np.empty(self.N_k,dtype=object)
            backward = np.empty(self.N_k,dtype=object)

            forward_propagator_new = np.empty((self.N_k,self.N_T+1),dtype=object)
            for k in range(self.N_k):
                forward[k] = self.initial_state[k]
                forward_propagator_new[k][0] = forward[k]
            
            for n in range(self.N_T):
                for k in range(self.N_k):
                    backward[k] = backward_propagator[k][n]
                
                if n % 2 == 0:
                    self.J_12[n] += (self.S_12[n]/self.step_12)*self._sum_bpf(backward,self.J0*self.H_12,forward)
                    self.J_34[n] += (self.S_34[n]/self.step_34)*self._sum_bpf(backward,self.J0*self.H_34,forward)
                    self.J_56[n] += (self.S_56[n]/self.step_56)*self._sum_bpf(backward,self.J0*self.H_56,forward)
                    self.J_78[n] += (self.S_78[n]/self.step_78)*self._sum_bpf(backward,self.J0*self.H_78,forward)
                else:
                    self.J_23[n] += (self.S_23[n]/self.step_23)*self._sum_bpf(backward,self.J0*self.H_23,forward)
                    self.J_45[n] += (self.S_45[n]/self.step_45)*self._sum_bpf(backward,self.J0*self.H_45,forward)
                    self.J_67[n] += (self.S_67[n]/self.step_67)*self._sum_bpf(backward,self.J0*self.H_67,forward)
                    self.J_89[n] += (self.S_89[n]/self.step_89)*self._sum_bpf(backward,self.J0*self.H_89,forward)
                
                Hr = self._Hamiltonian(J_12=self.J_12[n],J_23=self.J_23[n],J_34=self.J_34[n],J_45=self.J_45[n],J_56=self.J_56[n],J_67=self.J_67[n],J_78=self.J_78[n],J_89=self.J_89[n])
                Ur = (-1j*Hr*self.dt).expm()
                for k in range(self.N_k):
                    forward[k] = Ur*forward[k]
                    forward_propagator_new[k][n+1] = forward[k]
            
            forward_propagator = forward_propagator_new

            tau = 0
            for k in range(self.N_k):
                tau += np.real(((self.target_state[k].conj().trans())*forward_propagator[k][self.N_T]))
            self.JT_final[i] = 1-(tau/self.N_k)

            print("Iteration=%d, JT=%.10f" % tuple([i+1,self.JT_final[i]]))

            if i > 0:
                delta_JT = self.JT_final[i] - self.JT_final[i-1]
                if delta_JT > 0:
                    print("Not monotonous here!")
                    break
            
            if (i+1) % 100 == 0:
                self.save()

    def plot_C(self):
        plt.figure()
        plt.step(self.tlist,self.J_12,label='J_12',where='mid',color='blue')
        plt.step(self.tlist,self.J_23,label='J_23',where='mid',color='red')
        plt.step(self.tlist,self.J_34,label='J_34',where='mid',color='green')
        plt.step(self.tlist,self.J_45,label='J_45',where='mid',color='yellow')
        plt.step(self.tlist,self.J_56,label='J_56',where='mid',color='pink')
        plt.step(self.tlist,self.J_67,label='J_67',where='mid',color='orange')
        plt.step(self.tlist,self.J_78,label='J_78',where='mid',color='cyan')
        plt.step(self.tlist,self.J_89,label='J_89',where='mid',color='purple')
        plt.plot(self.tlist,np.zeros(self.N_T),color='black',ls='-',lw=2)
        plt.legend()
        plt.show()

    def save(self):
        self.data = {"J_12":self.J_12,"J_23":self.J_23,"J_34":self.J_34,"J_45":self.J_45,"J_56":self.J_56,"J_67":self.J_67,"J_78":self.J_78,"J_89":self.J_89}
        with open(self.datapath,"wb") as f:
            pickle.dump(self.data,f)
    
    def test(self):
        U = qt.qeye(90)
        for n in range(self.N_T):
            Hr = self._Hamiltonian(J_12=self.J_12[n],J_23=self.J_23[n],J_34=self.J_34[n],J_45=self.J_45[n],J_56=self.J_56[n],J_67=self.J_67[n],J_78=self.J_78[n],J_89=self.J_89[n])
            Ur = (-1j*Hr*self.dt).expm()
            U = Ur*U
        return U.full()
