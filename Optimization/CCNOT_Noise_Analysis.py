import os
import scipy.io
import pickle
import numpy as np
import qutip as qt

class CCNOT_Noise():
    def __init__(self,N_T=55):
        self.dirpath = os.path.dirname(__file__)

        self.dJ = 0

        self.dt = 1
        self.J0 = np.pi/self.dt
        self.N_T = N_T
        self.T = self.N_T*self.dt

        self.tlist = np.zeros(self.N_T)
        for i in range(self.N_T):
            self.tlist[i] = (i+0.5)*self.dt

        J_path = self.dirpath + os.sep + "data_CCNOT_55.pickle"
        with open(J_path,"rb") as f:
            J = pickle.load(f)

        self.J_12 = J["J_12"]
        self.J_23 = J["J_23"]
        self.J_34 = J["J_34"]
        self.J_45 = J["J_45"]
        self.J_56 = J["J_56"]
        self.J_67 = J["J_67"]
        self.J_78 = J["J_78"]
        self.J_89 = J["J_89"]

    def load_H(self):
        H_path = self.dirpath + os.sep + "H_3_qubit.mat"
        H = scipy.io.loadmat(H_path)

        self.H_12 = (1+self.dJ)*qt.Qobj(H["H_12"])
        self.H_23 = (1+self.dJ)*qt.Qobj(H["H_23"])
        self.H_34 = (1+self.dJ)*qt.Qobj(H["H_34"])
        self.H_45 = (1+self.dJ)*qt.Qobj(H["H_45"])
        self.H_56 = (1+self.dJ)*qt.Qobj(H["H_56"])
        self.H_67 = (1+self.dJ)*qt.Qobj(H["H_67"])
        self.H_78 = (1+self.dJ)*qt.Qobj(H["H_78"])
        self.H_89 = (1+self.dJ)*qt.Qobj(H["H_89"])

    def _Hamiltonian(self,J_12=0,J_23=0,J_34=0,J_45=0,J_56=0,J_67=0,J_78=0,J_89=0):
        return self.J0*(J_12*self.H_12+J_23*self.H_23+J_34*self.H_34+J_45*self.H_45+J_56*self.H_56+J_67*self.H_67+J_78*self.H_78+J_89*self.H_89)

    def U_search(self):
        U = qt.qeye(90)
        for n in range(self.N_T):
            Hr = self._Hamiltonian(J_12=self.J_12[n],J_23=self.J_23[n],J_34=self.J_34[n],J_45=self.J_45[n],J_56=self.J_56[n],J_67=self.J_67[n],J_78=self.J_78[n],J_89=self.J_89[n])
            Ur = (-1j*Hr*self.dt).expm()
            U = Ur*U
        U_numpy = U.full()

        CCNOT_U_numpy = np.zeros((24,24),dtype=np.complex128)
        for k in range(3):
            for i in range(8):
                for j in range(8):
                    CCNOT_U_numpy[i+8*k][j+8*k] = U_numpy[i-k+11*k**2][j-k+11*k**2]
        self.CCNOT_U_search = qt.Qobj(CCNOT_U_numpy)

    def U_decomposition(self):
        p1 = np.arccos(2*np.sqrt(3)/3-1)/np.pi-1
        p2 = np.arcsin((2*np.sqrt(3)-1)/3)/np.pi
        R12 = ((-1j*0.5*np.pi*self.H_34).expm())*((-1j*1.5*np.pi*self.H_45).expm())*((-1j*1*np.pi*self.H_34).expm())*((-1j*1*np.pi*self.H_56).expm())*((-1j*0.5*np.pi*self.H_45).expm())*((-1j*1.5*np.pi*self.H_34).expm())
        CNOT12 = ((-1j*p1*np.pi*self.H_45).expm())*((-1j*p2*np.pi*self.H_56).expm())*R12*((-1j*1*np.pi*self.H_23).expm())*R12*((-1j*1*np.pi*self.H_23).expm())*R12*((-1j*(2-p2)*np.pi*self.H_56).expm())*((-1j*(2-p1)*np.pi*self.H_45).expm())

        R23 = ((-1j*0.5*np.pi*self.H_67).expm())*((-1j*1.5*np.pi*self.H_78).expm())*((-1j*1*np.pi*self.H_67).expm())*((-1j*1*np.pi*self.H_89).expm())*((-1j*0.5*np.pi*self.H_78).expm())*((-1j*1.5*np.pi*self.H_67).expm())
        CNOT23 = ((-1j*p1*np.pi*self.H_78).expm())*((-1j*p2*np.pi*self.H_89).expm())*R23*((-1j*1*np.pi*self.H_56).expm())*R23*((-1j*1*np.pi*self.H_56).expm())*R23*((-1j*(2-p2)*np.pi*self.H_89).expm())*((-1j*(2-p1)*np.pi*self.H_78).expm())

        SWAP23 = ((-1j*1*np.pi*self.H_67).expm())*((-1j*1*np.pi*self.H_56).expm())*((-1j*1*np.pi*self.H_45).expm())*((-1j*1*np.pi*self.H_78).expm())\
                *((-1j*1*np.pi*self.H_67).expm())*((-1j*1*np.pi*self.H_56).expm())*((-1j*1*np.pi*self.H_89).expm())*((-1j*1*np.pi*self.H_78).expm())*((-1j*1*np.pi*self.H_67).expm())
        CNOT13 = SWAP23*CNOT12*SWAP23

        theta = 1+np.arccos((2-np.sqrt(2)+np.sqrt(70+36*np.sqrt(2)))/12)/np.pi
        T1Gate = ((-1j*theta*np.pi*self.H_12).expm())*((-1j*(1.25-theta)*np.pi*self.H_23).expm())*((-1j*theta*np.pi*self.H_12).expm())*((-1j*(1-theta)*np.pi*self.H_23).expm())
        T2Gate = ((-1j*theta*np.pi*self.H_45).expm())*((-1j*(1.25-theta)*np.pi*self.H_56).expm())*((-1j*theta*np.pi*self.H_45).expm())*((-1j*(1-theta)*np.pi*self.H_56).expm())
        T3Gate = ((-1j*theta*np.pi*self.H_78).expm())*((-1j*(1.25-theta)*np.pi*self.H_89).expm())*((-1j*theta*np.pi*self.H_78).expm())*((-1j*(1-theta)*np.pi*self.H_89).expm())

        phi1 = np.arccos(-1+np.sqrt(2)+(-2+np.sqrt(2))/np.sqrt(3))/np.pi
        phi2 = np.arccos((-1+np.sqrt(2)+np.sqrt(6))/3)/np.pi
        H3Gate = ((-1j*(phi1-1)*np.pi*self.H_78).expm())*((-1j*(1-phi2)*np.pi*self.H_89).expm())*((-1j*(1+phi2)*np.pi*self.H_78).expm())*((-1j*(2-phi1)*np.pi*self.H_89).expm())

        U = H3Gate*CNOT23*(T3Gate.dag())*CNOT13*T3Gate*CNOT23*(T3Gate.dag())*CNOT13*T2Gate*T3Gate*CNOT12*H3Gate*T1Gate*(T2Gate.dag())*CNOT12
        U_numpy = U.full()

        CCNOT_U_numpy = np.zeros((24,24),dtype=np.complex128)
        for k in range(3):
            for i in range(8):
                for j in range(8):
                    CCNOT_U_numpy[i+8*k][j+8*k] = U_numpy[i-k+11*k**2][j-k+11*k**2]
        self.CCNOT_U_decomposition = qt.Qobj(CCNOT_U_numpy)

    def inf(self,mode=0):
        CCNOT_U0_numpy = np.zeros((24,24),dtype=np.complex128)
        for i in range(24):
            if i < 6:
                CCNOT_U0_numpy[i][i] = 1
            elif i == 6:
                CCNOT_U0_numpy[i][i+1] = 1
            elif i == 7:
                CCNOT_U0_numpy[i][i-1] = 1
            elif i < 14:
                CCNOT_U0_numpy[i][i] = 1
            elif i == 14:
                CCNOT_U0_numpy[i][i+1] = 1
            elif i == 15:
                CCNOT_U0_numpy[i][i-1] = 1
            elif i < 22:
                CCNOT_U0_numpy[i][i] = 1
            elif i == 22:
                CCNOT_U0_numpy[i][i+1] = 1
            else:
                CCNOT_U0_numpy[i][i-1] = 1
        self.CCNOT_U0 = qt.Qobj(CCNOT_U0_numpy)

        if mode == 0:
            tracenorm = (np.abs(((self.CCNOT_U0.dag())*self.CCNOT_U_decomposition).tr()))**2
        else:
            tracenorm = (np.abs(((self.CCNOT_U0.dag())*self.CCNOT_U_search).tr()))**2

        inf = 1-(24+tracenorm)/(24*25)
        return inf
