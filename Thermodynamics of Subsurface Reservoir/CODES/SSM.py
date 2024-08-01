import pandas as pd
import numpy as np
import torch
from scipy.stats import norm
import matplotlib.pyplot as plt

R = 8.3144598*1e-6 #MPa../K..

Nc = 9
components = ["C1","C2","C3","nC4","nC5","C6","C7+","CO2","N2"]
# Component Properties
CP = {
    "C1":  {"Pc":4.6000, "Tc":190.55, "w":0.0111, "MW": 16.04000},
    "C2":  {"Pc":4.8750, "Tc":305.43, "w":0.0970, "MW": 30.06904},
    "C3":  {"Pc":4.2680, "Tc":369.82, "w":0.1536, "MW": 44.09562},
    "nC4": {"Pc":3.7960, "Tc":425.16, "w":0.2008, "MW": 58.12000},
    "nC5": {"Pc":3.3334, "Tc":467.15, "w":0.2635, "MW": 72.15000},
    "C6":  {"Pc":2.9688, "Tc":507.40, "w":0.2960, "MW": 86.17536},
    "C7+": {"Pc":2.6220, "Tc":604.50, "w":0.3565, "MW": 100.2000},
    "CO2": {"Pc":7.3820, "Tc":304.19, "w":0.2250, "MW": 44.01000},
    "N2":  {"Pc":3.3944, "Tc":126.25, "w":0.0390, "MW": 28.02000}
}
CP_df = pd.DataFrame(CP)

############# DATA PREPARING ###############
# Four fluid types characterized by different compositional ranges
FT = {
    "wet gas":         {"C1": 100,"C2": 7,"C3": 3,"nC4": 2,"nC5": 2,"C6": 2,"C7+": 1,"CO2": 2,"N2": 0.5}
#    "gas condencate": {"C1": ,"C2": ,"C3": ,"nC4": ,"nC5": ,"C6": ,"C7+": ,"CO2": ,"N2":},
#    "volatile oil":   {"C1": ,"C2": ,"C3": ,"nC4": ,"nC5": ,"C6": ,"C7+": ,"CO2": ,"N2":},
#    "black oil":      {"C1": ,"C2": ,"C3": ,"nC4": ,"nC5": ,"C6": ,"C7+": ,"CO2": ,"N2":}
}
FT_df = pd.DataFrame(FT)

def z_generator(alpha, Nc):
    y = np.random.gamma(alpha, 1, Nc)
    z = y/np.sum(y)
    return z

z_samples_wet_gas = {}
Nz = 50
for i in range(Nz):
    z_g = z_generator(np.array(FT_df["wet gas"]), Nc)
    z_samples_wet_gas["sample_{}".format(i)] = {"C1": z_g[0],"C2": z_g[1],"C3": z_g[2],"nC4": z_g[3],"nC5": z_g[4],"C6": z_g[5],"C7+": z_g[6],"CO2": z_g[7],"N2": z_g[8]}

z_samples_wet_gas_df = pd.DataFrame(z_samples_wet_gas)


P1 = 5  #MPa
P2 = 25 #MPa
Np = 100
T1 = 200 # K
T2 = 600 # K
Nt = 25

from pyDOE import lhs
np.random.seed(2)
samplesP = np.sort((np.array([P1]) +(np.array([P2]) - np.array([P1]))*lhs(1,Np)).reshape((-1)))
np.random.seed(3)
samplesT = np.sort((np.array([T1]) +(np.array([T2]) - np.array([T1]))*lhs(1,Nt)).reshape(-1))

sample_no = 0
sample_data = {}
for i in range(Np):
    for j in range(Nt):
        for k in range(Nz):
            aux = dict(z_samples_wet_gas_df["sample_{}".format(k)])
            sample_data["sample_{}".format(sample_no)] = {"P": samplesP[i],"T": samplesT[j]}
            for name in range(Nc):
                sample_data["sample_{}".format(sample_no)][components[name]] = z_samples_wet_gas_df["sample_{}".format(k)][name]
            sample_no += 1
#
sample_data_df = pd.DataFrame(sample_data).T

sample_data_df.to_csv(r'C:\Users\YUSIFOH\NNs\thermo_project\sample_data\sample_data_df.csv', index=False, encoding='utf-8')
########################################### SSM ###########################
sample_data_df = pd.read_csv(r'C:\Users\YUSIFOH\NNs\thermo_project\sample_data\sample_data_df.csv')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Calculation of   Binary Interaction Matrix (BIM) k 
MW = torch.tensor(np.array(CP_df))[3].to(device=device)
k = torch.zeros(Nc, Nc)
k_CO2, k_N2 = torch.tensor([0.15], dtype=torch.float), torch.tensor([0.1], dtype=torch.float)
n=0
for i in range(Nc):
    for j in range(Nc):
        if j>=n:
            if i == j:
                k[i, j] = torch.tensor([0.0], dtype=torch.float)
                k[j, i] = torch.tensor([0.0], dtype=torch.float)

            elif (i == 7 and j < 7) or (j == 7 and i < 7):
                k[i, j] = k_CO2
                k[j, i] = k_CO2

            elif (i == 8 and j < 7) or (j == 8 and i < 7):
                k[i, j] = k_N2
                k[j, i] = k_N2

            elif i >= 7 and j >= 7:
                k[i, j] = torch.tensor([0.0], dtype=torch.float)
                k[j, i] = torch.tensor([0.0], dtype=torch.float)
            
            else:
                k[i, j] = (0.0289 + 1.633*MW[j]*1e-04).float()
                k[j, i] = (0.0289 + 1.633*MW[j]*1e-04).float()
    n+=1
k = k.to(device=device)

Tc = torch.tensor(np.array(CP_df))[1].to(device=device)
Pc = torch.tensor(np.array(CP_df))[0].to(device=device)
w = torch.tensor(np.array(CP_df))[2].to(device=device)

# Rachford-Rice equation and its derivative 
def f(V, K, z):
    return torch.sum((z*(K-1))/(1 - V + V*K))
def dfdV(V, K, z):
    return torch.sum(-z*((K -1)/(1 - V + V*K))**2)

#Newton-Raphson Method to calculate V
def NewtonRaph(K, z, tol = 1e-6, V0 = 1, max_iter = 50):
    e = 100
    iter = 0
    while e > tol:
        f0 = f(V0, K, z)
        dfdV0 = dfdV(V0, K, z)
        V = V0 - (f0/dfdV0)
        V0 = V
        e = torch.abs(f(V0, K, z)).item()
        iter+=1
        #if iter > max_iter:
        #    break
    return V

# Calculation of Molar Composition of xi and yi 
def x(V, K, z):
    return z/(1 - V + V*K)

def y(V, K, z):
    return (K*z)/(1 - V + V*K)

# Peng-Robinson Equation

def aa_b_m(PhaseMol, a, b, k, alpha):
    X = PhaseMol
    xi, xj = torch.meshgrid(X, X.T)
    xixj = xi*xj

    ai, aj = torch.meshgrid(a, a.T)
    aiaj = ai*aj

    ali, alj = torch.meshgrid(alpha, alpha.T)
    alialj= ali*alj

    aam = torch.sum(xixj*torch.sqrt(aiaj*alialj)*(1 - k))
    bm = torch.sum(X*b)
    return aam, bm

def A_B(PhaseMol, a, b, k, alpha, P, T):
    aam, bm = aa_b_m(PhaseMol, a, b, k, alpha)
    A = aam*P/(R*T)**2
    B = bm*P/(R*T)

    return [A, B]

def Psi(PhaseMol, a, k, alpha):
    X= PhaseMol
    ai, aj = torch.meshgrid(a, a.T)
    aiaj = ai*aj

    ali, alj = torch.meshgrid(alpha, alpha.T)
    alialj= ali*alj

    psi = torch.sum(X*torch.sqrt(aiaj*alialj)*(1 - k), axis = 1)
    return psi

def PR_poly(A, B, Z):
    c1, c2, c3, c4 = 1, (B -1), (A - 3*B**2 - 2*B), -(A*B - B**2 - B**3)
    return c1*Z**3 + c2*Z**2 + c3*Z + c4

def PR_roots(A,B):
    c1, c2, c3, c4 = 1.0, (B -1), (A - 3*B**2 - 2*B), -(A*B - B**2 - B**3)
    roots = torch.tensor(np.roots([c1, c2.item(), c3.item(), c4.item()])).to(device = device)
    real_roots= roots[torch.isreal(roots)]
    return torch.real(real_roots)

def fugacity(x, b, bm,aam, psi, Z, P, A, B):
    c1 = (b*(Z - 1)/bm)

    c2 = torch.log(Z - B)

    c3 = (A/(2*torch.sqrt(torch.tensor(2.0))*B))

    c4 = ((2*psi/aam) - (b/bm))

    c5 = (Z +(1 + torch.sqrt(torch.tensor(2.0)))*B)

    c6 = (Z +(1 - torch.sqrt(torch.tensor(2.0)))*B)

    c7 = torch.log(c5/c6)

    f = x*P*torch.exp(c1 - c2 - c3*c4*c7)
    return f

class PTEOS: #  T   o calculate derivative of Fugacity
    def __init__(self):
        pass

    def daamdn(self, a, alpha, k, n):
        n = n.view(len(n))
        a = a.view(len(a))
        alpha = alpha.view(len(alpha))
        return 2*torch.sqrt(a*alpha)*torch.sum(torch.meshgrid(n, n)[1]*torch.sqrt(torch.meshgrid(a*alpha, a*alpha)[1]))
    def dbmdn(self, b):
        return b

    def dAdn(self, P, T, da):
        return (P/(R*T)**2)*da
    
    def dBdn(self, P, T, db):
        return (P/(R*T))*db
    
    def dZdn(self, Z, A, B, dA, dB):
        c0, c1, c2, c3 = 1, (B -1), (A - 3*B**2 - 2*B), -(A*B - B**2 - B**3)

        dc1 = -dB
        dc2 = dA - (2+6*B)*dB
        dc3 = B*dA + (A - 2*B -3*B**2)*dB

        return (dc1*Z**2 - dc2*Z + dc3)/(3*Z**2 - 2*c1*Z + c2)
    
    def AA_BB(self,b, am, bm, psi):
        return (2/am)*psi, b/bm
    
    def dAA(self, am, a, alpha, psi, AA, dam):
        return -(AA/am)*dam + (2/am)*a*alpha
    
    def dBB(self, BB, bm, dbm):
        return -(BB/bm)*dbm
    
    def dfugdn(self, A, B, dA, dB, AA, BB, dAA, dBB, Z, dZ, fug):

        e1 = BB*(Z-1)
        e2 = torch.log(Z -B)
        e3 = A/(B*2*2**0.5)
        e4 = AA - BB
        a1 = (Z +(1 + 2*2**0.5)*B)
        a2 = (Z +(1 - 2*2**0.5)*B)
        a3 = a1/a2
        e5 = torch.log(a3)

        de1 = (Z-1)*dBB + BB*dZ
        de2 = (1/(Z-B))*(dZ - dB)
        de3 = (1/(2*2**0.5))*((1/B)*dA - (A/B**2)*dB)
        de4 = dAA - dBB
        de5 = (1/a3)*((a2*(dZ + (1 + 2*2**0.5)*dB) - a1*(dZ + (1 + 2*2**0.5)*dB)) / (a2**2))

        de3e4e5 = e4*e5*de3 + e3*e5*de4 + e3*e4*de5

        dfug = fug*(de1 - de2 - de3e4e5)

        return dfug
    

a = 0.45724*((R*Tc)**2)/Pc
b = 0.07780*(R*Tc)/Pc
m = torch.zeros(Nc,).to(device=device)
m[(w<=0.49)] = (0.3796 + 1.54226*w[(w<=0.49)] - 0.2699*w[(w<=0.49)]**2).float()
m[~(w<=0.49)] = (0.379642 + 1.48503*w[~(w<=0.49)] - 0.1644*w[~(w<=0.49)]**2 + 0.016667*w[~(w<=0.49)]**3).float()
P = torch.tensor(np.array(sample_data_df["P"])).to(device=device)
T = torch.tensor(np.array(sample_data_df["T"])).to(device=device)
z = torch.tensor(np.array(sample_data_df))[:, 2:].to(device=device)

def fugacity_coef(P, T, Y):
    Tr = T/Tc
    alpha = (1 + m*(1 - torch.sqrt(Tr)))**2

    #Calculate Z
    A, B = A_B(Y, a, b, k, alpha, P, T)
    roots_ = PR_roots(A, B)
    Z = torch.max(roots_)

    #Calculate f
    aam, bm = aa_b_m(Y, a, b, k, alpha)
    psi = Psi(Y, a, k, alpha)
    f = fugacity(Y, b, bm, aam, psi, Z, P, A, B)

    fug_coeff = f/P

    return fug_coeff

def tpd(P, T, h, z):
    phi_h = fugacity_coef(P, T, h)
    phi_z = fugacity_coef(P, T, z)
    return torch.sum(h*(torch.log(h) + torch.log(phi_h) - torch.log(z) - torch.log(phi_z)))

def tm(P,T, W, z):
    phi_W = fugacity_coef(P, T, W)
    phi_z = fugacity_coef(P, T, z)
    return torch.sum(W*(torch.log(W) + torch.log(phi_W) - torch.log(z) - torch.log(phi_z) - 1))


def stability_analysis(P, T, W_0, z):
    e = 100.0
    tol = 1e-06
    phi_z = fugacity_coef(P, T, z)
    max_iter = 10
    iter = 0
    while e > tol and iter < max_iter:
        if tm(P,T, W_0, z).item() < 0:
            return True
        phi_W_0 = fugacity_coef(P, T, W_0)
        W_1 = torch.exp(torch.log(z) + torch.log(phi_z) - torch.log(phi_W_0))
        W_0 = W_1
        iter+=1


def SSM_PT_flash(P, T, z):
    Tr = T/Tc
    alpha = (1 + m*(1 - torch.sqrt(Tr)))**2
    tol = 1e-12
    e = 10000.0
    # Calculate Ki using Wilson's Correlation(Initial Guess) 
    K_n = (Pc/P)*torch.exp(5.37*(1 + w)*(1 - (Tc/T)))
    while e >tol:
        #calculation of V
        V = NewtonRaph(K_n, z, tol = 1e-6, V0 = 1)
        #Flash Calc for xi, yi
        X = x(V, K_n, z)
        Y = y(V, K_n, z)

        #Calculate Z_L and Z_V by PR-EOS
        ## For Liquid
        A_L, B_L = A_B(X, a, b, k, alpha, P, T)
        roots_L = PR_roots(A_L, B_L)
        Z_L = torch.min(roots_L)
        ## For Vapor
        A_V, B_V = A_B(Y, a, b, k, alpha, P, T)
        roots_V = PR_roots(A_V, B_V)
        Z_V = torch.max(roots_V)

        #Calculate f_L and f_V 

        ##For Liquid
        aam_L, bm_L = aa_b_m(X, a, b, k, alpha)
        psi_L = Psi(X, a, k, alpha)
        f_L = fugacity(X, b, bm_L,aam_L, psi_L, Z_L, P, A_L, B_L)
        ##For Vapor
        aam_V, bm_V = aa_b_m(Y, a, b, k, alpha)
        psi_V = Psi(Y, a, k, alpha)
        f_V = fugacity(Y, b, bm_V,aam_V, psi_V, Z_V, P, A_V, B_V)
        e = (torch.sum(((f_L/f_V) - 1)**2)).item()
        K_n = K_n*(f_L/f_V)
    print(e, "  and  ", V)

    fug_coeff_L = f_L/P
    fug_coeff_V = f_V/P

    results = {"phi_L": fug_coeff_L, "phi_V":fug_coeff_V, "K": K_n, "V": V, "X": X, "Y": Y}
    return results

def SSM(P, T, z, K_n, V): # Only one iteration
    Tr = T/Tc
    alpha = (1 + m*(1 - torch.sqrt(Tr)))**2
    #Flash Calc for xi, yi
    X = x(V, K_n, z)
    Y = y(V, K_n, z)

    #Calculate Z_L and Z_V by PR-EOS
    ## For Liquid
    A_L, B_L = A_B(X, a, b, k, alpha, P, T)
    roots_L = PR_roots(A_L, B_L)
    Z_L = torch.min(roots_L)
    ## For Vapor
    A_V, B_V = A_B(Y, a, b, k, alpha, P, T)
    roots_V = PR_roots(A_V, B_V)
    Z_V = torch.max(roots_V)

    #Calculate f_L and f_V 

    ##For Liquid
    aam_L, bm_L = aa_b_m(X, a, b, k, alpha)
    psi_L = Psi(X, a, k, alpha)
    f_L = fugacity(X, b, bm_L,aam_L, psi_L, Z_L, P, A_L, B_L)
    ##For Vapor
    aam_V, bm_V = aa_b_m(Y, a, b, k, alpha)
    psi_V = Psi(Y, a, k, alpha)
    f_V = fugacity(Y, b, bm_V,aam_V, psi_V, Z_V, P, A_V, B_V)
    e = (torch.sum(((f_L/f_V) - 1)**2)).item()
    K_n = K_n*(f_L/f_V)
    fug_coeff_L = f_L/P
    fug_coeff_V = f_V/P

    return {"phi_L": fug_coeff_L, "phi_V":fug_coeff_V, "K": K_n, "X": X, "Y": Y}

# Stability Analysis part 1: Selecting 0<V<1

selected_index = []
for i in range(len(P)):
    K_n = (Pc/P[i])*torch.exp(5.37*(1 + w)*(1 - (Tc/T[i])))
    for j in range(9):
        V_ = NewtonRaph(K_n, z[i], tol = 1e-6, V0 = 1)
        if 0 < V_ < 1:
            results = SSM(P[i], T[i], z[i], K_n, V_)
            K_n = results["K"]
            if j == 8:
                print("Index= {}".format(i))
                selected_index.append(i)

selected_index_f1_v2 = np.array(selected_index)
np.savetxt(r"C:\Users\YUSIFOH\NNs\thermo_project\sample_data\selected_index_f1_v2.dat", selected_index_f1_v2)
selected_index_f1_v2 =  np.genfromtxt(r"C:\Users\YUSIFOH\NNs\thermo_project\sample_data\selected_index_f1_v2.dat")
P_f1 = P[selected_index_f1_v2]
T_f1 = T[selected_index_f1_v2]
z_f1 = z[selected_index_f1_v2]

# Stability Analysis Part 2: tpd_x < 0 or tpd_y < 0 or dG < 0 and tm(W) < 0
selected_index_f2 = []
for i in range(len(P_f1)):
    #print("Index = {} = is being checked".format(i))
    K_n = (Pc/P_f1[i])*torch.exp(5.37*(1 + w)*(1 - (Tc/T_f1[i])))
    for j in range(3):
        V_ = NewtonRaph(K_n, z_f1[i], tol = 1e-6, V0 = 1)
        results = SSM(P_f1[i], T_f1[i], z_f1[i], K_n, V_)
        x_ = results["X"]
        y_ = results["Y"]
        K_n = results["K"]
        tpd_x = tpd(P_f1[i], T_f1[i], x_, z_f1[i])
        tpd_y = tpd(P_f1[i], T_f1[i], y_, z_f1[i])
        dG = tpd_x*V_ + tpd_y*(1 - V_)
        if tpd_x < 0 or tpd_y < 0 or dG < 0:
            selected_index_f2.append(i)
            print("Index = {} = is Unstable".format(i))
            print("Total Selected Indices = {}".format(len(selected_index_f2)))
            break
        else: 
            if j == 2:
                W_0_v = K_n*z_f1[i]
                W_0_l = z_f1[i]/K_n
                if stability_analysis(P_f1[i], T_f1[i], W_0_v, z_f1[i]) or stability_analysis(P_f1[i], T_f1[i], W_0_l, z_f1[i]):
                    selected_index_f2.append(i)
                    print("Index = {} = is Unstable".format(i))
                    print("Total Selected Indices = {}".format(len(selected_index_f2)))

selected_index_f2 = np.array(selected_index_f2)


T_f2 = T_f1[selected_index_f2]
P_f2 = P_f1[selected_index_f2]
z_f2 = z_f1[selected_index_f2]

# Do SSM to collect Ground Truth
results_dict = {}
for i in range(len(T_f2)):
    print("Sample #NO = {}".format(i))
    results = SSM_PT_flash(P_f2[i], T_f2[i], z_f2[i])
    results_dict["sample_{}".format(i)] = results