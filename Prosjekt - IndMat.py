import numpy as np

#Funksjonar
def truncSVD(U,S,Vt,d):
    return U[:,:d], S[:d], Vt[:d]

def WHfact(A,d):
    U,S,Vt = np.linalg.svd(A, full_matrices=False)
    Ud, Sd, Vtd = truncSVD(U,S,Vt,d)
    
    W = Ud
    H = (Sd * Vtd.T).T
    return W, H

def nnproj(A, b, d, maxiter = 50, delta = 10e-10):
    W = A[:,np.random.choice(A.shape[1], size = d, replace = False)]    
    WtW = W.T@W
    Wtb = W.T@b
    H = np.random.uniform(0,1,(W.shape[1],b.shape[1]))
    
    for k in range(maxiter):
        H = H * Wtb / (WtW @ H + delta)
    
    return W@H, H

def orthproj(W,b):
    return W@(W.T@b)

def dist(P,b):
    return np.linalg.norm(b-P, axis = 0)

#Testvektorar oppgåve 1
A1 = np.array([[1000,1],
               [0, 1],
               [0, 0]],dtype=float)
A2 = np.array([[1,0,0],
               [1,0,0],
               [0,0,1]],dtype=float)
b1 = np.array([2,1,0],dtype=float)
b2 = np.array([0,0,1],dtype=float)
b3 = np.array([0,1,0],dtype=float)
B = np.vstack((b1,b2,b3)).T

###############################

Wd, Hd = WHfact(A1,3)
Pd = orthproj(Wd,B)

Dp = dist(Pd,B)


P1, H1 = nnproj(A1,B,2)
P2, H2 = nnproj(A2,B,3)

D1 = dist(P1,B)
D2 = dist(P2,B)

print(Dp)
print(D1)
print(D2)