import numpy as np

#Funksjonar
def truncSVD(U,S,Vt,d):
    return U[:,:d], S[:d,:d], Vt[:d]

def WHfact(A,d):
    U,S,Vt = np.linalg.svd(A2, full_matrices=False); S = np.diag(S)
    Ud, Sd, Vtd = truncSVD(U,S,Vt,2)
    
    W = Ud
    H = Sd@Vtd
    return W, H

def orthproj(W,b):
    return W@(W.T@b)

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

W,H = WHfact(A2,2)

P = orthproj(W,B)
print(P)