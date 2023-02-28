import numpy as np
import matplotlib.pyplot as plt

#Funksjonar
def svd(A):
    U,S,Vt = np.linalg.svd(A1, full_matrices = False)
    return U, S, Vt

def truncSVD(U,S,Vt,d):
    return U[:,:d], S[:d], Vt[:d]

def WHfact(A,d):
    U,S,Vt = np.linalg.svd(A, full_matrices=False)
    Ud, Sd, Vtd = truncSVD(U,S,Vt,d)
    
    W = Ud
    H = Sd*Vtd
    return W, H

def orthproj(W,b):
    return W@(W.T@b)

def dist(W,b):
    P = orthproj(W,b)
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

W, H = WHfact(A1,3)

D = dist(W,B)
print(D)