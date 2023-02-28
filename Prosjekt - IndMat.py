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

##############################################################################

test = np.load('test.npy')/255.0
train = np.load('train.npy')/255.0


print(test.shape)
print(train.shape)


def plotimgs(imgs, nplot = 4):
    """
    Plots the nplot*nplot first images in imgs on an nplot x nplot grid. 
    Assumes heigth = width, and that the images are stored columnwise
    input:
        imgs: (height*width,N) array containing images, where N > nplot**2
        nplot: integer, nplot**2 images will be plotted
    """

    n = imgs.shape[1]
    m = int(np.sqrt(imgs.shape[0]))

    assert(n > nplot**2), "Need amount of data in matrix N > nplot**2"

    # Initialize subplots
    fig, axes = plt.subplots(nplot,nplot)

    # Set background color
    plt.gcf().set_facecolor("lightgray")

    # Iterate over images
    for idx in range(nplot**2):

        # Break if we go out of bounds of the array
        if idx >= n:
            break

        # Indices
        i = idx//nplot; j = idx%nplot

        # Remove axis
        axes[i,j].axis('off')

        axes[i,j].imshow(imgs[:,idx].reshape((m,m)), cmap = "gray")
    
    # Plot

    fig.tight_layout()
    plt.show()
    
plotimgs(train[:,0,:], nplot = 4)


# Plot the second image of the 2 digit
# Note that we have to reshape it to be 28 times 28!
plt.imshow(train[:, 2, 1].reshape((28,28)), cmap = 'gray')
plt.axis('off')
plt.show()

n = 1000 # Number of datapoints
c = 7 # Class

A = train[:,c,:n]

print(A.shape) # Expect (784,n)

##############################################################################
'''
U7, S7, Vt7 = svd(A)
#print(U7, S7, Vt7)

diagonal_S7 = np.diag(S7)
W7, H7 = truncSVD(U7, S7, Vt7, 16)
print(W7.shape)
print(H7.shape)

plotimgs(W7, nplot = 4)
'''

#2b)
U7, S7, Vt7 = svd(A)
#print(U7, S7, Vt7)

diagonal_S7 = np.diag(S7)
W7,H7,Vt7 = truncSVD(U7, diagonal_S7, Vt7, 17)
print(W7.shape)
print(H7.shape)


#2c
plotimgs(W7, nplot = 4)

plt.semilogy(S7)
plt.show()