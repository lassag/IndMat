import numpy as np
import matplotlib.pyplot as plt

test = np.load('test.npy')/255.0
train = np.load('train.npy')/255.0
pixels = 784

# print(train.shape)
# print(test.shape)

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
    print(b.shape)
    H = np.random.uniform(0,1,(d,b.shape[1]))
    print(H.shape)
    
    for k in range(maxiter):
        H = H * Wtb / (WtW @ H + delta)
    return W@H, H

def orthproj(W,b):
    return W@(W.T@b)

def dist(P,b):
    return np.linalg.norm(b-P, axis = 0)

def selectimg(source, digit, number):
    if source == "train":
        return train[:,digit,number]
    elif source == "test":
        return test[:,digit,number]
    
def dicttrain(digit,number):
    return train[:,digit,:number]

def checkalld(U,S,Vt,b,projectiontype="orth"):
    D = np.zeros(pixels)
    if projectiontype == "orth":
        for i in range(pixels):
            Ud, Sd, Vtd = truncSVD(U,S,Vt,i)
            P = orthproj(Ud,b)
            D[i] = dist(P,b)
    elif projectiontype == "nn":
        for i in range(pixels):
            Ud, Sd, Vtd = truncSVD(U,S,Vt,i)
            P, H = nnproj(Ud,b,i)
            D[i] = dist(P,b)
    plt.semilogy(D)
    plt.show()

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

##############################################################################

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

    assert(n >= nplot**2), "Need amount of data in matrix N > nplot**2"

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


# Plot the second image of the 2 digit
# Note that we have to reshape it to be 28 times 28!


##############################################################################

d = 1000
cA = 3
cB = 3
r = 46
n = 128

A = dicttrain(cA,n)
#b = selectimg("test",3,4)
B = test[:,cB,:]
print(B)

U, S, Vt = np.linalg.svd(A, full_matrices=False)
Ud, Sd, Vtd = truncSVD(U,S,Vt,d)

# Pnn, H = nnproj(Ud,B,d)
# print(Pnn)
# Dnn = dist(Pnn,b)
Porth = orthproj(Ud,B)
print(Porth)
# Dorth = dist(Porth,B)
# print(Dnn)
# print(Dorth)
#plt.semilogy(Sd)
#plt.show()

plt.imshow(B[:,r].reshape((28,28)), cmap = 'gray')
plt.axis('off')
plt.show()
# plt.imshow(Pnn.T[r].reshape((28,28)), cmap = 'gray')
# plt.axis('off')
# plt.show()
plt.imshow(Porth.T[r].reshape((28,28)), cmap = 'gray')
plt.axis('off')
plt.show()


# plotimgs(U, 4)
# plotimgs(Ud, 4)

#checkalld(U,S,Vt,B,"nn")