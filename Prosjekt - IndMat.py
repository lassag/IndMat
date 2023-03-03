import numpy as np
import matplotlib.pyplot as plt

test = np.load('test.npy')/255.0
train = np.load('train.npy')/255.0
testsize = test.shape[2]
trainsize = train.shape[2]
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

def getclasses(k,d):
    C = np.swapaxes(train[:,:10,:k],0,1)
    W = np.zeros((10,pixels,d))
    H = np.zeros((10,d,k))
    for i in range(10):
        W[i], H[i] = WHfact(C[i],d)
    return C, W, H

def classify(k,d,projectiontype="orth"):
    B = np.swapaxes(np.swapaxes(test,0,1),1,2)
    C, W, H = getclasses(k,d)
    P = np.zeros((10,10,pixels,testsize))
    D = np.zeros((10,10,testsize))
    
    for i in range(10):
        for j in range(10):
            P[j,i] = orthproj(W[j],B[i].T)
            D[i,j] = dist(P[j,i],B[i].T)
    
    return B, P, D, np.argmin(D, axis=0)

def display(B,P,D,C,c,r):
    plt.imshow(B[c,r].reshape((28,28)), cmap = 'gray')
    plt.axis('off')
    plt.show()
    fig, axes = plt.subplots(2,5)
    count = 0
    for i in range(2):
        for j in range(5):
            axes[i,j].imshow(P[count,c,:,r].reshape((28,28)), cmap = 'gray')
            if C[c,r] == count:
                axes[i,j].set_title(f'{count}', color = 'red')
            else:
                axes[i,j].set_title(f'{count}')
            axes[i,j].axis('off')
            count += 1
    plt.show()
    print(f'Skår: \n {D[:,c,r].reshape((2,5))}')
    print(f'Gjeting: {C[c,r]}\n Riktig: {c}')
    
def accuracy(C,indecies=[0,1,2,3,4,5,6,7,8,9]):
    A = np.tile(indecies,(testsize,1)).T
    return np.sum(A == C[indecies]) / C[indecies].size

def showaccuracy(C,indecies=[0,1,2,3,4,5,6,7,8,9]):
    A = np.zeros(len(indecies))
    for i in range(len(indecies)):
        A[i] = accuracy(C,i)
    plt.plot(indecies,A)
    plt.scatter(indecies,A)
    plt.xticks(indecies)
    plt.yticks(np.linspace(0, 1, num=11, endpoint=True))
    plt.title("Treffsikkerheit")
    plt.xlabel("Siffer")
    plt.ylabel("Riktige gjetingar")
    plt.show()
    print(f'Total treffsikkeheit: {np.round(np.sum(A)/A.size * 100,1)}%')

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

# plotimgs(U, 4)
# plotimgs(Ud, 4)

#checkalld(U,S,Vt,B,"orth")
#########################################

#Klassifisering
k = 1000 #Tal på treningsdatapunkt
d = 32 #Trunkeringskoeffisient
c = 7 #Klasse for test
r = 58 #Nummer for test
indecies = np.array([0,1,2,3,4,5,6,7,8,9])

# A, W, H = getclasses(k,d)
# U, S, Vt = np.linalg.svd(A[c], full_matrices=False)
# plt.semilogy(S)
# plt.show()

B, P, D, C = classify(k,d)
display(B, P, D, C, c, r)
showaccuracy(C,indecies)

###################################

# Q = np.ones(10)*2
# for i in range(10):
#     Q[i] = Q[i]**i
# acc = np.array([36.2, 47.7, 60.5, 68.4, 76.2, 77.1, 74.2, 69.8, 67.6, 39.0]) / 100
# plt.semilogx(Q,acc, base=2, subs=None)
# plt.title("Treffsikkerheit")
# plt.xticks(Q)
# plt.xlabel("Trunkeringskoeffisient")
# plt.yticks(np.linspace(0, 1, num=11, endpoint=True))
# plt.ylabel("Riktige gjetingar")
# plt.show()

##################################