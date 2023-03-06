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
    H = np.random.uniform(0,1,(d,b.shape[1]))
    
    for k in range(maxiter):
        H = H * Wtb / (WtW @ H + delta)
    return W@H

def orthproj(W,b):
    return W@(W.T@b)

def dist(P,b):
    return np.linalg.norm(b-P, axis = 0)

def checkalld(A,b,projectiontype="orth"):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    if projectiontype == "orth":
        D = np.zeros(b.shape)
        for i in range(pixels):
            Ud, Sd, Vtd = truncSVD(U,S,Vt,i)
            P = orthproj(Ud,b)
            D[i] = dist(P,b)
    elif projectiontype == "nn":
        D = np.zeros((A.shape[1],b.shape[1]))
        for i in range(A.shape[1]):
            Ud, Sd, Vtd = truncSVD(U,S,Vt,i)
            P = nnproj(Ud,b,i)
            D[i] = dist(P,b)
    plt.semilogy(D)
    plt.show()
    
def generatetest(t,indecies=[0,1,2,3,4,5,6,7,8,9]):
    B = np.swapaxes(test[:,:,np.random.choice(test.shape[2], size = t, replace = False)],0,1)
    return B[indecies]

def getclasses(k,d,indecies=[0,1,2,3,4,5,6,7,8,9]):
    C = np.swapaxes(train[:,:,:k],0,1)
    W = np.zeros((10,pixels,d))
    H = np.zeros((10,d,k))
    for i in range(10):
        W[i], H[i] = WHfact(C[i],d)
    return C[indecies], W[indecies], H[indecies]

def classify(B, k, d, t=800, indecies=[0,1,2,3,4,5,6,7,8,9], projectiontype="orth", maxiter = 75, delta = 10e-2):
    ind = len(indecies)
    C, W, H = getclasses(k,d,indecies)
    P = np.zeros((ind,ind,pixels,t))
    D = np.zeros((ind,ind,t))
    
    for i in range(ind):
        for j in range(ind):
            if projectiontype == "orth":
                P[j,i] = orthproj(W[j],B[i])
            elif projectiontype  == "nn":
                P[j,i] = nnproj(C[j],B[i],d, maxiter, delta)
            D[i,j] = dist(P[j,i],B[i])
    
    return P, D, np.argmin(D, axis=0)

def display(B,P,D,C,indecies,c,r):
    plt.imshow(B[c,:,r].reshape((28,28)), cmap = 'gray')
    plt.axis('off')
    plt.show()
    fig, axes = plt.subplots(2,int(len(indecies)/2))
    count = 0
    for i in range(2):
        for j in range(int(np.ceil(len(indecies)/2))):
            axes[i,j].imshow(P[count,c,:,r].reshape((28,28)), cmap = 'gray')
            if indecies[C[c,r]] == indecies[count]:
                axes[i,j].set_title(f'{indecies[count]}', color = 'red')
            else:
                axes[i,j].set_title(f'{indecies[count]}')
            axes[i,j].axis('off')
            count += 1
    plt.show()
    print(f'Skår: \n {D[:,c,r].reshape((2,int(np.ceil(len(indecies)/2))))}')
    print(f'Gjeting: {indecies[C[c,r]]}\n Riktig: {indecies[c]}')
    
def accuracy(C,t=800,indecies=[0,1,2,3,4,5,6,7,8,9]):
    A = np.tile(indecies,(t,1)).T
    return np.sum(A == C) / C.size

def showaccuracy(C,t=800,indecies=[0,1,2,3,4,5,6,7,8,9]):
    A = np.zeros(len(indecies))
    for i in range(len(indecies)):
        A[i] = accuracy(C[i],t,i)
    plt.plot(indecies,A)
    plt.scatter(indecies,A)
    plt.xticks(indecies)
    plt.yticks(np.linspace(0, 1, num=11, endpoint=True))
    plt.title("Treffsikkerheit")
    plt.xlabel("Siffer")
    plt.ylabel("Riktige gjetingar")
    plt.show()
    print(f'Total treffsikkerheit: {np.round(np.sum(A)/A.size * 100,1)}%')

def getaccuracies(k, n, t=800, indecies=[0,1,2,3,4,5,6,7,8,9], projectiontype="orth",maxiter=75,delta=10e-2):
    B = generatetest(t,indecies)
    Q = np.ones(n,dtype = int)*2
    acc = np.zeros(n)
    for i in range(n):
        Q[i] = Q[i]**i
        if projectiontype == "orth":
            P, D, C = classify(B, k,Q[i],t, indecies,"orth")
        elif projectiontype == "nn":
            P, D, C = classify(B, k,Q[i],t, indecies, "nn",maxiter,delta)
        acc[i] = accuracy(C,t,indecies)
    
    plt.semilogx(Q, acc, base=2, subs=None)
    plt.title("Treffsikkerheit")
    plt.xticks(Q)
    plt.xlabel("Trunkeringskoeffisient")
    plt.yticks(np.linspace(0, 1, num=11, endpoint=True))
    plt.ylabel("Riktige gjetingar")
    plt.show()

#Testvektorar oppgåve 1
# A1 = np.array([[1000,1],
#                [0, 1],
#                [0, 0]],dtype=float)
# A2 = np.array([[1,0,0],
#                [1,0,0],
#                [0,0,1]],dtype=float)
# b1 = np.array([2,1,0],dtype=float)
# b2 = np.array([0,0,1],dtype=float)
# b3 = np.array([0,1,0],dtype=float)
# B = np.vstack((b1,b2,b3)).T

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
n = 10 #Tal på toarpotensar 
t = 50 #Tal på testdatapunkt
dorth = 2**5 #Trunkeringskoeffisient
dnn = 2**9 #Utval ENMF
c = 1 #Klasse for test
r = 0 #Nummer for test
maxiter = 75
delta = 10e-2
indecies = [0,1,2,3,4,5,6,7,8,9]

# A, W, H = getclasses(k,d)
# U, S, Vt = np.linalg.svd(A[c], full_matrices=False)
# plt.semilogy(S)
# plt.show()

B = generatetest(t, indecies)
P, D, C = classify(B, k, dorth, t, indecies, "orth")
# b = np.nonzero(C[c] != c)[0][0]
# print(b)
display(B, P, D, C, indecies, c, 0)
showaccuracy(C,t,indecies)
getaccuracies(k,n,t,indecies,"orth",maxiter,delta)

# P, D, C = classify(B, k, dnn, t, "nn", maxiter, delta)
# display(B, P, D, C, c, r)
# showaccuracy(C,t,indecies)
# getaccuracies(k,n,t,"nn",maxiter,delta)


###################################


# acc = np.array([36.2, 47.7, 60.5, 68.4, 76.2, 77.1, 74.2, 69.8, 67.6, 39.0]) / 100




##################################