import numpy as np
import dataset
import koeficijenti
import utils
import matplotlib.pyplot as plt

np.set_printoptions(threshold=1e20)

#load
X, y, tX, ty = dataset.load_data("./data.csv")

#process
X, tX = dataset.load_processed("X.csv", "tX.csv")

#normalize
X = dataset.normalize(X)
tX = dataset.normalize(tX)

X = np.array(X)
tX = np.array(tX)

# X = X[:1000]
# y = y[:1000]


#init learn
k = 1e-1
w1 = np.random.uniform(-10, 10, (13, 5)) * k
b1 = np.random.uniform(-10, 10, (1, 5)) * k

w2 = np.random.uniform(-10, 10, (5, 201)) * k
b2 = np.random.uniform(-10, 10, (1, 201)) * k

epoch = int(1e2)
lr = int(1e+1)
mb = 10
r = len(X) // mb

lastD = [0, 0, 0, 0]
alfa = 0.9

L = []

for i in range(1, epoch + 1):
    if i % (epoch / 10) == 0:
        print("iteracija ", i)

    X, y = utils.Randomize(X, y)

    for i in range(mb):
        x = X[i * r:(i + 1) * r] 
        yTemp = y[i * r:(i + 1) * r]

        z1 = x @ w1 + b1
        a1 = utils.ReLU(z1)

        z2 = a1 @ w2 + b2
        yH = utils.Softmax(z2)

        L.append(utils.CELoss(yTemp, yH))

        #learn
        dz2 = (yH - yTemp) / yTemp.shape[0]
        dw2 = a1.transpose() @ dz2
        db2 = dz2.sum(axis=0)

        da1 = dz2 @ w2.transpose()
        dz1 = utils.dReLU(z1) * da1
        dw1 = x.transpose() @ dz1
        db1 = dz1.sum(axis=0)


        w2 -= lr * dw2 - alfa * lastD[0]
        lastD[0] = lr * dw2

        b2 -= lr * db2 - alfa * lastD[1]
        lastD[1] = lr * db2

        w1 -= lr * dw1 - alfa * lastD[2]
        lastD[2] = lr * dw1

        b1 -= lr * db1 - alfa * lastD[3]
        lastD[3] = lr * db1


# print("\nL: \n", L)

tz1 = tX @ w1 + b1
ta1 = utils.ReLU(tz1)

tz2 = ta1 @ w2 + b2
tyh = utils.Softmax(tz2)

print( np.sum( np.argmax( ty, axis=1 )==np.argmax( tyh, axis=1 ) )  / len(ty) * 100)

plt.plot(L)
plt.show()