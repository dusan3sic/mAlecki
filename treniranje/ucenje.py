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
X = [dataset.normalize(x) for x in X]
tX = [dataset.normalize(x) for x in tX]

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
lr = int(1e+0)

L = []

for i in range(1, epoch + 1):
    if i % (epoch / 10) == 0:
        print("iteracija ", i)

    #test   
    # X, y = utils.Randomize(X, y)

    # print("\nIDEGAS\n", X[:10])    

    z1 = X @ w1 + b1
    a1 = utils.ReLU(z1)

    z2 = a1 @ w2 + b2
    yH = utils.Softmax(z2)

    L.append(utils.CELoss(y, yH))

    #learn
    dz2 = (yH - y) / y.shape[0]
    dw2 = a1.transpose() @ dz2
    db2 = dz2.sum(axis=0)

    da1 = dz2 @ w2.transpose()
    dz1 = utils.dReLU(z1) * da1
    dw1 = X.transpose() @ dz1
    db1 = dz1.sum(axis=0)


    # print("W: \n", w2[0])
    # print("\nlr * dw2: \n", lr * dw2)
    # print("\ndw2: \n", dw2)
    # print("\nNESTO\n")
    w2 -= lr * dw2
    b2 -= lr * db2

    w1 -= lr * dw1
    b1 -= lr * db1

print("\nL: \n", L)

plt.plot(L)
plt.show()