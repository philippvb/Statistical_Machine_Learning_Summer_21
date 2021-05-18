import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import matplotlib.pyplot as plt

def cost_fn(w, x, y, lmbd):
    ''' L1 loss + L2 regularization

    w: weights to estimate d
    x: data points n x d
    y: true values n x 1
    lmbd: weight regularization

    output: loss ||x * w - y||_1 + lmbd * ||w||_2^2
    '''
    return np.abs(x @ np.expand_dims(w, 1) - y).sum() +\
           lmbd * (w ** 2).sum()

def L1LossRegression(X, Y, lmbd_reg=0.):
    ''' solves linear regression with
    L1 Loss + L2 regularization

    X: deisgn matrix n x d
    Y: true values n x 1
    lmbd_reg: weight regularization

    output: weight of linear regression d x 1
    '''
    w = minimize(cost_fn, np.zeros(X.shape[1]),
                 args=(X, Y, lmbd_reg)).x
    return w

# ------------------ (a)
def cost_fn_square(w, x, y, lmbd):
    return np.square(x @ np.expand_dims(w, 1) - y).sum() +\
        lmbd * (w ** 2).sum()

def RidgeRegression(X, Y, lmbd_reg=0.):
    w = minimize(cost_fn_square, np.zeros(X.shape[1]),
                args=(X, Y, lmbd_reg)).x
    return w

def LeastSquares(X, Y):
    return RidgeRegression(X, Y, 0.)


# ------------------ (b)
def Basis(X,k):
    basis_data = np.zeros((X.shape[0], 2*k+1))
    basis_data[:, 0] = np.ones(X.shape[0])
    for freq in range(1, 2*k+1, 2):
        # basis_data[:, freq] = np.squeeze(np.cos(2 * np.pi * freq * X))
        # basis_data[:, freq+1] = np.squeeze(np.sin(2 * np.pi * freq * X))
        basis_data[:, freq] = np.squeeze(np.cos(freq * X))
        basis_data[:, freq+1] = np.squeeze(np.sin(freq * X))
    return basis_data


# ------------------ (c)
data = np.load("exercise3/onedim_data.npy", allow_pickle=True).item()
x_train = data["Xtrain"]
y_train = data["Ytrain"]
x_test = data["Xtest"]
y_test = data["Ytest"]




# because there are many outlies, we choose L1, since it penalized outliers less
plot_x = np.linspace(0, 1, 1000)
plt.scatter(np.squeeze(x_train), np.squeeze(y_train), color="grey")

k_list = [1,2,3,5,10,15,20]
train_losses = []
test_losses = []
lambda_regularization = 30
for index, k in enumerate(k_list):
    x_train_mapped = Basis(x_train, k)
    w = L1LossRegression(x_train_mapped, y_train, lambda_regularization)
    train_losses.append(cost_fn(w, x_train_mapped, y_train, lambda_regularization)/x_train.shape[0])
    test_losses.append(cost_fn(w, Basis(x_test, k), y_test, lambda_regularization)/(x_test.shape[0]))
    plt.plot(plot_x, np.squeeze(Basis(plot_x, k) @ np.expand_dims(w, 1)))

# plt.show()
plt.plot(k_list, train_losses)
plt.plot(k_list, test_losses)
# plt.show()

# -------------------- (d)

def FourierBasisNormalized(X, k):
    basis_data = np.zeros((X.shape[0], 2*k+1))
    basis_data[:, 0] = np.ones(X.shape[0])
    for freq in range(1, 2*k+1, 2):
        basis_data[:, freq] = np.squeeze(np.cos(freq * X))/omega("cos", freq)
        basis_data[:, freq+1] = np.squeeze(np.sin(freq * X))/omega("sin", freq)
    return basis_data




def omega(f, freq):
    if f == "sin":
        outer_derivative = np.cos
    else:
        outer_derivative = np.sin
    return quad(lambda x: np.square(outer_derivative(freq * x) * freq), 0, 1)[0]



plt.clf()

plot_x = np.linspace(0, 1, 1000)
plt.scatter(np.squeeze(x_train), np.squeeze(y_train), color="grey")

k_list = [1,2,3,5,10,15,20]
# figs, axs = plt.subplots(len(k_list))
train_losses = []
test_losses = []
lambda_regularization = 0
for index, k in enumerate(k_list):
    x_train_mapped = FourierBasisNormalized(x_train, k)
    w = L1LossRegression(x_train_mapped, y_train, lambda_regularization)
    train_losses.append(cost_fn(w, x_train_mapped, y_train, lambda_regularization)/x_train.shape[0])
    test_losses.append(cost_fn(w, FourierBasisNormalized(x_test, k), y_test, lambda_regularization)/(x_test.shape[0]))
    plt.plot(plot_x, np.squeeze(FourierBasisNormalized(plot_x, k) @ np.expand_dims(w, 1)))

plt.show()
plt.plot(k_list, train_losses)
plt.plot(k_list, test_losses)
plt.show()


