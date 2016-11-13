import numpy as np
import matplotlib.pyplot as plt


def read_data(path='/home/johntan/Desktop/3.19.txt'):
    data = []
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split(' ')
            line = filter(lambda x: x != '', line)
            line = map(float, line)
            data.append(line)
        Y = data[0]
        X1 = data[1]
        X2 = data[2]
        X1_bar = sum(X1) / len(X1)
        X2_bar = sum(X2) / len(X2)
        X = map(lambda X: [X[0] - X1_bar, X[1] - X2_bar], zip(X1, X2))
        X = np.array(X)
        Y = np.array(Y)
        return X, Y

X, Y = read_data()

eig_value, eig_vec = np.linalg.eig(np.dot(X.T, X))
lamb = max(eig_value)/min(eig_value)
if lamb < 100:
    print "The multicollinearity is slight"
elif lamb < 1000:
    print "The multicollinearity is moderate"
else:
    print "The multicollinearity is sever"

n_alphas = 200 + 1
alphas = np.linspace(0, 1000, n_alphas)

coefs = []
for a in alphas:
    coefficience = np.dot((np.dot(X.T, X) + np.diag(a * np.ones(2))) ** (-1), np.dot(X.T, Y))
    coefs.append(coefficience)

ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim())  # reverse axis
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()

########## from the picture, We can see when alpha = 2 two lines in the ridge encounter




