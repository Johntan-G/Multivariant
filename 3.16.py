import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import pylab


def read_data(path='/home/johntan/Desktop/3.16.txt'):
    data = []
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split(' ')
            line = filter(lambda x: x != '', line)
            line = map(float, line)
            data.append(line)
        X1 = data[0] + data[3]
        X2 = data[1] + data[4]
        Y = data[2] + data[5]
        X = np.array(zip(X1, X2))
        Y = np.array(Y)
        return X, Y


def print_residual_plot(params, Y, X):

    Y_hat = np.dot(X, params)
    errors = Y - Y_hat
    plt.plot(Y_hat, errors, '*')
    plt.xlabel("Y_hat")
    plt.ylabel("e_hat")

    ax = pylab.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_position(('data', 0))
    plt.show()


def Box_Cox(Y, X, lamb_list=np.linspace(-2, 2, 21)):
    RSS = []
    for lamb in lamb_list:
        if lamb != 0:
            Y_lamb = (np.power(Y, lamb) - 1) / lamb
            results_lamb = sm.OLS(Y_lamb, X).fit()
            Y_hat_lamb = results_lamb.predict()

            errors_lamb = Y - Y_hat_lamb
            RSS.append(np.dot(errors_lamb, errors_lamb))
        else:
            Y_lamb = np.log(Y)
            results_lamb = sm.OLS(Y_lamb, X).fit()
            Y_hat_lamb = results_lamb.predict()
            errors_lamb = Y - Y_hat_lamb
            RSS.append(np.dot(errors_lamb, errors_lamb))
    RSS_min_index = RSS.index(min(RSS))
    return RSS, RSS_min_index


X, Y = read_data()
X = sm.add_constant(X)
results = sm.OLS(Y, X).fit()

print results.summary()
print_residual_plot(results.params, Y, X)


lamb_list = np.linspace(-2, 2, 21)
RSS, RSS_min_index = Box_Cox(Y, X, lamb_list)

plt.plot(lamb_list, RSS)
plt.xlabel("lambda")
plt.ylabel("RSS")
plt.show()
lamb_list_min = lamb_list[RSS_min_index]

print "the minimum value of the RSS is : %f where lambda is : %f" % (RSS[RSS_min_index], lamb_list_min)

Y_lamb = (np.power(Y, lamb_list_min) - 1) / lamb_list_min
results_lamb = sm.OLS(Y_lamb, X).fit()

print results_lamb.summary()
print_residual_plot(results_lamb.params, Y_lamb, X)





