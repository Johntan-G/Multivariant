import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import pylab

data = []
with open('/home/johntan/Desktop/3.15.txt','r') as f:
    for line in f.readlines():
        line = line.strip().split(' ')
        line = filter(lambda x: x != '', line)
        line = map(float, line)
        data.append(line)
    X = data[0]
    Y = data[1]
    X = np.array(X)
    Y = np.array(Y)
X = sm.add_constant(X)
model = sm.OLS(Y, X)
results = model.fit()
Y_hat = np.dot(X, results.params)
errors = Y - Y_hat


plt.plot(Y_hat, errors, '*')
ax = pylab.gca()

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_position(('data', 0))

plt.show()


########################################################
Y = np.sqrt(Y)

model = sm.OLS(Y, X)
results = model.fit()
Y_hat = np.dot(X, results.params)
errors = Y - Y_hat


plt.plot(Y_hat, errors, '*')
ax = pylab.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_position(('data', 0))

plt.show()

########################################################
RSS = []
for lamb in np.linspace(-2, 2, 8):
    if lamb != 0:
        Y = np.power(Y, lamb)

        model = sm.OLS(Y, X)
        results = model.fit()
        Y_hat = np.dot(X, results.params)
        errors = Y - Y_hat
        RSS.append(np.dot(errors, errors))
    else:
        Y = np.log(Y)

        model = sm.OLS(Y, X)
        results = model.fit()
        Y_hat = np.dot(X, results.params)
        errors = Y - Y_hat
        RSS.append(np.dot(errors, errors))

#plt.plot(np.linspace(-2, 2, 8), RSS)
#plt.show()
print np.column_stack((np.linspace(-2, 2, 8), RSS))

