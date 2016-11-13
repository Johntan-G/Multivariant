import statsmodels.api as sm
import numpy as np

data = []
with open('/home/johntan/Desktop/3.5.txt', 'r') as f:
    for line in f.readlines():
        line = line.strip().split(' ')
        line = filter(lambda x: x != '', line)
        line = map(float, line)
        data.append(line)
    X = data[0] + data[2]
    Y = data[1] + data[3]
    X = np.array(X)
    Y = np.array(Y)

X = sm.add_constant(X)

results = sm.OLS(Y, X).fit()

print results.summary()
