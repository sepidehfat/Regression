import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

#%%
import numpy as np
#%% read data
data = pd.read_csv("test.csv")
#%% cost function MSE
def cost_function(X, y, theta):
    m = len(y)
    h = np.dot(X, theta)
    error = h - y
    cost = (1/(2*m)) * np.sum(np.power(error,2))
    return cost
#%% Gradient descent
m= len(data['x'])
x= (np.array(data['x'])).reshape(m,1)
y= (np.array(data['y'])).reshape(m,1)
X = np.insert(x, 0, np.ones([m]), axis=1)
theta= theta= np.zeros((2,1))

prev_cost= 0
tolerance= 100000
costs= []
while(abs(tolerance)>0.0001):
    h = np.dot(X, theta)
    theta = theta - 0.0001/m * np.dot(X.T,h-y)
    cost= cost_function(X,y,theta)
    costs.append(cost)
    tolerance= abs(cost-prev_cost)/cost
    prev_cost= cost;
#%% total cost
total_cost= cost_function(X,y,theta)
#%% plot linear regression on data
plt.figure(figsize=(10, 10))
plt.title("regression")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x, h, 'r')
plt.plot(x, y, 'o')
plt.show()
#%% print cost
plt.figure(figsize=(10, 10))
plt.title("cost plot")
plt.plot(list(range(len(costs))),costs)
plt.xlabel("iterations")
plt.ylabel("cost")
plt.show()

