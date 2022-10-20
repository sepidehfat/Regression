import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
data = pd.read_csv("test.csv")

#%% cost function MSE
def cost_function(X, y, theta):
    m = len(y)
    h = np.dot(X, theta)
    error = h - y
    cost = (1/(2*m)) * np.sum(np.power(error,2))
    return cost

#%% make needed data
m= len(data['x'])
x= (np.array(data['x'])).reshape(m,1)
y= (np.array(data['y'])).reshape(m,1)
x_2= (np.array(np.power(data['x'],2))).reshape(m,1)
X = np.insert(x, 0, np.ones([m]), axis=1)
X = np.append(X, x_2, axis=1)
#%% Gradient descent
theta= np.array([[0],[0],[0]])
prev_cost= 0
tolerance= 100000
costs= []
while(abs(tolerance)>0.00001):
    h = np.dot(X, theta)
    theta = theta - 0.00000001/m * np.dot(X.T,h-y)
    cost= cost_function(X,y,theta)
    costs.append(cost)
    tolerance= abs(cost-prev_cost)/cost
    prev_cost= cost;
#%% total cost
total_cost= cost_function(X,y,theta)
#%%plot regression
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
pnt3d= ax.scatter(X[:,1],X[:,2],y, c=y)
ax.plot_surface(X[:,1],X[:,2],h, cmap='coolwarm',linewidth=0, alpha=0.01)
cbar=plt.colorbar(pnt3d)
cbar.set_label("y")
plt.show()
#%% print cost
plt.figure(figsize=(10, 10))
plt.title("cost plot")
plt.plot(list(range(len(costs))),costs)
plt.xlabel("iterations")
plt.ylabel("cost")
plt.show()