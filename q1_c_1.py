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

#%% normalize
normal_data= (data - data.max())/(data.max()-data.min())

#%% make needed data
m= len(normal_data['x'])
x= (np.array(normal_data['x'])).reshape(m,1)
y= (np.array(normal_data['y'])).reshape(m,1)
x_o= (np.array(data['x'])).reshape(m,1)
y_o= (np.array(data['y'])).reshape(m,1)
X_o= np.insert(x_o, 0, np.ones([m]), axis=1)

x_2= (np.array(np.power(data['x'],2))).reshape(m,1)
x_2_normal= (x_2 - x_2.max())/(x_2.max()-x_2.min())

X = np.insert(x, 0, np.ones([m]), axis=1)
X = np.append(X, x_2_normal, axis=1)
X_o = np.append(X_o, x_2, axis=1)

#%% Gradient descent
theta= np.array([[0],[0],[0]])
prev_cost= 0
tolerance= 100000
costs= []
while(abs(tolerance)>0.0001):
    h = np.dot(X, theta)
    # diff=h-y
    theta = theta - 0.1/m * np.dot(X.T,h-y)
    cost= cost_function(X,y,theta)
    # print("cost:", cost)
    costs.append(cost)
    tolerance= abs(cost-prev_cost)/cost
    # print("tolerance", tolerance)
    prev_cost= cost;
#%% total cost
total_cost= cost_function(X_o,y_o,theta)
# h_test = np.dot(X_o, theta)

#%%plot regression with normal data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
pnt3d= ax.scatter(X[:,1],X[:,2],y, c=y)
ax.plot_surface(X[:,1],X[:,2],h, cmap='coolwarm',linewidth=0, alpha=0.01)
# p1, p2 = np.meshgrid(X[:,1],X[:,2])

# pnt3d= ax.contour(p1,p2,h)
cbar=plt.colorbar(pnt3d)
cbar.set_label("y")
plt.show()
#%% print cost
plt.title("cost")
plt.plot(list(range(len(costs))),costs)
plt.show()

#%%