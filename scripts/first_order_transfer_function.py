import numpy as np
import matplotlib.pyplot as plt

def order_1(x, u, tau):
    return (u-x)/tau

T = 500
dt = 0.01
x0 = 0.
u = np.zeros((T,))
u[5] = 1.

tau = 0.5

x = np.zeros((T+1,))
x2 = np.zeros((T+1,))

for i in range(T):
    x[i+1] = x[i] + order_1(x[i], u[i], tau)*dt
    x2[i+1] = x2[i] + order_1(x2[i], x[i], tau) * dt

plt.subplot(2,1,1)
plt.plot(np.arange(T+1)*dt,x)

plt.subplot(2,1,2)
plt.plot(np.arange(T+1)*dt,x2)

plt.show()
