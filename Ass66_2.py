#Q2: Demonstrate Sigmoid, ReLU and Tanh Activation Functions
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10,10,100) #linspace means Linear Space.
#It creates equally spaced numbers between a starting value 
# and an ending value.
print(x)

sigmoid = 1/(1+np.exp(-x))

relu = np.maximum(0,x)

tanh = np.tanh(x)

plt.plot(x,sigmoid,label= "sigmoid")
plt.plot(x, relu, label="ReLU")
plt.plot(x, tanh, label="Tanh")

plt.title("Activation Functions")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)
plt.legend()

plt.show()