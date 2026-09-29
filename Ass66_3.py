import math

actual = [1,0,1,1]
predicted = [0.9,0.2,0.8,0.7]

# ----------------------------------------
# Task 1 : Mean Squared Error
# ----------------------------------------

MSE = 0

for i in range(len(actual)):
    MSE =MSE + (actual[i] - predicted[i])**2

MSE =MSE /len(actual)

print("Mean squared Error(MSE):",MSE)

# ----------------------------------------
# Task 2 : Binary Cross Entropy
# ----------------------------------------

bce = 0

for i in range(len(actual)):
    bce = bce + (actual[i]*math.log(predicted[i])+(1 - actual[i])*math.log(1-predicted[i]))

bce = -bce/len(actual)

print("Binary Cross Entropy (BCE) :", bce)
