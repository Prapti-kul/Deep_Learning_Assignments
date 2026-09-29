# Inputs
x = 2
weight = 0.5
bias = 0.1
target = 1
learning_rate = 0.1

# ----------------------------------------
# Task 2 : Prediction
# ----------------------------------------

prediction = (x*weight)+bias

print("Prediction : ",prediction)

# ----------------------------------------
# Task 3 : Error
# ----------------------------------------

error = target - prediction
print("Error : ",error)

# ----------------------------------------
# Task 4 : Update Weight
# ----------------------------------------
new_weight = weight + (learning_rate*error*x)

# ----------------------------------------
# Task 5 : Display Weights
# ----------------------------------------

print("Old Weight :", weight)
print("Updated Weight :", new_weight)