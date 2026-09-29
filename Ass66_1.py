import math


x1 = 2
x2 = 3

w1 = 0.4
w2 = 0.6

bias = 0.5

# Step 1: Calculate Weighted Sum
weighted_sum = (x1 * w1) + (x2 * w2) + bias

print("Weighted Sum =", weighted_sum)

# Step 2: Apply Sigmoid Function
output = 1 / (1 + math.exp(-weighted_sum))

# Step 3: Display Output
print("Output =", output)

# Step 4: Explain Result
if output > 0.5:
    print("Output is closer to 1")
else:
    print("Output is closer to 0")