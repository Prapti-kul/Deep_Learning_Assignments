import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# -------------------------------------------------
# Task 1 : Load or Create Dataset
# -------------------------------------------------

def createdataset():

    print("----------------------------------------")
    print("Task 1 : Create Dataset")
    print("----------------------------------------")

    X = np.array([
        [25, 500, 12, 1, 2],
        [30, 700, 24, 0, 1],
        [45, 1200, 6, 5, 8],
        [50, 1500, 5, 6, 10],
        [28, 600, 18, 1, 1],
        [35, 800, 30, 0, 0],
        [48, 1400, 4, 7, 9],
        [52, 1600, 3, 8, 12],
        [27, 550, 20, 0, 1],
        [42, 1300, 8, 4, 7]
    ])

    Y = np.array([
        0, 0, 1, 1, 0,
        0, 1, 1, 0, 1
    ])

    print("Dataset created successfully")
    print("Input Shape :", X.shape)
    print("Output Shape :", Y.shape)

    return X, Y

# -------------------------------------------------
# Task 2 : Clean the Dataset
# -------------------------------------------------

def CleanData(X, Y):

    print("----------------------------------------")
    print("Task 2 : Clean Dataset")
    print("----------------------------------------")

    print("Missing values in X :",np.isnan(X).sum())
    print("Missing values in Y :",np.isnan(Y).sum())

    print("Dataset is clean")

    return X,Y

# -------------------------------------------------
# Task 3 : Apply StandardScaler
# -------------------------------------------------

def ScaleData(X):

    print("----------------------------------------")
    print("Task 3 : Apply StandardScaler")
    print("----------------------------------------")

    scaler = StandardScaler()

    x_scaled = scaler.fit_transform(X)

    print("scaling completed")
    return x_scaled,scaler

# -------------------------------------------------
# Task 4 : Train FNN Model
# -------------------------------------------------


def TrainModel(X, Y):

    print("----------------------------------------")
    print("Task 4 : Train FNN Model")
    print("----------------------------------------")

    model = MLPClassifier(hidden_layer_sizes=(10,5),activation="relu",
                          solver="adam",max_iter=1000,random_state=42)

    model.fit(X,Y)

    print("FNN Model trained successfully")
    print("Number of iterations :", model.n_iter_)

    return model

# -------------------------------------------------
# Task 5 : Evaluate Accuracy
# -------------------------------------------------
def EvaluateModel(model, X, Y):

    print("----------------------------------------")
    print("Task 5 : Evaluate Accuracy")
    print("----------------------------------------")

    prediction = model.predict(X)

    accuracy = accuracy_score(Y,prediction)

    print("Actual values:",Y)
    print("Predicted values :",prediction)

    print("Accuracy :",accuracy*100,"%")


# -------------------------------------------------
# Test New Customer
# -------------------------------------------------

def PredictCustomer(model, scaler):

    print("----------------------------------------")
    print("Test New Customer")
    print("----------------------------------------")

    new_customer =[[46,1450,5,6,9]]

    new_customer_scaled = scaler.transform(new_customer)

    result = model.predict(new_customer_scaled)

    if result[0]==1:
        print("prediction : customer may leave")

    else:
        print("prediction : customer will stay")


# -------------------------------------------------
# Main Function
# -------------------------------------------------

def main():

    # Task 1
    X, Y = createdataset()

    # Task 2
    X, Y = CleanData(X, Y)

    # Task 3
    X_scaled, scaler = ScaleData(X)

    # Task 4
    model = TrainModel(X_scaled, Y)

    # Task 5
    EvaluateModel(model, X_scaled, Y)

    # Test new customer
    PredictCustomer(model, scaler)



if __name__ == "__main__":
    main()