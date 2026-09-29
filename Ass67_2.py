import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# -------------------------------------------------
# Task 1 : Create Dataset
# -------------------------------------------------

def CreateData():

    print("----------------------------------------")
    print("Task 1 : Create Dataset")
    print("----------------------------------------")

    X = np.array([
        [25000, 600, 200000, 10000, 0],
        [40000, 700, 300000, 8000, 1],
        [60000, 750, 500000, 12000, 1],
        [20000, 550, 150000, 15000, 0],
        [80000, 800, 700000, 10000, 1],
        [35000, 650, 250000, 9000, 1],
        [18000, 500, 100000, 12000, 0],
        [90000, 850, 800000, 15000, 1],
        [30000, 580, 200000, 14000, 0],
        [70000, 780, 600000, 10000, 1]
    ])

    Y = np.array([
        0, 1, 1, 0, 1,
        1, 0, 1, 0, 1
    ])

    print("Dataset created successfully")
    print("Input Shape :", X.shape)
    print("Output Shape :", Y.shape)

    return X, Y


# -------------------------------------------------
# Task 2 : Preprocess Categorical Values
# -------------------------------------------------

def PrepareData(X):

    print("----------------------------------------")
    print("Task 2 : Preprocess Categorical Values")
    print("----------------------------------------")

    print("Employment Status is already encoded:")
    print("0 = Not Stable")
    print("1 = Stable")

    print("No LabelEncoder is required.")

    return X


# -------------------------------------------------
# Task 3 : Apply Scaling
# -------------------------------------------------

def ScaleData(X):

    print("----------------------------------------")
    print("Task 3 : Apply Scaling")
    print("----------------------------------------")

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    print("StandardScaler applied successfully")

    return X_scaled, scaler


# -------------------------------------------------
# Task 4 : Train FNN Model
# -------------------------------------------------

def TrainModel(X, Y):

    print("----------------------------------------")
    print("Task 4 : Train FNN Model")
    print("----------------------------------------")

    model = MLPClassifier(
        hidden_layer_sizes=(10, 5),
        activation="relu",
        solver="adam",
        max_iter=2000,
        random_state=42
    )

    model.fit(X, Y)

    print("FNN Model trained successfully")
    print("Number of iterations :", model.n_iter_)

    return model


# -------------------------------------------------
# Task 5 : Evaluate Model
# -------------------------------------------------

def EvaluateModel(model, X, Y):

    print("----------------------------------------")
    print("Task 5 : Evaluate Model")
    print("----------------------------------------")

    prediction = model.predict(X)

    accuracy = accuracy_score(Y, prediction)

    print("Actual Values    :", Y)
    print("Predicted Values :", prediction)

    print("Accuracy :", accuracy * 100, "%")


# -------------------------------------------------
# Test New Applicant
# -------------------------------------------------

def PredictApplicant(model, scaler):

    print("----------------------------------------")
    print("Test New Applicant")
    print("----------------------------------------")

    new_applicant = [[55000, 720, 400000, 10000, 1]]

    new_applicant_scaled = scaler.transform(new_applicant)

    result = model.predict(new_applicant_scaled)

    if result[0] == 1:
        print("Prediction : Loan Approved")
    else:
        print("Prediction : Loan Rejected")


# -------------------------------------------------
# Main Function
# -------------------------------------------------

def main():

    # Task 1
    X, Y = CreateData()

    # Task 2
    X = PrepareData(X)

    # Task 3
    X_scaled, scaler = ScaleData(X)

    # Task 4
    model = TrainModel(X_scaled, Y)

    # Task 5
    EvaluateModel(model, X_scaled, Y)

    # Test Applicant
    PredictApplicant(model, scaler)


# -------------------------------------------------
# Starting Point
# -------------------------------------------------

if __name__ == "__main__":
    main()