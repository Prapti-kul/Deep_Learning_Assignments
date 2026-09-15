# Assignment 62 - Employee Attrition Prediction using MLPClassifier

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# -------------------------------------------------
# Step 1 : Load Data
# -------------------------------------------------

def LoadData():

    df = pd.read_csv("Employee_Attrition.csv")

    print("Dataset Loaded Successfully")
    print("----------------------------------------")
    print("Shape :", df.shape)
    print("Columns :", df.columns.tolist())
    print("----------------------------------------")
    print(df.head())

    return df


# -------------------------------------------------
# Step 2 : Prepare Data
# -------------------------------------------------

def PrepareData(df):

    print("----------------------------------------")
    print("Missing Values")
    print(df.isnull().sum())

    print("----------------------------------------")
    print("Numerical Columns")
    print(df.select_dtypes(include=["int64", "float64"]).columns.tolist())

    print("----------------------------------------")
    print("Categorical Columns")
    print(df.select_dtypes(include=["object"]).columns.tolist())

    # Remove extra spaces if any
    df["OverTime"] = df["OverTime"].str.strip()
    df["Attrition"] = df["Attrition"].str.strip()

    # Convert Yes / No into 1 / 0
    df["OverTime"] = df["OverTime"].map({"Yes": 1, "No": 0})
    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

    # Convert into integer
    df["OverTime"] = df["OverTime"].astype(int)
    df["Attrition"] = df["Attrition"].astype(int)

    X = df.drop("Attrition", axis=1)
    Y = df["Attrition"]

    print("----------------------------------------")
    print("Data Preparation Completed")

    return X, Y


# -------------------------------------------------
# Step 3 : Split Data
# -------------------------------------------------

def SplitData(X, Y):

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42
    )

    print("----------------------------------------")
    print("Dataset Split Successfully")

    return X_train, X_test, Y_train, Y_test


# -------------------------------------------------
# Step 4 : Feature Scaling
# -------------------------------------------------

def FeatureScaling(X_train, X_test):

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("----------------------------------------")
    print("Feature Scaling Completed")

    return X_train, X_test, scaler


# -------------------------------------------------
# Step 5 : Train Model
# -------------------------------------------------

def TrainModel(X_train, Y_train):

    model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        max_iter=500,
        random_state=42
    )

    model.fit(X_train, Y_train)

    print("----------------------------------------")
    print("Model Trained Successfully")
    print("Iterations :", model.n_iter_)

    return model


# -------------------------------------------------
# Step 6 : Training Accuracy
# -------------------------------------------------

def TrainingAccuracy(model, X_train, Y_train):

    Y_pred = model.predict(X_train)

    train_accuracy = accuracy_score(Y_train, Y_pred)

    print("----------------------------------------")
    print("Training Accuracy :", train_accuracy * 100, "%")

    return train_accuracy


# -------------------------------------------------
# Step 7 : Testing Accuracy
# -------------------------------------------------

def TestingAccuracy(model, X_test, Y_test):

    Y_pred = model.predict(X_test)

    test_accuracy = accuracy_score(Y_test, Y_pred)

    print("----------------------------------------")
    print("Testing Accuracy :", test_accuracy * 100, "%")

    return test_accuracy


# -------------------------------------------------
# Step 8 : Confusion Matrix
# -------------------------------------------------

def ShowConfusionMatrix(model, X_test, Y_test):

    Y_pred = model.predict(X_test)

    cm = confusion_matrix(Y_test, Y_pred)

    print("----------------------------------------")
    print("Confusion Matrix")
    print(cm)


# -------------------------------------------------
# Step 9 : Loss Curve
# -------------------------------------------------

def PlotLossCurve(model):

    plt.figure(figsize=(6,4))
    plt.plot(model.loss_curve_)

    plt.title("Loss Curve")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.show()


# -------------------------------------------------
# Step 10 : Predict Employee Attrition
# -------------------------------------------------

def PredictAttrition(model, scaler):

    print("----------------------------------------")
    print("Enter Employee Details")

    age = int(input("Age : "))
    income = int(input("Monthly Income : "))
    years_company = int(input("Years At Company : "))
    total_years = int(input("Total Working Years : "))
    distance = int(input("Distance From Home : "))
    job = int(input("Job Satisfaction (1-4) : "))
    worklife = int(input("Work Life Balance (1-4) : "))
    overtime = int(input("OverTime (1 = Yes, 0 = No) : "))
    companies = int(input("Number of Companies Worked : "))
    training = int(input("Training Times Last Year : "))

    employee = [[
        age,
        income,
        years_company,
        total_years,
        distance,
        job,
        worklife,
        overtime,
        companies,
        training
    ]]

    employee = scaler.transform(employee)

    prediction = model.predict(employee)

    if prediction[0] == 0:
        print("Prediction : Employee is likely to Stay")
    else:
        print("Prediction : Employee is likely to Leave")


# -------------------------------------------------
# Main Function
# -------------------------------------------------

def main():

    print("========================================")
    print("Employee Attrition Prediction System")
    print("========================================")

    df = LoadData()

    X, Y = PrepareData(df)

    X_train, X_test, Y_train, Y_test = SplitData(X, Y)

    X_train, X_test, scaler = FeatureScaling(X_train, X_test)

    model = TrainModel(X_train, Y_train)

    train_accuracy = TrainingAccuracy(model, X_train, Y_train)

    test_accuracy = TestingAccuracy(model, X_test, Y_test)

    ShowConfusionMatrix(model, X_test, Y_test)

    PlotLossCurve(model)

    print("----------------------------------------")
    print("Predict 5 Employees")
    print("----------------------------------------")

    for i in range(5):
        print("\nEmployee", i + 1)
        PredictAttrition(model, scaler)

    print("----------------------------------------")
    print("Overfitting / Underfitting Check")
    print("----------------------------------------")

    difference = train_accuracy - test_accuracy

    if difference > 0.05:
        print("Model is Overfitting")
    elif difference < -0.05:
        print("Model is Underfitting")
    else:
        print("Model is Well Fitted")


if __name__ == "__main__":
    main()