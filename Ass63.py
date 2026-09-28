import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

import matplotlib.pyplot as plt

# -------------------------------------------------
# Step 1 : Load Data
# -------------------------------------------------
def loaddata():
    df = pd.read_csv("Loan_Default.csv")
    print(df.head())

    return df

# -------------------------------------------------
# Step 2 : Exploratory Analysis
# -------------------------------------------------
def ExploratoryAnalysis(df):

    print("----------------------------------------")
    print("Dataset Information")
    print(df.info())

    print("----------------------------------------")
    print("description")
    print(df.describe())

# -------------------------------------------------
# Task 3 : Find Missing Values
# -------------------------------------------------

def MissingValues(df):

    print("----------------------------------------")
    print("Task 3 : Missing Values")
    print("----------------------------------------")

    print(df.isnull().sum())


# -------------------------------------------------
# Task 4 : Check Whether Target Classes are Balanced
# -------------------------------------------------
def CheckTargetBalance(df):
    print("----------------------------------------")
    print("Task 4 : Target Class Distribution")
    print("----------------------------------------")

    print(df["Default"].value_counts())

# -------------------------------------------------
# Task 5 : Encode Categorical Variables
# -------------------------------------------------
def EncodeData(df):
    print("----------------------------------------")
    print("Task 5 : Encode Categorical Variables")
    print("----------------------------------------")

    encoder = LabelEncoder()

    df["PreviousDefault"] = encoder.fit_transform(df["PreviousDefault"])
    df["HomeOwnership"] = encoder.fit_transform(df["HomeOwnership"])

    print("Categorical columns encoded successfully")
    print(df.head())

    return df

# -------------------------------------------------
# Task 6 : Separate X and Y
# -------------------------------------------------
   
def PrepareData(df):
   
    print("----------------------------------------")
    print("Task 6 : Separate X and Y")
    print("----------------------------------------")

    X = df.drop("Default",axis=1)
    Y = df["Default"]

    return X,Y

# -------------------------------------------------
# Task 7 & Task 8 : Split Dataset (Stratified)
# -------------------------------------------------

def SplitData(X, Y):

    print("----------------------------------------")
    print("Task 7 : Train Test Split")
    print("----------------------------------------")

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42,stratify=Y)

    return X_train,X_test,Y_train,Y_test

# -------------------------------------------------
# Task 9 : Feature Scaling
# -------------------------------------------------

def ScaleFeatures(X_train, X_test):

    print("----------------------------------------")
    print("Task 9 : Feature Scaling")
    print("----------------------------------------")

    scaler = StandardScaler()

    # Fit scaler only on training data
    X_train_scaled = scaler.fit_transform(X_train)

    # Use same scaler on test data
    X_test_scaled = scaler.transform(X_test)

    print("Feature Scaling Completed")

    return X_train_scaled, X_test_scaled, scaler

# -------------------------------------------------
# Task 10 : Create MLPClassifier
# -------------------------------------------------
def CreateModel():

    print("----------------------------------------")
    print("Task 10 : Create MLPClassifier")
    print("----------------------------------------")

    model = MLPClassifier(hidden_layer_sizes=(32,16),
                          activation="relu",
                          solver="adam",
                          max_iter=1000,
                          random_state=42)

    print("MLPClassifier Created Successfully")

    return model

# -------------------------------------------------
# Task 11 : Train the Model
# -------------------------------------------------

def Trainmodel(model,X_train,Y_train):

    print("----------------------------------------")
    print("Task 11 : Train the Model")
    print("----------------------------------------")

    model.fit(X_train,Y_train)
    
    print("Model Trained Successfully")
    print("Number of Iterations :", model.n_iter_)

    return model

# -------------------------------------------------
# Task 12 : Calculate Accuracy
# -------------------------------------------------

def checkaccuracy(model,X_train,X_test,Y_train,Y_test):

    print("----------------------------------------")
    print("Task 12 : Calculate Accuracy")
    print("----------------------------------------")

    train_prediction = model.predict(X_train)
    test_prediction = model.predict(X_test)

    train_acc = accuracy_score(Y_train,train_prediction)
    test_acc = accuracy_score(Y_test,test_prediction)

    
    print("Training Accuracy :", train_acc * 100, "%")
    print("Testing Accuracy :", test_acc * 100, "%")

    return train_acc, test_acc

# -------------------------------------------------
# Task 13 : Confusion Matrix
# -------------------------------------------------
def ShowConfusionMatrix(model, X_test, Y_test):

    print("----------------------------------------")
    print("Task 13 : Confusion Matrix")
    print("----------------------------------------")

    prediction = model.predict(X_test)

    cm = confusion_matrix(Y_test, prediction)

    print(cm)

# -------------------------------------------------
# Task 14 : Classification Report
# -------------------------------------------------


def ShowClassificationReport(model, X_test, Y_test):

    print("----------------------------------------")
    print("Task 14 : Classification Report")
    print("----------------------------------------")

    prediction = model.predict(X_test)

    print(classification_report(Y_test, prediction))

# -------------------------------------------------
# Task 15 : Precision, Recall and F1 Score
# -------------------------------------------------

def showmetrics(model,X_test,Y_test):
    
    print("----------------------------------------")
    print("Task 15 : Precision, Recall and F1 Score")
    print("----------------------------------------")

    prediction = model.predict(X_test)

    print("Precision :", precision_score(Y_test, prediction))
    print("Recall    :", recall_score(Y_test, prediction))
    print("F1 Score  :", f1_score(Y_test, prediction))

# -------------------------------------------------
# Task 16 : Plot Training Loss
# -------------------------------------------------
def PlotLossCurve(model):

    print("----------------------------------------")
    print("Task 16 : Plot Training Loss Curve")
    print("----------------------------------------")

    plt.figure(figsize=(6,4))

    plt.plot(model.loss_curve_)
    plt.title("Training Loss curve")
    plt.xlabel("Iterations")
    plt.grid(True)
    plt.show()

# -------------------------------------------------
# Task 17 : Predict Loan Default
# -------------------------------------------------
def PredictLoan(model, scaler):

    print("----------------------------------------")
    print("Task 17 : Predict Loan Default")
    print("----------------------------------------")

    age = int(input("Age : "))
    income = int(input("Income : "))
    loan = int(input("Loan Amount : "))
    credit = int(input("Credit Score : "))
    employment = int(input("Employment Years : "))
    existing = int(input("Existing Loans : "))
    debt = int(input("Monthly Debt : "))
    term = int(input("Loan Term : "))
    previous = int(input("Previous Default (1=Yes, 0=No) : "))
    home = int(input("Home Ownership (0=Own, 1=Rent) : "))

    applicant = [[
        age,
        income,
        loan,
        credit,
        employment,
        existing,
        debt,
        term,
        previous,
        home
    ]]



    applicant = scaler.transform(applicant)

    result = model.predict(applicant)

    if result[0] == 1:
        print("Prediction : High Probability of Loan Default")
    else:
        print("Prediction : Loan is Safe")


# -------------------------------------------------
# Main Function
# -------------------------------------------------

def main():

    # Task 1
    df = loaddata()

    # Task 2
    ExploratoryAnalysis(df)

    # Task 3
    MissingValues(df)

    # Task 4
    CheckTargetBalance(df)

    # Task 5
    df = EncodeData(df)

    # Task 6
    X, Y = PrepareData(df)

    # Task 7 & 8
    X_train, X_test, Y_train, Y_test = SplitData(X, Y)

    # Task 9
    X_train, X_test, scaler = ScaleFeatures(X_train, X_test)

    # Task 10
    model = CreateModel()

    # Task 11
    model = Trainmodel(model, X_train, Y_train)

    # Task 12
    train_acc, test_acc = checkaccuracy(
        model, X_train, X_test, Y_train, Y_test
    )

    # Task 13
    ShowConfusionMatrix(model, X_test, Y_test)

    # Task 14
    ShowClassificationReport(model, X_test, Y_test)

    # Task 15
    showmetrics(model, X_test, Y_test)

    # Task 16
    PlotLossCurve(model)

    # Task 17
    print("----------------------------------------")
    print("Predict Loan for 5 Applicants")
    print("----------------------------------------")

    for i in range(5):
        print("\nApplicant", i + 1)
        PredictLoan(model, scaler)

    # Overfitting / Underfitting Check
    print("----------------------------------------")
    print("Overfitting / Underfitting Check")
    print("----------------------------------------")

    difference = train_acc - test_acc

    if difference > 0.05:
        print("Model is Overfitting")
    elif difference < -0.05:
        print("Model is Underfitting")
    else:
        print("Model is Well Fitted")


# -------------------------------------------------
# Start Program
# -------------------------------------------------

if __name__ == "__main__":
    main()