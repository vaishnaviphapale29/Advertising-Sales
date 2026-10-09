import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from pathlib import Path
import seaborn as sns
 
from sklearn.model_selection import train_test_split 
from sklearn.pipeline import Pipeline 
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score 
 
def SalesAdvertise(DataPath): 
 
    Border = "-"*40 

    # ---------------------------------------------------
    # Project paths
    # ---------------------------------------------------

    BASE_DIR = Path(__file__).resolve().parent
    DATASET_PATH = BASE_DIR / "Advertising.csv"
    IMAGE_DIR = BASE_DIR
 
    #--------------------------------------------------- 
    # Step 1 : Load dataset 
    #--------------------------------------------------- 
 
    print(Border) 
    print("Step 1 : Load dataset") 
    print(Border) 

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Advertising.csv not found at: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)
 
    print("Few records from the dataset : ") 
    print(df.head()) 
 
    #--------------------------------------------------- 
    # Step 2 : Remove unwanted columns 
    #--------------------------------------------------- 
 
    print(Border) 
    print("Step 2 : Remove unwanted columns") 
    print(Border) 
 
    print("Shape of dataset before removal : ",df.shape) 
 
    if 'Unnamed: 0' in df.columns: 
        df.drop(
            columns=['Unnamed: 0'], 
            inplace=True
        ) 
 
    print("Shape of dataset after removal : ",df.shape) 
 
    print("Clean dataset is : ") 
    print(df.head()) 
 
    #--------------------------------------------------- 
    # Step 3 : Check missing values 
    #--------------------------------------------------- 
 
    print(Border) 
    print("Step 3 : Check missing values") 
    print(Border) 
 
    print(
        "Missing values count : \n", 
        df.isnull().sum()
    ) 
 
    #--------------------------------------------------- 
    # Step 4 : Display statistical summary 
    #--------------------------------------------------- 
 
    print(Border) 
    print("Step 4 : Display statistical summary") 
    print(Border) 
 
    print(df.describe()) 
 
    #--------------------------------------------------- 
    # Step 5 : Correlation between columns 
    #--------------------------------------------------- 
 
    print(Border) 
    print("Step 5 : Correlation between columns") 
    print(Border) 
 
    print("Correlation matrix : ") 
    print(df.corr()) 

    # Visualization 1: Correlation heatmap

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        df.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Advertising Correlation Heatmap")
    plt.tight_layout()

    plt.savefig(
        IMAGE_DIR / "correlation_heatmap.png",
        dpi=300
    )

    plt.show()

    plt.close()

    # Visualization 2: Advertising channels vs sales

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, column in zip(
        axes,
        ['TV', 'radio', 'newspaper']
    ):

        ax.scatter(df[column], df['sales'], alpha=0.7)

        ax.set_xlabel(column)
        ax.set_ylabel("Sales")
        ax.set_title(f"{column} vs Sales")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        IMAGE_DIR / "advertising_channels_vs_sales.png",
        dpi=300
    )

    plt.show()

    plt.close()
 
    #--------------------------------------------------- 
    # Step 6 : Split dataset into independent & 
    #          dependent variables 
    #--------------------------------------------------- 
 
    print(Border) 
    print(
        "Step 6 : Split dataset into independent "
        "& dependent variables"
    ) 
    print(Border) 
 
    X = df[
        [
            'TV',
            'radio',
            'newspaper'
        ]
    ] 
 
    Y = df['sales'] 
 
    print(
        "Shape of independent variables : ",
        X.shape
    ) 
 
    print(
        "Shape of dependent variables : ",
        Y.shape
    ) 
 
    #--------------------------------------------------- 
    # Step 7 : Split dataset for training & testing 
    #--------------------------------------------------- 
 
    print(Border) 
    print("Step 7 : Split dataset for training & testing") 
    print(Border) 
 
    X_train, X_test, Y_train, Y_test = train_test_split( 
        X, 
        Y, 
        test_size=0.2, 
        random_state=42
    ) 
 
    print("X_train shape : ",X_train.shape) 
    print("X_test shape : ",X_test.shape) 
    print("Y_train shape : ",Y_train.shape) 
    print("Y_test shape : ",Y_test.shape) 
 
    #--------------------------------------------------- 
    # Step 8 : Create ML Pipeline 
    #--------------------------------------------------- 
 
    print(Border) 
    print("Step 8 : Create ML Pipeline") 
    print(Border) 
 
    ModelPipeline = Pipeline([ 
        (
            "imputer", 
            SimpleImputer(strategy="median")
        ), 
        (
            "scaler", 
            StandardScaler()
        ), 
        (
            "model", 
            LinearRegression()
        ) 
    ]) 
 
    print("ML Pipeline created successfully") 
 
    #--------------------------------------------------- 
    # Step 9 : Train the model 
    #--------------------------------------------------- 
 
    print(Border) 
    print("Step 9 : Train the model") 
    print(Border) 
 
    ModelPipeline.fit(
        X_train, 
        Y_train
    ) 
 
    print("Model trained successfully") 
 
    #--------------------------------------------------- 
    # Step 10 : Test the model 
    #--------------------------------------------------- 
 
    print(Border) 
    print("Step 10 : Test the model") 
    print(Border) 
 
    Y_pred = ModelPipeline.predict(X_test) 
 
    print("Prediction completed successfully") 
 
    #--------------------------------------------------- 
    # Step 11 : Evaluate the model 
    #--------------------------------------------------- 
 
    print(Border) 
    print("Step 11 : Evaluate the model") 
    print(Border) 
 
    MSE = mean_squared_error(Y_test, Y_pred) 
    RMSE = np.sqrt(MSE) 
    R2 = r2_score(Y_test, Y_pred) 
 
    print("Mean Squared Error : ",MSE) 
    print("Root Mean Squared Error : ",RMSE) 
    print("R Square value : ",R2) 
 
    #--------------------------------------------------- 
    # Step 12 : Calculate model coefficient 
    #--------------------------------------------------- 
 
    print(Border) 
    print("Step 12 : Calculate model coefficient") 
    print(Border) 
 
    Model = ModelPipeline.named_steps["model"] 
 
    for column, value in zip(X.columns, Model.coef_): 
        print(f"{column} : {value}") 
 
    print("Intercept : ", Model.intercept_) 
 
    #--------------------------------------------------- 
    # Step 13 : Compare actual and predicted values 
    #--------------------------------------------------- 
 
    print(Border) 
    print("Step 13 : Compare actual and predicted values") 
    print(Border) 
 
    Result = pd.DataFrame({ 
        'Actual sale': Y_test.values, 
        'Predicted sale': Y_pred 
    }) 
 
    print(Result.head()) 
 
    #--------------------------------------------------- 
    # Step 14 : Plot actual vs predicted 
    #--------------------------------------------------- 
 
    print(Border) 
    print("Step 14 : Plot actual vs predicted") 
    print(Border) 
 
    plt.figure(figsize=(8,5)) 
 
    plt.scatter(Y_test, Y_pred) 

    # Ideal prediction reference line
    plt.plot(
        [Y_test.min(), Y_test.max()],
        [Y_test.min(), Y_test.max()],
        linestyle="--"
    )
 
    plt.xlabel("Actual sales") 
    plt.ylabel("Predicted sales") 
    plt.title("Actual sales vs predicted sales") 
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(
        IMAGE_DIR / "actual_vs_predicted.png",
        dpi=300
    )

    plt.show()

    plt.close()

    #---------------------------------------------------
    # Step 15 : Residual plot
    #---------------------------------------------------

    print(Border)
    print("Step 15 : Residual plot")
    print(Border)

    Residuals = Y_test.values - Y_pred

    plt.figure(figsize=(8, 5))

    plt.scatter(Y_pred, Residuals)
    plt.axhline(y=0, linestyle="--")

    plt.xlabel("Predicted sales")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        IMAGE_DIR / "residual_plot.png",
        dpi=300
    )

    plt.show()

    plt.close()

    print("All visualizations saved in:", IMAGE_DIR)

def main(): 
    SalesAdvertise("Advertising.csv") 

if __name__ == "__main__": 
    main()