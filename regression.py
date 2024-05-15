import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression


def main():
    # loadDF = quandl.get("WIKI/AAPL")
    loadDF = pd.read_parquet("aaplStockPrice.parquet")
    df = pd.DataFrame(data=loadDF)
    # print(df.head())
    df = df[["Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume"]]
    df["volHL"] = (df["Adj. High"] - df["Adj. Low"]) / df["Adj. Low"] * 100.0
    df["volOC"] = (df["Adj. Close"] - df["Adj. Open"]) / df["Adj. Open"] * 100.0

    df = df[["Adj. Close", "volHL", "volOC", "Adj. Volume"]]
    # print(df.head())
    forecastCol = "Adj. Close"
    df.fillna(-99999, inplace=True)
    forecastOut = int(math.ceil(0.1 * len(df)))

    df["label"] = df[forecastCol].shift(-forecastOut)
    df.dropna(inplace=True)
    X = np.array(df.drop(["label"], axis=1))
    y = np.array(df["label"])
    X = preprocessing.scale(X)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.2
    )

    # chosing the classifier and
    classifier = LinearRegression()
    classifier.fit(X_train, y_train)
    accuracy = classifier.score(X_test, y_test)
    print(accuracy)


if __name__ == "__main__":
    main()
