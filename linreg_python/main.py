from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# linear regression is supervised


def get_linreg_model():
    # retrieve data
    boston = datasets.load_boston()

    # segment data
    X = boston.data
    y = boston.target

    # initialize linear regression
    l_reg = linear_model.LinearRegression()

    # get train test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # create our model
    lin_model = l_reg.fit(X_train, y_train)

    # get our predictions for the data set
    lin_pred = lin_model.predict(X_test)

    # print line values
    print("R^2 value: ", l_reg.score(X,y))
    print("coef: ", l_reg.coef_)
    print("intercept: ", l_reg.intercept_)

    return X, y, l_reg.coef_, l_reg.intercept_


def get_plot(X, y, coef, inter):
    plt.plot([0, inter], [coef[2], coef[2]+inter], color='red')
    plt.scatter(X[:, 2], y, color='blue')
    plt.show()


X, y, coef, inter = get_linreg_model()
get_plot(X, y, coef, inter)

