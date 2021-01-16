from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn import neighbors, metrics, svm
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# knn and svm are both supervised classification models


def get_class_models(file_name):
    # get data in regular csv form
    data = pd.read_csv(file_name)

    # narrow down X to necessary values
    X = data[[
        'buying',
        'maint',
        'safety'
    ]]

    # get target y values
    y = data[['class']]

    #  converting x

    # map x values to integers using a dictionary
    Xlabel_mapping = {
        'low': 0,
        'med': 1,
        'high': 2,
        'vhigh': 3
    }

    # change chosen values to integers
    X['maint'] = X['maint'].map(Xlabel_mapping)
    X['safety'] = X['safety'].map(Xlabel_mapping)
    X['buying'] = X['buying'].map(Xlabel_mapping)

    # get into array of form [[x,y,z],[x,y,z],[x,y,z]...]
    X = np.array(X)

    #  converting y

    # map y targets to integers using dictionary
    ylabel_mapping = {
        'unacc': 0,
        'acc': 1,
        'good': 2,
        'vgood': 3
    }

    # change target values to integers
    y['class'] = y['class'].map(ylabel_mapping)

    # get into array of form [[x],[x],[x]...]
    y = np.array(y)

    # set knn and svm values
    knn_model = neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform')
    svm_model = svm.SVC()

    # segment array to train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # train our knn data set
    knn_model.fit(X_train, y_train)

    # train our svm data set
    svm_model.fit(X_train, y_train)

    # predict our y values for using X from test set
    knn_pred = knn_model.predict(X_test)
    svm_pred = svm_model.predict(X_test)

    # get accuracy of our test
    knn_acc = metrics.accuracy_score(y_test, knn_pred)
    svm_acc = metrics.accuracy_score(y_test, svm_pred)

    # print predictions and accuracy of predictions
    print('knn accuracy: ' + str(knn_acc))
    print('svm accuracy: ' + str(svm_acc))

    return X, y


def get_plot(X, y):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # data for three-dimensional scattered points
    for i in range(len(X)):
        x1 = X[i, 0]
        y1 = X[i, 1]
        z1 = X[i, 2]
        if y[i] == 0:
            ax.scatter(x1, y1, z1, color='blue')
        elif y[i] == 1:
            ax.scatter(x1, y1, z1, color='green')
        elif y[i] == 2:
            ax.scatter(x1, y1, z1, color='red')
        else:
            ax.scatter(x1, y1, z1, color='black')

    # create plot
    plt.show()


X, y = get_class_models('car.data')
get_plot(X, y)
