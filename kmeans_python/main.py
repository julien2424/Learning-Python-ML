from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
from sklearn import metrics
import pandas as pd

bc = load_breast_cancer()

X = scale(bc.data)

y = bc.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = KMeans(n_clusters=2, random_state=0)

model.fit(X_train)

predictions = model.predict(X_test)

labels = model.labels_

print("labels: ", labels)
print("Predictions: ", predictions)
print("accuracy: ", accuracy_score(y_test, predictions))
print("Actual: ", y_test)

# Commented out IPython magic to ensure Python compatibility.
def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print((name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))

bench_k_means(model, "1", X)

print(pd.crosstab(y_train, labels))
