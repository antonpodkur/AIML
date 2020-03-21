import sklearn
from sklearn import datasets
from sklearn import metrics
from sklearn import svm

cancer = datasets.load_breast_cancer()

print(cancer.feature_names)
print()
print(cancer.target_names)

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.2)

print(x_train)
print()
print(y_train)

classes = ['malignant', 'benign']

clf = svm.SVC(kernel = "linear")
clf.fit(x_train,y_train)


y_prediction = clf.predict((x_test))

acc = clf.score(x_test,y_test)
print(acc)
