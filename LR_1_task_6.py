import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from utilities import visualize_classifier

# Завантаження даних
input_file = 'data_multivar_nb.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Розбивка на навчальний (80 %) і тестовий (20 %) набори
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=3)

# Класифікатор SVM з радіальним ядром
svm_classifier = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_classifier.fit(X_train, y_train)
y_test_pred = svm_classifier.predict(X_test)

accuracy = 100.0 * (y_test == y_test_pred).sum() / X_test.shape[0]
print("Accuracy of SVM classifier on test data =", round(accuracy, 2), "%")
visualize_classifier(svm_classifier, X_test, y_test)

# Порівняння SVM і наївного байєсівського класифікатора
num_folds = 3
models = {'SVM': SVC(kernel='rbf', C=1.0, gamma='scale'),
          'Naive Bayes': GaussianNB()}
scorings = {'Accuracy': 'accuracy',
            'Precision': 'precision_weighted',
            'Recall': 'recall_weighted',
            'F1': 'f1_weighted'}

for model_name, model in models.items():
    print("\n" + model_name + ":")
    for metric_name, scoring in scorings.items():
        values = cross_val_score(model, X, y, scoring=scoring, cv=num_folds)
        print(metric_name + ": " + str(round(100 * values.mean(), 2)) + "%")