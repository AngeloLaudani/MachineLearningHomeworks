import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from plotly import tools
import plotly.offline as offline
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import datasets, svm


# plot SVM
def plot_svm(clf, c, x_, y_, xx, yy, X, y, kernel):
    fig = tools.make_subplots(rows=1, cols=1,
                              print_grid=False)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    p1 = go.Contour(x=x_, y=y_, z=Z,
                    colorscale="Picnic",
                    showscale=False)
    fig.append_trace(p1, 1, 1)

    # Plot the training points
    p2 = go.Scatter(x=X[:, 0], y=X[:, 1],
                    mode='markers',
                    marker=dict(color=y,
                                colorscale="Picnic",
                                showscale=False,
                                line=dict(color='black', width=1))
                    )
    fig.append_trace(p2, 1, 1)
    fig['layout'].update(title=kernel + " Kernel SVC with C = " + str(c))
    offline.plot(fig)


# plot accuracy
def plot_accuracy(c_values, score_vector):
    traces = []
    trace1 = go.Scatter(x=list(c_values), y=score_vector)
    traces.append(trace1)
    layout = go.Layout(
        xaxis=dict(
            type='log',
            autorange=True,
            title='C values'),
        yaxis=dict(
            type='log',
            autorange=True,
            title='Accuracy'),
        title='Accuracy on validation set when changing C')
    fig = dict(data=traces, layout=layout)
    time.sleep(0.1)
    offline.plot(fig)


# plot Table
def plot_table(c_values, gamma_values, score_matrix):
    score_matrix = np.around(score_matrix, decimals=2)
    trace = go.Table(
    header=dict(values=["<b>C\Gamma</b>", c_values[0], c_values[1], c_values[2], c_values[3], c_values[4], c_values[5], c_values[6]],
    font = dict(color = 'black', size = 14)),
    cells=dict(values=[gamma_values, score_matrix[:,0], score_matrix[:,1], score_matrix[:,2], score_matrix[:,3], score_matrix[:,4], score_matrix[:,5], score_matrix[:,6]],
    font = dict(color = 'black', size = 14)))

    data = [trace]
    offline.plot(data)

# main
def main():

    # Load Iris Dataset
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    labels_array = iris.target
    print("First two dimensions of Iris Dataset loaded.\nDataset dimension:", len(X))

    # Normalize dataset
    scal = StandardScaler()
    X_std = scal.fit_transform(X)

    # Train, validation and test sets in proportion 5:2:3
    X_train, X_test, y_train, y_test = train_test_split(
        X_std, labels_array, test_size=.2, random_state=32)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, shuffle=True, test_size=.375, random_state=5)
    print("Train Set:", len(X_train), "\nValidation Set:",
          len(X_val), "\nTest Set:", len(X_test))

    # Plot setup
    h = .02
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    x_ = np.arange(x_min, x_max, h)
    y_ = np.arange(y_min, y_max, h)
    xx, yy = np.meshgrid(x_, y_)

    # Linear SVM
    c_values = np.logspace(start=-3, stop=3, num=7)
    score_vector = np.zeros((len(c_values)))
    print("\nSVM with linear Kernel:")

    # Plot linear SVM
    for j, c in enumerate(c_values):
        clf = svm.SVC(kernel='linear', C=c).fit(X_train, y_train)
        clf_score = clf.score(X_val, y_val)
        print("C =", c, "\nScore =", clf_score)
        score_vector[j] = clf_score
        plot_svm(clf, c, x_, y_, xx, yy, X_train, y_train, "Linear")

    # Plot accuracy on C variation
    plot_accuracy(c_values, score_vector)

    # Evaluation with best C on test set
    c = c_values[np.argmax(score_vector)]
    clf = svm.SVC(kernel='linear', C=c).fit(X_train, y_train)
    clf_score = clf.score(X_test, y_test)
    print("Best C =", c, "\nScore on Test Set =", clf_score)

    # RBF SVM
    score_vector = np.zeros((len(c_values)))
    print("\nSVM with RBF Kernel:")

    # Plot RBF SVM
    for j, c in enumerate(c_values):
        clf = svm.SVC(kernel='rbf', C=c, gamma='auto').fit(X_train, y_train)
        clf_score = clf.score(X_val, y_val)
        print("C =", c, "\nScore =", clf_score)
        score_vector[j] = clf_score
        plot_svm(clf, c, x_, y_, xx, yy, X_train, y_train, "RBF")

    # Plot accuracy on C variation
    plot_accuracy(c_values, score_vector)

    # Evaluation with best C on test set
    c = c_values[np.argmax(score_vector)]
    clf = svm.SVC(kernel='rbf', C=c, gamma='auto').fit(X_train, y_train)
    clf_score = clf.score(X_test, y_test)
    print("Best C =", c, "\nScore on Test Set =", clf_score)

    # Grid search of RBF parameters
    gamma_values = np.logspace(start=-3, stop=3, num=7)
    score_matrix = np.zeros((len(c_values), len(gamma_values)))
    print("\nGrid Search for best RBF parameters:")
    print("C values =", c_values, "\nGamma values =", gamma_values)
    for j, c in enumerate(c_values):
        for i, gamma in enumerate(gamma_values):
            clf = svm.SVC(kernel='rbf', C=c, gamma=gamma).fit(X_train, y_train)
            score_matrix[j, i] = clf.score(X_val, y_val)

    # Tables with parameters score on validation set
    plot_table(c_values, gamma_values, score_matrix)

    # Evaluation with best C and gamma on test set
    best_score = np.unravel_index(score_matrix.argmax(), score_matrix.shape)
    c = c_values[best_score[0]]
    gamma = gamma_values[best_score[1]]
    clf = svm.SVC(kernel='rbf', C=c, gamma=gamma).fit(X_train, y_train)
    clf_score = clf.score(X_test, y_test)
    print("Best C =", c, "Best gamma =", gamma, "\nScore on Test Set =", clf_score)
    plot_svm(clf, c, x_, y_, xx, yy, X_test, y_test, "RBF")

    # Merge the training and validation split
    X_train = np.concatenate((X_train, X_val), axis=0)
    y_train = np.concatenate((y_train, y_val), axis=0)

    # K-fold cross validation
    print("\nGrid Search for best RBF parameters with K-fold Cross-validation:")
    k_folds = 5
    score_matrix = np.zeros((len(c_values), len(gamma_values)))
    for j, c in enumerate(c_values):
        for i, gamma in enumerate(gamma_values):
            clf = svm.SVC(kernel='rbf', C=c, gamma=gamma)
            score_matrix[j, i] = cross_val_score(clf, X_train, y_train, cv=k_folds).mean()

    # Tables with parameters score on training set after k-fold
    plot_table(c_values, gamma_values, score_matrix)

    # Evaluation with best C and gamma on test set
    best_score = np.unravel_index(score_matrix.argmax(), score_matrix.shape)
    c = c_values[best_score[0]]
    gamma = gamma_values[best_score[1]]
    clf = svm.SVC(kernel='rbf', C=c, gamma=gamma).fit(X_train, y_train)
    clf_score = clf.score(X_test, y_test)
    print("Best C =", c, "Best gamma =", gamma, "K-fold =", k_folds, "\nScore on Test Set =", clf_score)

    sys.exit(0)


if __name__ == "__main__":
    main()
