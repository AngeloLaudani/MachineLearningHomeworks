import sys
import time
import numpy as np
from PIL import Image
from os import listdir
from os.path import join
import matplotlib.pyplot as plt
from plotly import tools
import plotly.offline as offline
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split


# Path for the Images
path = 'D:\PCA\PACS_homework'
img_num = 9
classes = []


# Get classes name
def get_classes():
    classes = [folder for folder in listdir(path)]
    print("Selected classes:", classes)
    return classes


# Project Picture
def project_picture(comp, X_std, scal):
    pca = PCA(n_components=comp)
    X_pca_image = pca.fit_transform(X_std)
    print("Picture with first", str(comp), "PC")
    X_pca_reverse = scal.inverse_transform(pca.inverse_transform(X_pca_image))
    print("Variance of dataset with first", str(comp), "PC:",
          pca.explained_variance_ratio_.cumsum()[comp - 1] * 100, "%")
    plt.imshow(X_pca_reverse[img_num].astype(int).reshape(227, 227, 3))
    plt.show()


# Naive Bayes Classification
def bayes_class(classes, X_std, labels_array, isNotComplete, comp1, comp2):
    X_train, X_test, y_train, y_test = train_test_split(
        X_std, labels_array, test_size=.4, random_state=42)
    gnb = GaussianNB()
    bayes = gnb.fit(X_train, y_train)
    y_pred = bayes.predict(X_test)
    print("Naive Bayes Classification")
    print("Number of mislabeled points out of a total %d points: %d" %
          (X_test.shape[0], (y_test != y_pred).sum()))
    score = gnb.score(X_test, y_test)
    print("Accuracy:", score * 100, "%\n")
    if isNotComplete:
        return
        #plot_bayes(bayes, classes, X_train, X_test)


# Show results with Plotly Python Library
def plot_pca(classes, matrix, labels_array, comp1, comp2):
    traces = []

    for name in classes:

        trace = go.Scatter(
            x=matrix[labels_array == name, 0],
            y=matrix[labels_array == name, 1],
            mode='markers',
            name=name,
            marker=dict(
                size=12,
                line=dict(
                    color='rgba(217, 217, 217, 0.14)',
                    width=0.5),
                opacity=0.8))
        traces.append(trace)

    layout = dict(xaxis=dict(title='PC' + str(comp1), showline=False),
                  yaxis=dict(title='PC' + str(comp2), showline=False),
                  title='PCA with PC' + str(comp1) + ' and PC' + str(comp2))
    fig = dict(data=traces, layout=layout)
    offline.plot(fig)


"""def plot_bayes(clf, classes, X_train, X_test):
    colors = ['red','green','blue','purple']
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    colors = clf.predict(X_test)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=colors, s=20, edgecolor='k')
    plt.suptitle("Naive Bayes Classification")
    plt.show()"""


# main
def main():
    classes = get_classes()

# Load all the images in the dataset array
    dataset_array = []

    for class_folder in classes:
        elements = [join(path, join(class_folder, file)) for file in listdir(
            join(path, class_folder)) if file.endswith("jpg") or file.endswith("png")]
        for element in elements:
            dataset_array.append(
                (np.asarray(Image.open(element)), class_folder))

    examples_number = len(dataset_array)
    print("Images loaded:", examples_number)

# Obtain ordinal labels of Images
    dataset_np = np.asarray(dataset_array)
    labels_array = dataset_np[:, 1]

# Obtain flat array for dataset
    flat_dataset_array = []

    for img_data in dataset_array:
        flat_dataset_array.append(img_data[0].ravel())

    examples_features = len(flat_dataset_array[0])
    print("Features for every image:", examples_features, "\n")

# Standardize: mean 0 and variance 1
    scal = StandardScaler()
    X_std = scal.fit_transform(flat_dataset_array)

# Show original picture
    print("Original picture")
    print("Variance of original dataset: 100 %")
    plt.imshow(flat_dataset_array[img_num].reshape(227, 227, 3))
    plt.show()

# Show picture with first 60 PC
    project_picture(60, X_std, scal)

# Show picture with first 6 PC
    project_picture(6, X_std, scal)

# Show picture with first 2 PC
    project_picture(2, X_std, scal)

# Show picture with last 6 PC
    pca = PCA()
    pca.fit_transform(X_std)
    pca_last_comp = pca.components_[-6:, :]
    X_pca_last = X_std.dot(pca_last_comp.T)
    X_pca_reverse = scal.inverse_transform(X_pca_last.dot(pca_last_comp))
    print("Picture with last 6 PC")
    print("Variance of dataset with last 6 PC:",
          (1 - pca.explained_variance_ratio_.cumsum()[1080]) * 100, "%\n")
    plt.imshow(X_pca_reverse[img_num].astype(int).reshape(227, 227, 3))
    plt.show()


# PCA for the first 2 PC
    pca = PCA(n_components=2)
    X_pca1 = pca.fit_transform(X_std)
    plot_pca(classes, X_pca1, labels_array, 1, 2)
    print("PCA computed for 1st and 2nd components")

# PCA for the 3rd and 4th PC
    pca = PCA(n_components=4)
    X_pca2 = pca.fit_transform(X_std)
    plot_pca(classes, X_pca2[:, [2, 3]], labels_array, 3, 4)
    print("PCA computed for 3rd and 4th components")

# PCA for the 10th and 11th PC
    pca = PCA(n_components=11)
    X_pca3 = pca.fit_transform(X_std)
    plot_pca(classes, X_pca3[:, [9, 10]], labels_array, 10, 11)
    print("PCA computed for 10th and 11th components\n")

# Variance for each PC
    var_ex = pca.explained_variance_ratio_
    var_ex_cum = pca.explained_variance_ratio_.cumsum()
    traces = []

    trace1 = go.Bar(
        x=['PC %s' % i for i in range(1, 12)],
        y=var_ex,
        showlegend=False)
    traces.append(trace1)

    trace2 = go.Scatter(
        x=['PC %s' % i for i in range(1, 12)],
        y=var_ex_cum,
        name='cumulative explained variance')
    traces.append(trace2)

    layout = dict(
        yaxis=dict(title='Explained variance in percent'),
        title='Explained variance by different principal components')

    fig = dict(data=traces, layout=layout)
    offline.plot(fig)

# Classification for unmodified dataset
    bayes_class(classes, X_std, labels_array, 0, 0, 0)

# Classification for the first 2 PC
    bayes_class(classes, X_pca1, labels_array, 1, 1, 2)
    # sleep needed for correct output on plotly
    time.sleep(1)

# Classification for the 3rd and 4th
    bayes_class(classes, X_pca2[:, [2, 3]], labels_array, 1, 3, 4)

    sys.exit(0)


if __name__ == "__main__":
    main()
