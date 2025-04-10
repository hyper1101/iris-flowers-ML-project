import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
iris=pd.read_csv('Iris.csv')
""" print(iris.head())
print(iris.describe())
print(iris.tail()) """
iris['Species'] = iris['Species'].map({'Iris-setosa': 1,'Iris-versicolor': 2,'Iris-virginica': 3})
""" print(iris.head())
print(iris.tail()) """
features=['Id','SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
X=iris.loc[:,features]
Y=iris.loc[:,['Species']]
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, train_size = .75)
st_x=StandardScaler()
X_test,X_train=st_x.fit_transform(X_test),st_x.fit_transform(X_train)
print(X_train)
print(X_test)
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
print(y_pred)
from sklearn.metrics import confusion_matrix ,ConfusionMatrixDisplay
cm=confusion_matrix(y_pred,y_test)
cm_display =ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [0,1,2])
cm_display.plot()
plt.show()
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
mesh_points = np.c_[xx.ravel(), yy.ravel()]
approx_inverse_pca = pca.inverse_transform(mesh_points)  # Approximate inverse
scaled_mesh = st_x.transform(approx_inverse_pca)  # Apply scaling
Z = classifier.predict(scaled_mesh)
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF']))
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train.values, cmap=ListedColormap(['red', 'green', 'blue']), edgecolor='k', s=50)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-NN Decision Boundary (PCA Projection)')
plt.legend(handles=scatter.legend_elements()[0], labels=['Setosa', 'Versicolor', 'Virginica'])
plt.show()