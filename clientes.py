import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Cargar el archivo BI_Clientes09.xlsx
df = pd.read_excel('BI_Clientes09.xlsx')

# Seleccionar las columnas relevantes
data = df[['HouseOwnerFlag', 'NumberCarsOwned', 'Age', 'BikeBuyer']]

# Dividir los datos en conjunto de entrenamiento y prueba
X = data.drop('BikeBuyer', axis=1)
y = data['BikeBuyer']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar un modelo de árbol de decisiones
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Visualizar el árbol de decisiones
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns, class_names=['No Bike Buyer', 'Bike Buyer'], filled=True)
plt.show()
