import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el archivo BI_Postulantes.xlsx
df = pd.read_excel('BI_Postulantes09.xlsx')

# Seleccionar las columnas relevantes
data = df[['Cod_Especialidad', 'Apertura Nuevos Conoc.', 'Nivel Organización', 
           'Participación Grupo Social', 'Grado Empatía', 'Grado Nerviosismo', 
           'Dependencia Internet']]

# Aplicar el algoritmo k-means con 3 clusters
kmeans = KMeans(n_clusters=3, random_state=0).fit(data)

# Agregar la columna de clusters al DataFrame original
df['Cluster'] = kmeans.labels_

# Generar histogramas cruzando dimensiones
sns.pairplot(df, hue='Cluster', vars=['Cod_Especialidad', 'Apertura Nuevos Conoc.', 
                                      'Nivel Organización', 'Participación Grupo Social', 
                                      'Grado Empatía', 'Grado Nerviosismo', 
                                      'Dependencia Internet'])
plt.show()
