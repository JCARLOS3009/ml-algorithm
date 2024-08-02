# Importar bibliotecas necessárias
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Dados de treinamento
X_train = np.array([[2, 3], [3, 4], [5, 1], [6, 2]])  # Coordenadas dos pontos
y_train = np.array(['A', 'A', 'B', 'B'])              # Classes correspondentes

# Novo ponto de dados
X_new = np.array([[4, 3]])

# Criar o classificador k-NN
k = 3  # Número de vizinhos
knn = KNeighborsClassifier(n_neighbors=k)

# Treinar o classificador
knn.fit(X_train, y_train)

# Classificar o novo ponto de dados
y_pred = knn.predict(X_new)

print(f"O novo ponto {X_new[0]} é classificado como: {y_pred[0]}")
