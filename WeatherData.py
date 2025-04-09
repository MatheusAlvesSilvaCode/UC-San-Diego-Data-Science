import pandas as pd
from sklearn.model_selection import train_test_split

# Criando um DataFrame simples
data = pd.DataFrame({
    'temperatura': [20, 21, 19, 18, 22, 23],
    'umidade': [30, 35, 28, 25, 40, 45],
    'baixa_umidade': [1, 0, 1, 1, 0, 0]
})

# X são as variáveis de entrada (features)
X = data[['temperatura', 'umidade']]

# y é a variável de saída (label)
y = data['baixa_umidade']

# Separando em treino e teste (33% para teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Exibindo as divisões dos dados
print("X_train:")
print(X_train)
print("\nX_test:")
print(X_test)
print("\ny_train:")
print(y_train)
print("\ny_test:")
print(y_test)
