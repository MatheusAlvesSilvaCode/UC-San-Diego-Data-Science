import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Lê o arquivo CSV com os dados meteorológicos
data = pd.read_csv(r'data/daily_weather.csv')

# Mostra os nomes das colunas do dataset
data.columns

# Exibe as primeiras 21 linhas do dataframe
data.head(21)

# Mostra todas as linhas que possuem pelo menos um valor nulo
data[data.isnull().any(axis=1)]

# Remove a coluna 'number' do dataframe (provavelmente um índice ou identificador irrelevante)
del data['number']

# Salva o número de linhas antes de remover os valores nulos
before_rows = data.shape[0]
print(before_rows)

# Remove todas as linhas com valores nulos
data = data.dropna()

# Salva o número de linhas após remover os valores nulos
after_rows = data.shape[0]
print(after_rows)

# Calcula quantas linhas foram removidas
before_rows - after_rows

# Cria uma cópia dos dados limpos para manipulação posterior
clean_data = data.copy()

# Cria uma nova coluna binária chamada 'high_humidity_label': 1 se a umidade às 15h for maior que 24.99, senão 0
clean_data['high_humidity_label'] = (clean_data['relative_humidity_3pm'] > 24.99)*1

# Mostra os valores da nova coluna binária
print(clean_data['high_humidity_label'])

# Define o alvo (variável dependente) como a coluna recém-criada
y = clean_data[['high_humidity_label']].copy()

# Mostra os primeiros valores da coluna 'relative_humidity_3pm'
clean_data['relative_humidity_3pm'].head()

# Mostra os primeiros valores da variável alvo (y)
y.head()

# Mostra as 5 primeiras linhas do dataframe original
data.head(5)

# Define as colunas (features) que serão usadas como entrada para o modelo — todas com dados da manhã
morning_feature = ['air_pressure_9am','air_temp_9am','avg_wind_direction_9am','avg_wind_speed_9am','max_wind_direction_9am','max_wind_speed_9am',
                   'rain_accumulation_9am','rain_duration_9am']

# Cria um novo dataframe x com apenas as features selecionadas
x = clean_data[morning_feature].copy()

# Mostra os nomes das colunas de x
x.columns

# Mostra os nomes das colunas de y
y.columns

# Divide os dados em treino (67%) e teste (33%) com uma semente de aleatoriedade fixa
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=324)

# Cria um classificador de Árvore de Decisão com no máximo 10 folhas e uma semente fixa
humidity_classifier = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)

# Treina (ajusta) o classificador com os dados de treino
humidity_classifier.fit(X_train, y_train)

# Mostra o tipo do objeto classificador
type(humidity_classifier)

# Faz previsões com os dados de teste
predictions = humidity_classifier.predict(X_test)

# Mostra as 10 primeiras previsões do modelo
predictions[:10]

# Mostra os 10 primeiros valores reais (esperados) da variável alvo no conjunto de teste
y_test['high_humidity_label'][:10]

# Avalia a precisão do modelo comparando as previsões com os valores reais
accuracy_score(y_true = y_test, y_pred = predictions)
