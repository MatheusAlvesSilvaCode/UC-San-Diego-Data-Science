import sqlite3
import pandas as pd 
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

cnx = sqlite3.connect(r'C:\Users\mathe\Downloads\archive (5)\database.sqlite')
df = pd.read_sql_query("SELECT * FROM Player_Attributes", cnx)
df.head(5)
df.shape
df.columns

feature = [
       'potential', 'crossing', 'finishing', 'heading_accuracy',
       'short_passing', 'volleys', 'dribbling', 'curve', 'free_kick_accuracy',
       'long_passing', 'ball_control', 'acceleration', 'sprint_speed',
       'agility', 'reactions', 'balance', 'shot_power', 'jumping', 'stamina',
       'strength', 'long_shots', 'aggression', 'interceptions', 'positioning',
       'vision', 'penalties', 'marking', 'standing_tackle', 'sliding_tackle',
       'gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning',
       'gk_reflexes']

target = ['overall_rating']
df = df.dropna() # Retiando valores nulos
x = df[feature] # Criando um novo dataframe que contém apenas as colunas de feature
y = df[target] # Criando um novo dataframe com contém apenas as colunas de target 'overrall_rating'
x.iloc[2] # Acessando índice 2 do dataframe x
y
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33, random_state=324 )
regressor = LinearRegression() # Cria um modelo de regressão Linear
regressor.fit(x_train, y_train) # Treina o modelo com entrada com e saída
y_prediction = regressor.predict(x_test)
y_prediction
y_test.describe()
#Root Mean Squared Error (RMSE)
#Pega a diferença entre o que o modelo previu e o que era o valor real, eleva ao quadrado, soma tudo e tira a média.
RMSE = sqrt(mean_squared_error(y_true= y_test, y_pred= y_prediction)) 
print(RMSE)
regressor = DecisionTreeRegressor(max_depth=20)
regressor.fit(x_train, y_train)
y_prediction = regressor.predict(x_test)
y_prediction
y_test.describe()
RMSE = sqrt(mean_squared_error(y_true = y_test, y_pred = y_prediction))
print(RMSE)