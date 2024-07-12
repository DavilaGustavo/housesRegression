import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scikeras.wrappers import KerasRegressor

# Armazenamento dos dados presentes no .csv
base = pd.read_csv('House Price India.csv')

# Tratamento dos dados
base = base.drop(['id', 'Date', 'number of views', 'Postal Code', 'Lattitude', 'Longitude', 'Number of schools nearby', 'Distance from the airport'], axis=1)
base = base[base['number of bedrooms'] < 5]
base = base[base['Price'] < 1000000]
base = base[base['Built Year'] > 1940]

previsores = base.iloc[:, 0:13].values
preco_real = base.iloc[:, 14].values

# Escalonamento dos dados
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# Dividir os dados em treinamento/teste (75/25)
previsores_train, previsores_test, preco_train, preco_test = train_test_split(previsores, preco_real, test_size=0.25)

# Estrutura da rede neural
regressor = Sequential()
regressor.add(Dense(units=8, activation='relu', input_dim=previsores.shape[1]))
# regressor.add(Dropout(0.2))
regressor.add(Dense(units=8, activation='relu'))
# regressor.add(Dropout(0.2))
regressor.add(Dense(units=4, activation='relu'))
# regressor.add(Dropout(0.2))
regressor.add(Dense(units=1, activation='linear'))
regressor.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

# Treinamento
regressor.fit(previsores_train, preco_train, batch_size=64, epochs=600)

# Prever os valores
preco_previsto = regressor.predict(previsores_test)

# Cria um dataframe com o preço real, preço estimado e o percentual de erro entre eles
resultado = pd.DataFrame({
    'Preço real': preco_test,
    'Preço previsto': preco_previsto.ravel(),
    'Erro': abs((preco_test - preco_previsto.ravel()) / preco_test) * 100
})

# Exibe os primeiros 15 resultados para fins de comparação
print(resultado.head(15))

# Exibe a média dos percentuais de erro
media_erro = resultado['Erro'].mean()
print(f'Média do Erro Percentual: {media_erro:.2f}%')
