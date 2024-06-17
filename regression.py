import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
import numpy as np
import streamlit as st
import altair as alt

data = pd.read_excel('Car_Purchasing_Data.xlsx')

X = data.drop(columns=['Car Purchase Amount', 'Customer Name', 'Customer e-mail', 'Country'])
y = data['Car Purchase Amount']

X_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_scaled = X_scaler.fit_transform(X)
y_reshape = y.values.reshape(-1, 1)
y_scaled = y_scaler.fit_transform(y_reshape)

random_state = 42

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=random_state)

models = {
	'Linear Regression': LinearRegression(),
	'Decision Tree': DecisionTreeRegressor(),
	'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state),
	'Gradient Boosting': GradientBoostingRegressor(random_state=random_state),
	'SVR': SVR(),
	'K-Neighbors': KNeighborsRegressor(),
	'MLP Regressor': MLPRegressor(random_state=random_state, max_iter=1000),
	'Ridge': Ridge(),
	'Lasso': Lasso(),
	'ElasticNet': ElasticNet()
}

results = {}
predictions = {}

for name, model in models.items():
	model.fit(X_train, y_train) #train model using the training data
	y_pred = model.predict(X_test) #predict the data
	predictions[name] = y_pred
	rmse = mean_squared_error(y_test, y_pred, squared=False)
	results[name] = rmse

st.title('Car Purchase Amount Prediction')
st.write('Model Performance')
rmse_data = pd.DataFrame(list(results.items()), columns=['Model', 'RMSE'])
st.bar_chart(rmse_data.set_index('Model'))



for name, y_pred in predictions.items():
	df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
	chart = alt.Chart(df).mark_circle().encode(
		x='Actual',
		y='Predicted',
		tooltip=['Actual', 'Predicted']
	).properties(
		title=name
	)
	
	st.altair_chart(chart, use_container_width=True)