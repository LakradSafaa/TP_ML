import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv("./cars_CO2_emission.csv")
features = ['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']
data = df[features]
train, test = train_test_split(data, test_size=0.2, random_state=42)
x_train = train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']]
y_train = train['CO2EMISSIONS']
x_test = test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']]
y_test = test['CO2EMISSIONS']
model = LinearRegression()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
print('Coefficients:', model.coef_)
print("Mean squared error: %.2f" % mean_squared_error(y_test, predictions))
print('Variance score: %.2f' % r2_score(y_test, predictions))

plt.figure(figsize=(10, 6))
plt.hist(data['CO2EMISSIONS'], bins=30, color='skyblue', edgecolor='black')
plt.xlabel("CO2 Emissions")
plt.ylabel("Frequency")
plt.title("Distribution of CO2 Emissions")
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data['CO2EMISSIONS'], color='lightgreen')
plt.xlabel("CO2 Emissions")
plt.title("Box Plot of CO2 Emissions")
plt.show()

sns.pairplot(data, diag_kind='kde', markers='+')
plt.suptitle("Pair Plot of Features and CO2 Emissions", y=1.02)
plt.show()

plt.figure(figsize=(10, 6))
sns.residplot(predictions, y_test - predictions, lowess=True, color='purple')
plt.xlabel("Predicted CO2 Emissions")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()
