### SA-1 ASSIGNMENT
### Objective 1 :
### To Create a scatter plot between cylinder vs Co2Emission (green color)
### Code :
```py
'''
Developed By : K KESAVA SAI
Register Number : 212223230105
'''
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('FuelConsumption.csv')
plt.scatter(df['CYLINDERS'], df['CO2EMISSIONS'], color='green')
plt.xlabel('Cylinders')
plt.ylabel('CO2 Emission')
plt.title('Cylinder vs CO2 Emission')
plt.show()
```
### Output :
![image](https://github.com/Kesavasai20/ML-WORKSHOP/assets/138849303/2a9f1ee5-0782-4808-bbfd-0b49ffe084e8)

### Objective 2 :
### Using scatter plot compare data cylinder vs Co2Emission and Enginesize Vs Co2Emission using different colors
### Code :
```py
'''
Developed By : K KESAVA SAI
Register Number : 212223230105
'''
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('FuelConsumption.csv')
plt.scatter(df['CYLINDERS'], df['CO2EMISSIONS'], color='blue', label='Cylinder')
plt.scatter(df['ENGINESIZE'], df['CO2EMISSIONS'], color='red', label='Engine Size')
plt.xlabel('Cylinders/Engine Size')
plt.ylabel('CO2 Emission')
plt.title('Cylinder vs CO2 Emission and Engine Size vs CO2 Emission')
plt.legend()
plt.show()
```
### Output :
![image](https://github.com/Kesavasai20/ML-WORKSHOP/assets/138849303/63b4ab2e-5879-41c8-be11-9e8b1eec6917)

### Objective 3 :
### Using scatter plot compare data cylinder vs Co2Emission and Enginesize Vs Co2Emission and FuelConsumption_comb Co2Emission using different colors
### Code :
```py
'''
Developed By : K KESAVA SAI
Register Number : 212223230105
'''
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('FuelConsumption.csv')
plt.scatter(df['CYLINDERS'], df['CO2EMISSIONS'], color='green', label='Cylinder')
plt.scatter(df['ENGINESIZE'], df['CO2EMISSIONS'], color='red', label='Engine Size')
plt.scatter(df['FUELCONSUMPTION_COMB'], df['CO2EMISSIONS'], color='yellow', label='Fuel Consumption')
plt.xlabel('Cylinders/Engine Size/Fuel Consumption')
plt.ylabel('CO2 Emission')
plt.title('Cylinder vs CO2 Emission, Engine Size vs CO2 Emission, and Fuel Consumption vs CO2 Emission')
plt.legend()
plt.show()
```
### Output :
![image](https://github.com/Kesavasai20/ML-WORKSHOP/assets/138849303/dc0fc430-4983-4c30-9b17-c61a4d6d8943)

### Objective 4 :
### Train your model with independent variable as cylinder and dependent variable as Co2Emission
### Code :
```py
'''
Developed By : K KESAVA SAI
Register Number : 212223230105
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df = pd.read_csv('FuelConsumption.csv')
X_cylinder = df[['CYLINDERS']]
y_co2 = df['CO2EMISSIONS']
X_train_cylinder, X_test_cylinder, y_train_cylinder, y_test_cylinder = train_test_split(X_cylinder, y_co2, test_size=0.2, random_state=42)
model_cylinder = LinearRegression()
model_cylinder.fit(X_train_cylinder, y_train_cylinder)
```
### Output :
![image](https://github.com/Kesavasai20/ML-WORKSHOP/assets/138849303/52ce42d8-4140-44d3-bb15-1a2b2cadfa66)

### Objective 5 :
### Train another model with independent variable as FuelConsumption_comb and dependent variable as Co2Emission
### Code :
```py
'''
Developed By : K KESAVA SAI
Register Number : 212223230105
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df = pd.read_csv('FuelConsumption.csv')
X_fuel = df[['FUELCONSUMPTION_COMB']]
y_co2 = df['CO2EMISSIONS']
X_train_fuel, X_test_fuel, y_train_fuel, y_test_fuel = train_test_split(X_fuel, y_co2, test_size=0.2, random_state=42)
model_fuel = LinearRegression()
model_fuel.fit(X_train_fuel, y_train_fuel)
```
### Output :
![image](https://github.com/Kesavasai20/ML-WORKSHOP/assets/138849303/bdf7bd5b-e5f4-4614-bd9a-d6fcc7fcb024)

### Objective 6 :
### Train your model on different train test ratio and train the models and note down their accuracies
### Code :
```py
'''
Developed By : K KESAVA SAI
Register Number : 212223230105
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv('FuelConsumption.csv')
X_cylinder = df[['CYLINDERS']]
y_co2 = df['CO2EMISSIONS']
ratios = [0.1, 0.4, 0.5, 0.8]
for ratio in ratios:
    X_train, X_test, y_train, y_test = train_test_split(X_cylinder, y_co2, test_size=ratio, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Train-Test Ratio: {1-ratio}:{ratio} - Mean Squared Error: {mse:.2f}, R-squared: {r2:.2f}')
```
### Output :
![image](https://github.com/Kesavasai20/ML-WORKSHOP/assets/138849303/85753716-daaa-4a12-a6c7-f01baa2475d4)

### Result : 
All the programs executed successfully

### Developed By : K KESAVA SAI
### Register Number : 212223230105
### Dept : AIDS







