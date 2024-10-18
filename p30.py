import pandas as pd
data=pd.read_excel("salary.xlsx")
x=data.iloc[:,:-1].values
y=data.iloc[:,1].values

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x,y)
ypred=model.predict(x)

from sklearn.metrics import r2_score
print("r2 = ",r2_score(y,ypred))

import matplotlib.pyplot as plt
plt.scatter(x,y,label="data")
plt.plot(x,ypred,label="fit")
plt.legend()
plt.show()
