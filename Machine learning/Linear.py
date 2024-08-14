import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

time_studied = np.array([18, 60, 46, 12, 32, 53, 26, 35, 42, 42, 55, 10, 30, 36]).reshape(-1, 1)
scores = np.array([46, 75, 64, 35, 55, 56, 66, 43, 55, 60, 85, 30, 45, 40]).reshape(-1, 1)

time_train, time_test, score_train, score_test = train_test_split(time_studied, scores, test_size=0.3)

model = LinearRegression()
model.fit(time_train, score_train)

print(model.score(time_test, score_test))

plt.scatter(time_train, score_train)
plt.plot(np.linspace(0,70,100).reshape(-1,1), model.predict(np.linspace(0,70,100).reshape(-1,1)), color='blue')
plt.show()
