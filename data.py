import pandas as pd
import plotly.express as pe

import matplotlip.pylot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression


data = pd.read_csv("scores.csv")

score = data["Score"].tolist()
accepted = data["Accepted"].tolist()

fig = pe.scatter(x = score, y = accepted)
# fig.show()

# ----------------------------------------------------------------

X = np.reshape(score , (len(score),1))
Y = np.reshape(accepted , (len(accepted),1))

lr = LogisticRegression()
lr.fit(X,Y)


plt.figure()
plt.scatter(X.ravel() , Y , color='black' , zorder= 20)


def model(X):
    return 1/(1 + np.exp(-X))


X_test = np.linspace(0, 100, 200)

chances = model(X_test * lr.coef_ + lr.intercept_).ravel()

plt.plot(X_test, chances, color='red', linewidth=3)
plt.axhline(y=0, color='k', linestyle='-')
plt.axhline(y=1, color='k', linestyle='-')
plt.axhline(y=0.5, color='b', linestyle='--')

# do hit and trial by changing the value of X_test
plt.axvline(x=X_test[165], color='b', linestyle='--')

plt.ylabel('y')
plt.xlabel('X')
plt.xlim(75, 85)
# plt.show()

# ---------------------------------------------------------------------------

userinfo = float(input("Enter your marks here:"))

chances = model(X_test * lr.coef_ + lr.intercept_).ravel()


if chances<0.01:
    print("Sorry, you have not been accepted")
elif chances >= 1:
    print("Congratulations, you have been accepted")
elif chances< 0.5:
    print("Might not get accepted") 
else:
    print("Might get accepted")9``
