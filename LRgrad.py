import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
def costf(i_b,i_m,X,Y):
    sumcst=0
    for i in range(len(X)):
        sumcst+=((i_m*X[i]+i_b)-Y[i])**2
    return sumcst*0.5/len(X)
def stepgradient(i_b,i_m,alpha,X,Y):
    sum_b=0
    sum_m=0
    for i in range(len(X)):
        sum_b+=(i_m*X[i]+i_b)-Y[i]
        sum_m+=((i_m *X[i] +i_b)-Y[i])*X[i]
    i_b=i_b-(alpha*1.0/len(X)*sum_b)
    i_m = i_m - (alpha * 1.0 / len(X) * sum_m)
    return [i_b,i_m]
dataset=pandas.read_csv("data.csv")
X=dataset.values[:,0]
Y=dataset.values[:,1]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
alpha=0.0001
initial_b=0
initial_m=0
iterations=1000
cost=[]
for i in range(iterations):
    initial_b,initial_m=stepgradient(initial_b,initial_m,alpha,X_train,y_train)
    cost.append(costf(initial_b,initial_m,X_train,y_train))
print initial_b
print initial_m
#initial_b=0.0889365199374
#initial_m=1.47774408519
predict=[]
for i in range(len(X_test)):
    predict.append(initial_m*X_test[i]+initial_b)
predictnp=np.array(predict)
print r2_score(y_test,predictnp)
f1=plt.figure(1)
plt.scatter(X_test, y_test)
yfit = [initial_b + initial_m * xi for xi in X_test]         # Y LIST.
plt.plot(X_test, yfit)                                       # LINE OF BEST FIT.
itr=[]
for i in range(15):
    itr.append(i)
f2=plt.figure(2)
plt.plot(itr,cost[0:15])
plt.xlabel("No of iterations")
plt.ylabel("Cost")
plt.show()