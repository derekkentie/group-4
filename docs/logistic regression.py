import numpy 

X = numpy.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1,1) #data
y = numpy.array([0, 0, 0, 0, 1, 1, 1, 1]) #yes or no
from sklearn import linear_model
log = linear_model.LogisticRegression()
log.fit(X, y)

predict = log.predict(numpy.array([7]).reshape(-1,1))

print(predict)