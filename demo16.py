from sklearn import tree

X = [[0, 0], [1, 1]]
Y = [0, 1]
classifier = tree.DecisionTreeClassifier()
classifier.fit(X, Y)
print(classifier)

print(classifier.predict([[2, 2], [2, -2], [-2, -2], [3, 5], [5, -3]]))