from subprocess import check_call

import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.tree import export_graphviz

X = [[0, 0], [1, 1], [0, 1], [1, 0]]
Y = [0, 0, 1, 1]
col = ['red', 'green']
marker = ['o', 'd']
index = 0
while index < len(X):
    type = Y[index]
    plt.scatter(X[index][0], X[index][1],
                c=col[type], marker=marker[type])
    index += 1
plt.show()

classifier = tree.DecisionTreeClassifier()
classifier.fit(X, Y)
# manual make a directory graph
export_graphviz(classifier, out_file='graph\\simple1.dot',
                filled=True, rounded=True,
                special_characters=True)
#check_call(['dot', '-Tsvg', 'graph\\simple1.dot', '-o', 'graph\\demo17.svg'])
check_call(['dot', '-Tpdf', 'graph\\simple1.dot', '-o', 'graph\\demo17.pdf'])
# check_call(['dot', '-Tpng', 'graph\\simple1.dot', '-o', 'graph\\demo17.png'])