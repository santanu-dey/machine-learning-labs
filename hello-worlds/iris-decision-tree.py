from sklearn import datasets
import numpy as np
from sklearn import tree 
import pydotplus as py_dot
from sklearn.externals.six import StringIO

test_idx = [0,50,100];

iris= datasets.load_iris();
print(iris.feature_names);
print(iris.target_names);
print(iris.data[0]);
print(iris.target[0]);


#training data
train_target = np.delete(iris.target, test_idx);
train_data = np.delete(iris.data, test_idx, axis=0);

# testing data 
test_target = iris.target[test_idx];
test_data =  iris.data[test_idx]


# create a decision tree classifier
clf = tree.DecisionTreeClassifier();
clf = clf.fit(train_data, train_target);



print (test_target);
print (clf.predict(test_data));

dot_data = StringIO();

tree.export_graphviz(clf, 
						 out_file=dot_data, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,
                         impurity=False,  
                         special_characters=True)  
graph = py_dot.graph_from_dot_data(dot_data.getvalue());
graph.write_pdf("iris.pdf");  