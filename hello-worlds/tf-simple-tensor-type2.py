import tensorflow as tf

matrix1 = tf.constant([[1., 2.]]);
matrix2 = tf.constant([[1],
						[2]]);
myTensor = tf.constant([ [[1,2],
							[3,4],
							[5,6]],
							[[7,8],
							[9,10],
							[11,12]] ]);
myTensor2 =  tf.ones([500,500,500]) * 0.5
myTensor3 =  tf.ones([500,500,500]) * 0.5

neg_myTensor2 = tf.neg(myTensor2);
add_myTensor = tf.add(myTensor2, myTensor3);

print(matrix1);
print(matrix2);
print(myTensor);
print(myTensor2);
print(neg_myTensor2);
print(add_myTensor);