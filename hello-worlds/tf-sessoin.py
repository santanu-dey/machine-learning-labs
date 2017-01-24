import tensorflow as tf

matrix1 = tf.constant([[1. , 2. , 3.],
					   [4. , 5. ,6.]]);

matrix2 = tf.constant([[1],
						[2],[3]]);
matrix3 = tf.constant([[7. , 8.],
                       [9. , 10.],
                       [11. , 12.]]);

myTensor = tf.constant([ [[1,2],
							[3,4],
							[5,6]],
							[[7,8],
							[9,10],
							[11,12]] ]);
myTensor2 =  tf.ones([100,100,100]) * 0.5
myTensor3 =  tf.ones([100,100,100]) * 0.5

neg_myTensor2 = tf.neg(myTensor2);
add_myTensor = tf.add(myTensor2, myTensor3);
product = tf.matmul( matrix1, matrix3);


print(matrix1);
print(matrix2);
print(myTensor);
print(myTensor2);
print(neg_myTensor2);
print(add_myTensor);

sess = tf.InteractiveSession();
result = add_myTensor.eval();
print("Results");
print(result);
print(neg_myTensor2.eval());
print(product.eval());
print("shape of matrix1: ", matrix1.get_shape());
print("shape of matrix3: ", matrix3.get_shape());
print("shape of myTensor2: ", myTensor2.get_shape());
sess.close();