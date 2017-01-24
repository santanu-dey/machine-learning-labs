import tensorflow as tf
from math import pi

with tf.Session() as sess:
	mean_value = 18
	sigma_value = 10
	myTensor =  tf.random_normal([100], mean=mean_value, stddev=sigma_value, dtype=tf.float32);
	print (sess.run(tf.floor(sess.run(myTensor))));
	


    