import tensorflow as tf
from math import pi
mean =  tf.ones([100]) * 1.0
sigma = tf.ones([100]) * 0.2

myTensor =  tf.ones([100]) * 1.0

myTensor = tf.Print(myTensor, [myTensor], message="This is the distribution: ");

standard_distribution = ((tf.exp(tf.neg(tf.pow(tf.sub(myTensor, mean), 2.0) /
(2.0 * tf.pow(sigma, 2.0) ))) * (1.0 / (tf.mul(sigma, tf.sqrt(2.0 * pi) )))));


with tf.Session() as sess:
    print (sess.run(mean))
    print (sess.run(myTensor))

    