import tensorflow as tf
weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35),
                      name="weights");
                     
biases = tf.Variable(tf.zeros([200]), name="biases");
init_op = tf.initialize_all_variables();
sess = tf.InteractiveSession();
sess.run(init_op);
print(weights.eval());
sess.close();