import tensorflow as tf

x_data = [ [73.,80.,75.], [93.,88.,93.], [89.,91.,90.], [96.,98.,100.], [73.,66.,70.] ];
y_data = [[15+2.],[185.],[180.],[196.],[142.]];

X = tf.compat.v1.placeholder(tf.float32, shape=[None,3]);
Y = tf.compat.v1.placeholder(tf.float32, shape=[None,1]);

W = tf.compat.v1.Variable(tf.random_normal([3,1]),name='weight');
b = tf.compat.v1.Variable(tf.random_normal([1]),name='bias');

hypothesis = tf.matmul(X,W)+b;

cost = tf.reduce_mean(tf.square(hypothesis-Y));

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5);
train = optimizer.minimize(cost);

sess = tf.compat.v1.Session();
sess.run(tf.global_variables_initializer());

for step in range(2001):
    cost_val,hy_val,_ = sess.run([cost,hypothesis,train],feed_dict={X:x_data,Y:y_data});
    if step & 10 ==0:
        print("cost : ",cost_val ," ","Prediction : ",hy_val,"\n");
         