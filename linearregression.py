import tensorflow as tf
"""
x_train=[1,2,3];
y_train=[1,2,3];

W=tf.Variable(tf.random.normal([1]),name='weight')
b=tf.Variable(tf.random.normal([1]),name='bias')

hypothesis = x_train*W + b;
cost = tf.reduce_mean(tf.square(hypothesis-y_train));

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01);
train = optimizer.minimize(cost);
#그래프 빌드

sess = tf.Session();
sess.run(tf.global_variables_initializer())
for step in range(2001):
    sess.run(train)
    if step % 20==0:
        print(step,sess.run(cost),sess.run(W),sess.run(b))
        print("test")
"""

X = tf.compat.v1.placeholder(tf.float32);
Y = tf.compat.v1.placeholder(tf.float32);

W = tf.Variable(tf.random.normal([1]),name='weight')
b = tf.Variable(tf.random.normal([1]),name='bias')

hypothesis = X*W + b;
cost = tf.reduce_mean(tf.square(hypothesis-Y))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.compat.v1.Session();
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(1001):
    cost_val, W_val, b_val, _ = sess.run([cost,W,b,train],feed_dict={X:[1,2,3],Y:[1,2,3]})
    if step%20 == 0:
        print(step,cost_val,W_val,b_val);

print(sess.run(hypothesis,feed_dict={X:[5]}))
