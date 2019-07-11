import tensorflow as tf
x1_data = [73.,93.,89.,96.,73.];
x2_data = [80.,88.,91.,98.,66.];
x3_data = [75.,93.,90.,100.,70.];
y_data = [152.,185.,180.,196.,142.];

X1 = tf.compat.v1.placeholder(tf.float32);
X2 = tf.compat.v1.placeholder(tf.float32);
X3 = tf.compat.v1.placeholder(tf.float32);
Y = tf.compat.v1.placeholder(tf.float32);

w1 = tf.Variable(tf.random.normal([1]), name='weight1');
w2 = tf.Variable(tf.random.normal([1]), name='weight2');
w3 = tf.Variable(tf.random.normal([1]), name='weight3');
b = tf.Variable(tf.random.normal([1]),  name='bias');

hypothesis = w1*X1+w2*X2+w3*X3 + b;
cost = tf.reduce_mean(tf.square(hypothesis-Y));

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5);
train = optimizer.minimize(cost);

sess = tf.compat.v1.Session();
sess.run(tf.compat.v1.global_variables_initializer());
for step in range(2001):
    cost_val,hy_val,_ = sess.run([cost,hypothesis,train],feed_dict={X1:x1_data,X2:x2_data,X3:x3_data,Y:y_data});
    if step % 10 ==0:
        print(step,"cost : ",cost_val, "\nPrediction : \n",hy_val);

## 이렇게 코딩하면 x data가 많아지면 코드가 복잡해진다.
                                                            
