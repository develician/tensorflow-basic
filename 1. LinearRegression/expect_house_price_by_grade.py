import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('./kc_house_data.csv', dtype=float,
                     delimiter=',', names=True)

# print(data['price'], data['grade'])


grade_arr = data['grade']
price_arr = data['price']


W = tf.Variable(tf.random_normal([1], -1., 1.))
b = tf.Variable(tf.random_normal([1], -1., 1.))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = W * X + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(30000):
        _, cost_val = sess.run([train_op, cost], feed_dict={
                               X: grade_arr, Y: price_arr})

        print(step, cost_val, sess.run(W), sess.run(b))

    print("grade가 4일때, price 예측값:", sess.run(hypothesis, feed_dict={X: 4}))
    print("grade가 8일때, price 예측값:", sess.run(hypothesis, feed_dict={X: 8}))

    plt.figure(1)
    plt.title('집 등급에 따른 집값 예측')
    plt.xlabel('집 등급')
    plt.ylabel('예측된 집값')
    # x_data와 y_data를 기준으로 점을 찍는다.
    plt.plot(grade_arr, price_arr, 'ro')
    # 예측한 일차함수를 선으로 표시
    plt.plot(grade_arr, sess.run(W) * grade_arr + sess.run(b), 'b')
    # x = 5 일때 초록점 찍기
    plt.plot([4], sess.run(hypothesis, feed_dict={X: 4}), 'go')
    # x = 10 일때 초록점 찍기
    plt.plot([8], sess.run(hypothesis, feed_dict={X: 8}), 'go')
