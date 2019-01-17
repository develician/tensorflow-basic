# $ conda install tensorflow
# $ conda install matplotlib

import tensorflow as tf
import matplotlib.pyplot as plt

x_data = [1, 2, 3]
y_data = [2, 4, 6]

# 위와 같은 두배의 값을 규칙으로 하는 값들이 있다고 하자.

# y = W * X + b
# 출력값 = 가중치 * 입력값 + 편향

W = tf.Variable(tf.random_normal([1], -1., 1.))
b = tf.Variable(tf.random_normal([1], -1., 1.))

# 가중치와 편향 값을 랜덤하게 하나를 뽑는 작업.

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# x_data와 y_data의 값을 담을 플레이스홀더를 선언.

hypothesis = W * X + b

# 위의 식을 바탕으로 hypothesis 라는 식 선언
# X와 W가 행렬이 아니므로 tf.matmul이 아니라 기본 곱셈 연산자를 사용.

# 손실함수 작성 : 잔차의 최소 제곱법을 적용한후 모든 데이터에 대한 손실값의 평균을 내어 구한것.
# 잔차 = hypothesis - y
# tf.square(잔차) -> 잔차의 최소 제곱법을 적용
# tf.reduce_mean(tf.square(잔차)) -> 잔차의 최소 제곱법을 적용한후 모든 데이터에 대한 손실값의 평균을 내어 구한것.
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# 경사 하강법 최적화 함수를 이용해 손실값을 최소화 하는 연산 그래프 생성.
# 최적화 함수
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(100):
        _, cost_val = sess.run([train_op, cost], feed_dict={
                               X: x_data, Y: y_data})

        print(step, cost_val, sess.run(W), sess.run(b))
        # 여기서 찍힌 값을 보면, step이 늘어날수록, 손실값은 작아지는 것을 볼수 있다.

    prediction_X_4 = sess.run(hypothesis, feed_dict={X: 4})
    prediction_X_5 = sess.run(hypothesis, feed_dict={X: 5})
    print("X: 5 일때 예측값 Y:", prediction_X_4)
    print("X: 10 일때 예측값 Y:", prediction_X_5)

    plt.figure(1)
    plt.title('Linear Regression')
    plt.xlabel('입력값')
    plt.ylabel('출력값')
    # x_data와 y_data를 기준으로 점을 찍는다.
    plt.plot(x_data, y_data, 'ro')
    # 예측한 일차함수를 선으로 표시
    plt.plot(x_data, sess.run(W) * x_data + sess.run(b), 'b')
    # x = 5 일때 초록점 찍기
    plt.plot([4], prediction_X_4, 'go')
    # x = 10 일때 초록점 찍기
    plt.plot([5], prediction_X_5, 'go')
