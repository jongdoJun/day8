import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from pandas.io.parsers import read_csv
import numpy as np
"""
Weight,Age 결과 : content[혈액농도]
"""
class Blood:
    def initialize(self,Weight,Age):
        self._Weight = Weight
        self._Age = Age

    @staticmethod
    def raw_data():
        tf.set_random_seed(777)
        return np.genfromtxt('blood.txt', skip_header=36)
    @staticmethod
    def model(raw_data):
        x_data = np.array(raw_data[:,2,4], dtype=np.float32)
        y_data = np.array(raw_data[:,4], dtype=np.float32)
        y_data = y_data.reshape(25,1)

        X = tf.placeholder(tf.float32, shape=[None, 2], name='x_input')  # 데이터의 행, 열구조 [Weight,Age] 2개
        Y = tf.placeholder(tf.float32, shape=[None, 1], name='y_input')  # 가격:1개
        W = tf.Variable(tf.random_normal([2, 1]), name='weight')  # 정규분포로 간다.
        b = tf.Variable(tf.random_normal([1]), name='bias')
        hypothesis = tf.matmul(X, W) + b  # y[예측값] = Wx +b :가설
        cost = tf.reduce_mean(tf.square(hypothesis - Y))  # 가설값에서 실제값을 빼줌
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000005)  # 최적화
        train = optimizer.minimize(cost)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        cost_history=[]

        tf.global_variables_initializer()
        data = read_csv('blood.txt', sep=',')
        xy = np.array(data, dtype=np.float32)
        # learning 과정 avgTemp,minTemp,maxTemp,rainFall 가격[결과값] :avgPrice
        for step in range(2000):
            cost_, hypo_, _ = sess.run([cost, hypothesis, train], {X: x_data, Y: y_data})
            if step % 500 == 0:
                print(f' step :{step}, cost : {cost_}')
                cost_history.append(sess.run(cost, {X:x_data, Y:y_data}))

        saver = tf.train.Saver()
        saver.save(sess, 'blood.ckpt')


    def service(self):
        X = tf.placeholder(tf.float32, shape=[None, 2])
        W = tf.Variable(tf.random_normal([2, 1]), name='weight')
        b = tf.Variable(tf.random_normal([1]),name= 'bias')
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, 'blood/blood.ckpt')#저장된 학습파일 가져오기 : 실행하는 곳이 app이기 때문에 blood하나더 blood/blood.ckpt
            #saver.restore(sess, 'blood.ckpt')
            val = sess.run(tf.matmul(X,W)+b, {X:[[self._weight,self.age]]})
        print(f'혈중지방농도: {val}')
        if val < 150:
            result = '정상'
        elif 150 <= val < 200:
            result = '경계역 중성지방혈증'
        elif 200 <= val < 500:
            result = '고 중성지방혈증'
        elif 500 <= val < 1000:
            result = '초고 중성지방혈증'
        elif 1000 <= val:
            result = '췌장염 발병 가능성 고도화'
        print(result)
        return result

if __name__ == '__main__':

    blood = Blood()
    raw_data = blood.raw_data()
    Blood.model(raw_data)

    #blood.initialize(100,30)
    #blood.service()
    #blood.initialize(Weight, Age)
    #result = blood.service( ) #cabbage 클래스로 값을 보내서 텐서플로우에서 계산된 값을 가져오기

