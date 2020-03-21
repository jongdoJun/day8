import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from pandas.io.parsers import read_csv
import numpy as np
"""
avgTemp,minTemp,maxTemp,rainFall,avgPrice
"""
class Cabbage:
    def model(self):
        tf.global_variables_initializer()
        data = read_csv('cabbage_price.csv',sep=',')
        xy = np.array(data, dtype=np.float32)
        x_data = xy[:, 1:-1]
        y_data = xy[:, [-1]]#y값 전체
        X = tf.placeholder(tf.float32, shape=[None, 4]) #데이터의 행, 열구조 [avgTemp,minTemp,maxTemp,rainFall] 4개
        Y = tf.placeholder(tf.float32, shape=[None, 1] ) #가격:1개
        W = tf.Variable(tf.random_normal([4, 1]), name='weight') #정규분포로 간다.
        b = tf.Variable(tf.random_normal([1]), name='bias')

        hypothesis = tf.matmul(X, W) +b # y[예측값] = Wx +b :가설
        cost = tf.reduce_mean(tf.square(hypothesis - Y))#가설값에서 실제값을 빼줌
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000005)#최적화
        train = optimizer.minimize(cost)
        #learning 과정 avgTemp,minTemp,maxTemp,rainFall 가격[결과값] :avgPrice
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for step in range(100000):
                cost_, hypo_, _ = sess.run([cost, hypothesis, train], {X:x_data, Y:y_data})
                if step %500 ==0:
                    print(f' step :{step}, cost : {cost_}')
                    print(f' price: {hypo_}')

            saver = tf.train.Saver()
            saver.save(sess, 'cabbage.ckpt')

    def initialize(self,avgTemp,minTemp,maxTemp,rainFall):
        self.avgTemp = avgTemp
        self.minTemp = minTemp
        self.maxTemp = maxTemp
        self.rainFall = rainFall

    def service(self):
        X = tf.placeholder(tf.float32, shape=[None, 4])
        W = tf.Variable(tf.random_normal([4, 1]), name='weight')
        b = tf.Variable(tf.random_normal([1]),name= 'bias')
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, 'cabbage/cabbage.ckpt')#저장된 학습파일 가져오기
            data = [[self.avgTemp, self.minTemp,self.maxTemp,self.rainFall], ] #리스트구조[]  텐서구조 [[]]
            arr = np.array(data, dtype=np.float32)
            dict = sess.run(tf.matmul(X, W) +b, {X:arr[0:4]})
        return int(dict[0])

if __name__ == '__main__':
    cabbage = Cabbage()
    cabbage.model()