import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
class Calculator:
    @staticmethod
    def add_model():
        w1 = tf.placeholder(tf.float32, name = 'w1')
        w2 = tf.placeholder(tf.float32, name = 'w2')
        feed_dict = {'w1' : 8.0, 'w2' : 2.0}#더미데이터 학습을 위해서
        r = tf.add(w1, w2, name = 'op_add')
        sess = tf.Session()
        _=tf.Variable(initial_value='fake_variable')
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        result = sess.run(r, {w1: feed_dict['w1'], w2 : feed_dict['w2']})
        print(f'덧셈 결과 {result}')
        saver.save(sess, './calculator_add_model/model', global_step=1000)

    @staticmethod
    def sub_model():
        w1 = tf.placeholder(tf.float32, name = 'w1')
        w2 = tf.placeholder(tf.float32, name = 'w2')
        feed_dict = {'w1' : 8.0, 'w2' : 2.0}
        r = tf.subtract(w1, w2, name = 'op_sub')
        sess = tf.Session()
        _=tf.Variable(initial_value='fake_variable')
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        result = sess.run(r, {w1: feed_dict['w1'], w2 : feed_dict['w2']})
        print(f'뺄셈 결과 {result}')
        saver.save(sess,'./calculator_sub_model/model', global_step=1000)

        @staticmethod
        def mul_model():
            w1 = tf.placeholder(tf.float32, name='w1')
            w2 = tf.placeholder(tf.float32, name='w2')
            feed_dict = {'w1': 8.0, 'w2': 2.0}
            r = tf.multiply(w1, w2, name='op_mul')
            sess = tf.Session()
            _ = tf.Variable(initial_value='fake_variable')
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            result = sess.run(r, {w1: feed_dict['w1'], w2: feed_dict['w2']})
            print(f'곱셈 결과 {result}')
            saver.save(sess,'./calculator_mul_model/model', global_step=1000)

        @staticmethod
        def div_model():
            w1 = tf.placeholder(tf.float32, name='w1')
            w2 = tf.placeholder(tf.float32, name='w2')
            feed_dict = {'w1': 8.0, 'w2': 2.0}
            r = tf.divide(w1, w2, name='op_div')
            sess = tf.Session()
            _ = tf.Variable(initial_value='fake_variable')
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            result = sess.run(r, {w1: feed_dict['w1'], w2: feed_dict['w2']})
            print(f'나누기 결과 {result}')
            saver.save(sess,'./calculator_div_model/model', global_step=1000)
    @staticmethod
    def service(num1,num2, opcode):
        print(f'{num1} {opcode} {num2}')
        tf.reset_default_graph()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.import_meta_graph(f'calculator/calculator_{opcode}_model/model-1000.meta')
            saver.restore(sess, tf.train.latest_checkpoint(f'calculator/calculator_{opcode}_model'))
            graph = tf.get_default_graph()
            w1 = graph.get_tensor_by_name('w1:0')
            w2 = graph.get_tensor_by_name('w2:0')
            feed_dict = {w1: float(num1), w2: float(num2)}
            for key in feed_dict.keys():
                print(key, ':', feed_dict[key])
            op_to_restore = graph.get_tensor_by_name(f'op_{opcode}:0')
            result = sess.run(op_to_restore, feed_dict)
            print(f'텐서가 계산한 값 : {result}')
        return result
if __name__ == '__main__':
    if not os.path.exists('calculator_add_model/checkpoint'):
        Calculator.add_model()
    if not os.path.exists('calculator_sub_model/checkpoint'):
        Calculator.sub_model()
    #if not os.path.exists('calculator_mul_model/checkpoint'):
    Calculator.mul_model()
    #if not os.path.exists('calculator_div_model/checkpoint'):
    #    Calculator.div_model()
