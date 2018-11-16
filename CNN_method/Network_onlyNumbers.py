import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from core.CNN_method.Data_onlyNumbers import gen_captcha_text_image, number, get_next_batch, convert2gray


def cnn_structure(w_alpha=0.01, b_alpha=0.1, debug=False):
    x = tf.reshape(X, shape=[-1, image_height, image_width, 1])

    wc1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
    bc1 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, wc1, strides=[1, 1, 1, 1], padding='SAME'), bc1))
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    out1 = tf.nn.dropout(pool1, keep_prob)

    wc2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    bc2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(out1, wc2, strides=[1, 1, 1, 1], padding='SAME'), bc2))
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    out2 = tf.nn.dropout(pool2, keep_prob)

    wc3 = tf.get_variable(name='wc3', shape=[3, 3, 64, 128], dtype=tf.float32,
                          initializer=tf.contrib.layers.xavier_initializer())
    # wc3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 128]))
    bc3 = tf.Variable(b_alpha * tf.random_normal([128]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(out2, wc3, strides=[1, 1, 1, 1], padding='SAME'), bc3))
    pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    out3 = tf.nn.dropout(pool3, keep_prob)

    wd1 = tf.get_variable(name='wd1', shape=[8 * 20 * 128, 1024], dtype=tf.float32,
                          initializer=tf.contrib.layers.xavier_initializer())
    # wd1 = tf.Variable(w_alpha * tf.random_normal([7*20*128,1024]))
    bd1 = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(out3, [-1, wd1.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, wd1), bd1))
    dense = tf.nn.dropout(dense, keep_prob)

    wout = tf.get_variable('name', shape=[1024, max_captcha * char_set_len], dtype=tf.float32,
                           initializer=tf.contrib.layers.xavier_initializer())
    # wout = tf.Variable(w_alpha * tf.random_normal([1024, max_captcha * char_set_len]))
    bout = tf.Variable(b_alpha * tf.random_normal([max_captcha * char_set_len]))
    out = tf.add(tf.matmul(dense, wout), bout)
    if (debug):
        print("x size=", x.shape)
        print("conv1 size =", conv1.shape)
        print("pool1 size =", pool1.shape)
        print("out1 size =", out1.shape)
        print("conv2 size =", conv2.shape)
        print("pool2 size =", pool2.shape)
        print("out2 size =", out2.shape)
        print("conv3 size =", conv3.shape)
        print("pool3 size =", pool3.shape)
        print("out3 size =", out3.shape)
        print("dense size =", dense.shape)
    return out


def train_cnn(debug=False):
    output = cnn_structure(debug=debug)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    predict = tf.reshape(output, [-1, max_captcha, char_set_len])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, max_captcha, char_set_len]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        step = 0
        while True:
            batch_x, batch_y = get_next_batch(100, image_height, image_width, char_set_len, max_captcha)
            _, cost_ = sess.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
            print(step, cost_)
            if step % 10 == 0:
                batch_x_test, batch_y_test = get_next_batch(100, image_height, image_width, char_set_len, max_captcha)
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                print(step, acc)
                if acc > 0.99:
                    saver.save(sess, "./model_numbers/crack_captcha.model", global_step=step)
                    break
            step += 1


def crack_captcha(captcha_image):
    output = cnn_structure()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "./model_numbers/crack_captcha.model-2510")

        predict = tf.argmax(tf.reshape(output, [-1, max_captcha, char_set_len]), 2)
        text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1.})
        text = text_list[0].tolist()
        return text


if __name__ == '__main__':
    train = 1
    if train == 0:
        text, image = gen_captcha_text_image()
        print("验证码大小：", image.shape)  # (60,160,3)

        image_height = 60
        image_width = 160
        max_captcha = len(text)
        print("验证码文本最长字符数", max_captcha)
        char_set = number
        char_set_len = len(char_set)

        X = tf.placeholder(tf.float32, [None, image_height * image_width])
        Y = tf.placeholder(tf.float32, [None, max_captcha * char_set_len])
        keep_prob = tf.placeholder(tf.float32)
        train_cnn()

    if train == 1:
        image_height = 60
        image_width = 160
        char_set = number
        char_set_len = len(char_set)
        X = tf.placeholder(tf.float32, [None, image_height * image_width])
        keep_prob = tf.placeholder(tf.float32)
        correct_text, image = gen_captcha_text_image(number, 4)
        # image = plt.imread('0062.png', 'r')
        image = np.array(image)
        max_captcha = 4
        image_array = convert2gray(image)
        image_array = image_array.flatten() / 255
        predict_text = crack_captcha(image_array)
        print("正确文本为：", correct_text)
        print("预测文本为：", predict_text)
        plt.imshow(image)
        plt.show()

