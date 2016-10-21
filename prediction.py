import xlrd
import xlwt
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def save_dataset(filename, prediction, realvalue):
    '''
    save the prediction as xls file
    '''
    book = xlwt.Workbook()
    sheet1 = book.add_sheet("Sheet1")
    for num in xrange(len(prediction)):
        row = sheet1.row(num)
        row.write(0, prediction[num])
        row.write(1, realvalue[num])
    book.save(filename)


def load_dataset(filename):
    '''
    load dataset.xls and convert it into list
    '''
    with xlrd.open_workbook(filename, 'rb') as dataset:
        table = dataset.sheets()[0]
        nrows = table.nrows
        ncols = table.ncols
        datas = []
        for row in xrange(1, nrows):
            row_data = [table.cell(row, 0).value]  # the first col saves date
            for col in xrange(1, ncols):
                # save as float type
                row_data.append(float(table.cell(row, col).value))
            datas.append(row_data)
    return datas


def reshape_datas(datas):
    '''
    raw:datas.shape=(nrows-1,ncols)
    target: shaped_data.shape=(one_hot,)
    temperature | sunshine | air pressure | wind speed | rainfall
    380           20         40             10           20      = 470
    76            5          5              5            5       = 96
    '''
    reshape_datas = []
    for row in xrange(len(datas)):  # nrows-1
        # temperature 380: [-2,36)
        temp_temp = [0] * 380
        temp_indice = int(datas[row][1] * 10) + 20
        temp_temp[temp_indice] = 1
        # sunshine 20: [0,20)
        temp_sun = [0] * 20
        temp_indice = int(round(datas[row][2]))
        temp_sun[temp_indice] = 1
        # air pressure 40: [0,40)
        temp_air = [0] * 40
        temp_indice = int(round(datas[row][3]))
        temp_air[temp_indice] = 1
        # wind speed 10: [0,10)
        temp_wind = [0] * 10
        temp_indice = int(round(datas[row][4]))
        temp_wind[temp_indice] = 1
        # rainfall 20: [0,200)
        temp_rain = [0] * 20
        temp_indice = int(round(datas[row][5] / 10.0))
        temp_rain[temp_indice] = 1
        # concate all temps
        row_data = temp_temp + temp_sun + temp_air + temp_wind + temp_rain
        reshape_datas.append(row_data)
    return reshape_datas


def get_random_batch(datas, n_steps, batch_size):
    '''
    input in the form of (temp,sun,air,wind,rain) shape=(5,)
    output in the form of (0,0,...,1,...,0) shape=(380,)
    [-2,36) 1 decimal points, (36-(-2))*10
    '''
    random_batch_x = []
    random_batch_y = []
    indices = np.random.randint(0, len(datas) - n_steps, batch_size)
    for i in indices:
        # random_batch_x.shape=(batch_size,n_steps,470)
        temp_x = []  # temp_x.shape=(n_steps,470)
        for step in xrange(n_steps):
            temp_x.append(datas[i + step])  # datas[i].shape=(470,)
        random_batch_x.append(temp_x)
        temp_y = datas[i + n_steps][:380]
        random_batch_y.append(temp_y)
    return random_batch_x, random_batch_y


def get_all_batch(datas, n_steps):
    batch_x = []
    batch_y = []
    for indice in xrange(len(datas) - n_steps):
        temp_x = []
        for step in xrange(n_steps):
            temp_x.append(datas[indice + step])
        batch_x.append(temp_x)
        temp_y = datas[indice + n_steps][:380]
        batch_y.append(temp_y)
    return batch_x, batch_y


def reform_y_display(test_y):
    # test_y.shape=(len(test_datas),380)
    indices = np.argmax(test_y, 1)  # indices.shape=(len(test_datas),)
    display_y = (indices - 20) / 10.0
    return display_y


training_iters = 350000
learning_rate = 0.001
display_step = 200
batch_size = 128
# network parameters
n_input = 470
n_steps = 2
n_hidden = 180
n_output = 380

# tf graph weights, biases
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_output])
# define weights, biases
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_output]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_output]))
}


def RNN(x, weights, biases):
    # permuting batch_size and n_input
    x = tf.transpose(x, [1, 0, 2])
    # reshape into (n_steo*batch_size,n_input)
    x = tf.reshape(x, [-1, n_input])
    # split to get a list of 'n_steps'
    x = tf.split(0, n_steps, x)
    with tf.variable_scope('n_steps4'):
        lstm_cell = rnn_cell.BasicLSTMCell(
            n_hidden, forget_bias=1.0)
        outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)
# define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# read dataset
filename = 'Kyoto_weather_1986_2015.xls'
datas = load_dataset(filename)
datas = reshape_datas(datas)
print('Data Reading Finished!')

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < training_iters:
        batch_x, batch_y = get_random_batch(datas, n_steps, batch_size)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter" + str(step * batch_size) + ", Minibatch Loss=" +
                  "{:.6f}".format(loss) + ", Training Accuracy=" +
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # test
    test_datas = load_dataset('Kyoto_weather_2016.xls')
    test_datas = reshape_datas(test_datas)
    print("Test Data Reading Finished!")
    test_x, test_y = get_all_batch(test_datas, n_steps)
    pred_test_y = sess.run(pred, feed_dict={x: test_x})
    test_step = 0
    display_pred_test_y = reform_y_display(pred_test_y)
    display_test_y = reform_y_display(test_y)
    while test_step < len(test_datas) - n_steps:
        if test_step % 3 == 0:
            print("pred: {}, real: {}".format(
                display_pred_test_y[test_step], display_test_y[test_step]
            ))
        test_step += 1
    print("Testing Cost:",
          sess.run(cost, feed_dict={x: test_x, y: test_y}))
    print("Testing Accuracy:",
          sess.run(accuracy, feed_dict={x: test_x, y: test_y}))

# change the n_steps and name of the file
save_dataset("2016_2.xls", display_pred_test_y, display_test_y)
print("Prediction Saved")

plt.plot(display_pred_test_y, 'g')
plt.plot(display_test_y, 'r')
red_patch = mpatches.Patch(color='red', label='real')
green_patch = mpatches.Patch(color='green', label='pred')
plt.legend(handles=[red_patch, green_patch])
plt.show()
