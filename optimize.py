import xlrd
import tensorflow as tf
import numpy as np
import json


def save_model(filename, weights_1, weights_2, weights_3,
               biases_1, biases_2, biases_3):
    data = {'weights_1': weights_1,
            'weights_2': weights_2,
            'weights_3': weights_3,
            'biases_1': biases_1,
            'biases_2': biases_2,
            'biases_3': biases_3}
    f = open(filename, 'w')
    json.dump(data, f)
    f.close()


def load_dataset(filename):
    '''
    load dataset.xls and save it as python list
    '''
    with xlrd.open_workbook(filename, 'rb') as dataset:
        table = dataset.sheets()[0]
        nrows = table.nrows
        ncols = table.ncols
        datas = []
        for row in xrange(1, nrows):
            row_data = []
            for col in xrange(1, ncols):
                row_data.append(float(table.cell(row, col).value))
            datas.append(row_data)
    return datas  # shape = (357,8)


def calculate_y(x, weights, biases):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['layer1']),
                                   biases['layer1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['layer2']),
                                   biases['layer2']))
    output = tf.add(tf.matmul(layer_2, weights['layer3']),
                    biases['layer3'])
    return output


def get_batch(batch_size, datas):
    indices = np.random.randint(0, len(datas), batch_size)
    random_batch_x = []
    random_batch_y = []
    for i in indices:
        temp_x = datas[i][:7]
        temp_y = [datas[i][7]]
        random_batch_x.append(temp_x)
        random_batch_y.append(temp_y)
    return random_batch_x, random_batch_y

# Parameters
learning_rate = 0.01
batch_size = 32
display_step = 41
# Network Parameters
layer1 = 6
layer2 = 4
x = tf.placeholder(tf.float32, [None, 7])
weights = {
    'layer1': tf.Variable(tf.random_normal([7, layer1])),
    'layer2': tf.Variable(tf.random_normal([layer1, layer2])),
    'layer3': tf.Variable(tf.random_normal([layer2, 1]))
}
biases = {
    'layer1': tf.Variable(tf.random_normal([layer1])),
    'layer2': tf.Variable(tf.random_normal([layer2])),
    'layer3': tf.Variable(tf.random_normal([1]))
}


pred_y = calculate_y(x, weights, biases)
y = tf.placeholder(tf.float32, [None, 1])

cost = tf.reduce_mean(tf.pow(y - pred_y, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


datas = load_dataset('2015_2_8.xls')
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for i in xrange(8000):
        batch_x, batch_y = get_batch(batch_size, datas)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if i % display_step == 0:
            # print(sess.run(pred_y, feed_dict={x: batch_x}))
            print(sess.run(cost, feed_dict={x: batch_x, y: batch_y}))
    save_model('opt_model_paras', weights['layer1'].eval().tolist(),
               weights['layer2'].eval().tolist(),
               weights['layer3'].eval().tolist(),
               biases['layer1'].eval().tolist(),
               biases['layer2'].eval().tolist(),
               biases['layer3'].eval().tolist())
    print('Opt Model Para Saved')
