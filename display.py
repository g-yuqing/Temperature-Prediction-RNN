import xlrd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def load_model(filename):
    f = open(filename, 'r')
    data = json.load(f)
    f.close()
    weights = []
    biases = []
    weights_1 = data['weights_1']
    weights_2 = data['weights_2']
    weights_3 = data['weights_3']
    biases_1 = data['biases_1']
    biases_2 = data['biases_2']
    biases_3 = data['biases_3']
    weights.append(weights_1)
    weights.append(weights_2)
    weights.append(weights_3)
    biases.append(biases_1)
    biases.append(biases_2)
    biases.append(biases_3)
    return weights, biases


def load_dataset_all(filename):
    with xlrd.open_workbook(filename, 'rb') as dataset:
        table = dataset.sheets()[0]
        nrows = table.nrows
        ncols = table.ncols
        pred_datas = []  # shape=(nrows-1, ncols-2)
        real_datas = []  # shape=(nrows-1, 1)
        for row in xrange(1, nrows):
            row_data = []
            for col in xrange(1, ncols - 1):
                row_data.append(float(table.cell(row, col).value))
            pred_datas.append(row_data)
            real_datas.append([float(table.cell(row, ncols - 1).value)])
    return pred_datas, real_datas


def load_dataset_incol(filename):
    '''
    load dataset.xls and save it as python list
    '''
    with xlrd.open_workbook(filename, 'rb') as dataset:
        table = dataset.sheets()[0]
        nrows = table.nrows
        ncols = table.ncols
        datas = []
        for col in xrange(1, ncols):
            col_data = []
            for row in xrange(1, nrows):
                col_data.append(float(table.cell(row, col).value))
            datas.append(col_data)
    return datas  # shape=(ncols-1,nrows-1)


def display_all():
    datas = load_dataset_incol('2016_2_8.xls')
    plt.title('Temperature in 2016')
    plt.xlabel('date')
    plt.ylabel('temperature')
    plt.plot(datas[0], 'b')
    plt.plot(datas[2], 'm')
    plt.plot(datas[6], 'g')
    plt.plot(datas[7], 'r')
    blue_patch = mpatches.Patch(color='blue', label='pred2')
    yellow_patch = mpatches.Patch(color='magenta', label='pred4')
    green_patch = mpatches.Patch(color='green', label='pred8')
    red_patch = mpatches.Patch(color='red', label='real')
    plt.legend(handles=[blue_patch, yellow_patch, green_patch, red_patch])
    plt.show()


def display_detail():
    datas = load_dataset_incol('2016_2_8.xls')
    # plt.title('Temperature')
    plt.subplot(4, 1, 1)
    # plt.xlabel('2 days')
    plt.ylabel('temperature')
    plt.plot(datas[7][247:250], color='r', linewidth=2)
    plt.scatter(2, datas[0][249], s=60, color='b')
    blue_patch = mpatches.Patch(color='blue', label='2-day pred')
    red_patch = mpatches.Patch(color='red', label='real')
    plt.legend(handles=[blue_patch, red_patch], loc=2)

    plt.subplot(4, 1, 2)
    # plt.xlabel('4 days')
    plt.ylabel('temperature')
    plt.plot(datas[7][245:250], color='r', linewidth=2)
    plt.scatter(4, datas[2][249], s=60, color='m')
    magenta_patch = mpatches.Patch(color='magenta', label='4-day pred')
    red_patch = mpatches.Patch(color='red', label='real')
    plt.legend(handles=[magenta_patch, red_patch], loc=2)

    plt.subplot(4, 1, 3)
    # plt.xlabel('6 days')
    plt.ylabel('temperature')
    plt.plot(datas[7][243:250], color='r', linewidth=2)
    plt.scatter(6, datas[4][249], s=60, color='y')
    yellow_patch = mpatches.Patch(color='yellow', label='6-day pred')
    red_patch = mpatches.Patch(color='red', label='real')
    plt.legend(handles=[yellow_patch, red_patch], loc=2)

    plt.subplot(4, 1, 4)
    # plt.xlabel('8 days')
    plt.ylabel('temperature')
    plt.plot(datas[7][241:250], color='r', linewidth=2)
    plt.scatter(8, datas[6][249], s=60, color='g')
    green_patch = mpatches.Patch(color='green', label='8-day pred')
    red_patch = mpatches.Patch(color='red', label='real')
    plt.legend(handles=[green_patch, red_patch], loc=3)

    plt.show()


def display_optimization():
    pred_datas, real_datas = load_dataset_all('2016_2_8.xls')
    weights, biases = load_model('opt_model_paras')
    weights1 = np.matrix(weights[0])
    weights2 = np.matrix(weights[1])
    weights3 = np.matrix(weights[2])
    biases1 = np.matrix(biases[0])
    biases2 = np.matrix(biases[1])
    biases3 = np.matrix(biases[2])
    layer1 = sigmoid(pred_datas * weights1 + biases1)
    layer2 = sigmoid(layer1 * weights2 + biases2)
    result = layer2 * weights3 + biases3

    plt.title('Improved Temperature in 2016')
    plt.xlabel('date')
    plt.ylabel('temperature')
    plt.plot(result, 'g', linewidth=1.5)
    plt.plot(real_datas, 'r', linewidth=1.5)
    red_patch = mpatches.Patch(color='red', label='real')
    green_patch = mpatches.Patch(color='green', label='pred')
    plt.legend(handles=[red_patch, green_patch])
    plt.show()


def display_error():
    pred_datas, real_datas = load_dataset_all('2016_2_8.xls')
    weights, biases = load_model('opt_model_paras')
    weights1 = np.matrix(weights[0])
    weights2 = np.matrix(weights[1])
    weights3 = np.matrix(weights[2])
    biases1 = np.matrix(biases[0])
    biases2 = np.matrix(biases[1])
    biases3 = np.matrix(biases[2])
    layer1 = sigmoid(pred_datas * weights1 + biases1)
    layer2 = sigmoid(layer1 * weights2 + biases2)
    result = layer2 * weights3 + biases3
    result = result.tolist()
    err = []
    abs_err = []
    for i in xrange(len(real_datas)):
        err.append(real_datas[i][0] - result[i][0])
        abs_err.append(abs(real_datas[i][0] - result[i][0]))
    sum_abs_err = 0
    for i in abs_err:
        sum_abs_err += i
    average_abs_err = sum_abs_err / ((len(abs_err)) * 1.0)
    x_pos = np.arange(len(err))
    plt.bar(x_pos, err, align='center', alpha=0.5, color='m')
    plt.xlabel('date')
    plt.ylabel('temperature')
    plt.title('error-average:{}'.format(average_abs_err))
    plt.show()

if __name__ == '__main__':
    # display_all()
    # display_optimization()
    # display_error()
    display_detail()
