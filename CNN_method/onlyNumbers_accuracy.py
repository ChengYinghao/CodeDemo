import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    file_name = 'onlyNumbers.txt'
    iterations = []
    accuracy = []
    file = open(file_name, 'r')
    lines = file.readlines()
    for line in lines:
        data = [float(s) for s in line.split(' ')]
        iterations.append(data[0])
        accuracy.append(data[1])
    file.close()
    plt.plot(iterations, accuracy)
    plt.show()

