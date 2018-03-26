import tensorflow as tf
import numpy as np

def softmax(M):
    M = tf.exp(M)
    M = M / tf.reduce_sum(M)
    return M

def refineWord(list_):
    for i, word in enumerate(list_):
        if '.' in word:
            list_[i] = list_[i].replace('.', '')
        elif '?' in word:
            list_[i] = list_[i].replace('?', '')

def matrixToStr (M, name='None'):
    MList = M.tolist()
    str1 = name + "\n" + "[" +", ".join(str(x) for x in MList) + "]"
    return str1

def strToMatrix (file):
    matrix = np.array(file)
    return matrix

def bars(persent, name=''):
    bars = ""
    persentCnt = persent
    while True:
        persentCnt -= 2
        bars = bars + "|"
        if persentCnt < 2:
            break
    str1 = name.ljust(12) + bars + " " + str(persent) + "%" 
    return str1
