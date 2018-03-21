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
    str1 = name + "\n" + "[" +", ".join(str(x) for x in M) + "]"
    # str1 <= String
    return str1

def strToMatrix (file):
    matrix = None
    # matrix <= numpy matrix
    return matrix

# M : numpy 형식의 Matrix
# name : 이름

'''
예를 들어서

파라미터
M = [ [1 2 3]
      [4 5 6] ]
name = 'Weights'

리턴값
str1 = 'Weights\n[[1, 2, 3], [4, 5, 6]]'

이 되는 함수를 작성
'''
