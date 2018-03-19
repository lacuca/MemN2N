import tensorflow as tf
import numpy as np
import data
import utils
from model import cMemN2N

# 상수 Constant
test = 1  # TEST 할 문제 번호
stddev = 0.1  # 표준 편차
learning_rate = 0.01  # 학습률
_WORD = 20  # 사전의 크기 (사전이 저장할 수 있는 최대 WORD)
_MEMORY = 177  # Memory Vector 의 크기

###############################
# Data Reading
###############################
print("[Constant]", "test: {}, stddev: {}, learning_rate: {}, WORD: {}, MEMORY: {}"
      .format(test, stddev, learning_rate, _WORD, _MEMORY))

dataTrain = data.cData(_WORD, _MEMORY)
story, question, ans, storyArr, questionArr, ansArr, dictionary = dataTrain.read('train_1k.txt')

###############################
# MemN2N
###############################
MemN2N = cMemN2N(_WORD, _MEMORY, dictionary, (story, question, ans), (storyArr, questionArr, ansArr))
MemN2N.set(stddev, learning_rate, test)

