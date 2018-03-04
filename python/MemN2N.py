import tensorflow as tf
import numpy as np

# train data 가져오기
f = open('train_1k.txt', 'r')
data = f.readlines()
f.close()

# 상수 Constant
_WORD = 20      # 사전의 크기 (사전이 저장할 수 있는 최대 WORD)
_MEMORY = 10    # Memory Vector의 크기
#################


story = []
question = []
ans = []

flag = True

for l in data:
    if flag:
        story.append([])
        flag = False
    line = l.split()
    if '.' in line[-1]:
        line[-1] = line[-1].replace('.', '')
        story[-1].append(line[1:])
    for i, word in enumerate(line):
        if '?' in word:
            line[i] = word.replace('?', '')
            question.append(line[1:i+1])
            ans.append(line[i+1])
            flag = True
            break

# 테스트 출력
print('받아온 문항 수: {}문항'.format(len(ans)))

dictionary = []

def isNumber(s):
  try:
    float(s)
    return True
  except ValueError:
    return False

for l in data:
    line = l.split()
    for word in line:
        if isNumber(word):
            continue
        if '.' in word:
            word = word.replace('.', '')
        elif '?' in word:
            word = word.replace('?', '')
        if word not in dictionary:
            dictionary.append(word)

# 테스트 출력
print('사용된 단어: {}종류\n'.format(len(dictionary)), dictionary)

sentence = np.zeros([_WORD, 1])

storyArr = []
questionArr = []
ansArr = []

for sents in story:
    storyArr.append(np.zeros([_WORD, len(sents)]))
    for i, sent in enumerate(sents):
        s = sentence.copy()
        for word in sent:
            s[dictionary.index(word), 0] = 1
        storyArr[-1][:, [i]] = s

for sent in question:
    s = sentence.copy()
    for word in sent:
        s[dictionary.index(word), 0] = 1
    questionArr.append(s)

for word in ans:
    s = sentence.copy()
    s[dictionary.index(word), 0] = 1
    ansArr.append(s)

'''
# 행렬로 잘 저장되는지 테스트 출력
print("\n")
for i in range(len(ans)):
    print('{}\n{}\n'.format(story[i][0], story[i][1]), '{}\n{}\n'.format(storyArr[i][0].T, storyArr[i][1].T))
for i in range(len(ans)):
    print(question[i], '\n', questionArr[i].T)
for i in range(len(ans)):
    print(ans[i], '\n', ansArr[i].T)
'''

# Input (Sentences)
X = tf.placeholder(tf.float32, shape=[_WORD, None])
# Question q
Q = tf.placeholder(tf.float32, shape=[_WORD, 1])
# Desired Answer
Answer = tf.placeholder(tf.float32, shape=[_WORD, 1])
# Predicted Answer ( 계산해서 나온것 )
hypothesis = None

# Weights
W = tf.Variable(tf.random_normal([_WORD, _MEMORY]), name='weight')
C = tf.transpose(W)
A = tf.Variable(tf.random_normal([_MEMORY, _WORD]), name='Embedding_A')
B = A

# Inputs -> Weights -> Outputs -> O + u -> Predicted Answer
Inputs = tf.matmul(A, X)
u = tf.matmul(B, Q)

def softmax(M):
    M = tf.exp(M)
    M = M / tf.reduce_sum(M)
    return M

Weights = softmax(tf.matmul(tf.transpose(u), Inputs))
Outputs = tf.matmul(C, X)
o = tf.reduce_sum(Outputs * Weights)
hypothesis = softmax(tf.matmul(W, o+u))

# cost
cost = tf.reduce_mean(tf.square(hypothesis - Answer))

# Minimizing
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

print("\n\n[STORY]")
print(" ".join(story[0][0]))
print(" ".join(story[0][1]))
print("[Question] {}?".format(" ".join(question[0])))
print("[Answer] {}".format(ans[0]))
result = sess.run(tf.transpose(hypothesis), feed_dict={X: storyArr[0], Q: questionArr[0], Answer: ansArr[0]})
# result = result * 100
# print(result, np.sum(result), type(result))
i = np.argmax(result)
print("\n[Output] {}".format(dictionary[i]))
