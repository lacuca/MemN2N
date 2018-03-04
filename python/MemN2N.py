import tensorflow as tf
import numpy as np

# train data 가져오기
f = open('train_1k.txt', 'r')
data = f.readlines()
f.close()

# 상수 Constant
_WORD = 20      # 사전의 크기 (사전이 저장할 수 있는 최대 WORD)
_MEMORY = 11    # Memory Vector의 크기
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
for i in range(_WORD):
    dictionary.append('')
index = 0

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
            dictionary[index] = word
            index = index + 1

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
C = tf.transpose(W)     # Embedding C
A = tf.Variable(tf.random_normal([_MEMORY, _WORD]), name='Embedding_A')
B = A                   # Embedding B

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
cost = tf.reduce_sum(tf.square(hypothesis - Answer))

# Minimizing
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

print("\nTrain이 잘됐는지 다음 스토리로 테스트 해봅시다\n\n[STORY]")
print(" ".join(story[0][0]))
print(" ".join(story[0][1]))
print("[Question] {}?".format(" ".join(question[0])))
print("[Answer] {}".format(ans[0]))

'''
# TEST
for step in range(1001):
    if step % 200 == 0:
        result = sess.run(tf.transpose(hypothesis), feed_dict={X: storyArr[0], Q: questionArr[0], Answer: ansArr[0]})
        i = np.argmax(result)
        print("[{}'s Output] {}".format(step, dictionary[i]))
        W_value, A_value, cost_value = sess.run([W, A, cost],
                                                feed_dict={X: storyArr[0], Q: questionArr[0], Answer: ansArr[0]})
        print(W_value[0, 0], cost_value)
    sess.run(train, feed_dict={X: storyArr[0], Q: questionArr[0], Answer: ansArr[0]})
'''

# 초기값설정
cost_mean = 1000
step = 0

# Training
while cost_mean > 0.5:
    if step % 1 == 0:
        result = sess.run(tf.transpose(hypothesis), feed_dict={X: storyArr[0], Q: questionArr[0], Answer: ansArr[0]})
        i = np.argmax(result)
        print("[{}'s Output] {}".format(step, dictionary[i]))
        print(cost_mean)
    cost_mean = 0
    for index in range(len(storyArr)):
        sess.run(train, feed_dict={X: storyArr[index], Q: questionArr[index], Answer: ansArr[index]})
        cost_value = sess.run(cost, feed_dict={X: storyArr[index], Q: questionArr[index], Answer: ansArr[index]})
        cost_mean = cost_mean + cost_value
    cost_mean = cost_mean / len(storyArr)
    step = step + 1

result = sess.run(tf.transpose(hypothesis), feed_dict={X: storyArr[0], Q: questionArr[0], Answer: ansArr[0]})
i = np.argmax(result)
print("[{}'s Output] {}".format(step, dictionary[i]))
print(cost_mean)