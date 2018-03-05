import tensorflow as tf
import numpy as np

# train data 가져오기
f = open('train_1k.txt', 'r')
data = f.readlines()
f.close()

# 상수 Constant
_WORD = 20  # 사전의 크기 (사전이 저장할 수 있는 최대 WORD)
_MEMORY = 30  # Memory Vector의 크기
#################


story = []
question = []
ans = []

story.append([])
for l in data:
    line = l.split()
    if float(line[0]) == 1:
        del story[-1]
        story.append([])
    if '.' in line[-1]:
        line[-1] = line[-1].replace('.', '')
        story[-1].append(line[1:])
    else:
        for i, word in enumerate(line):
            if '?' in word:
                line[i] = word.replace('?', '')
                question.append(line[1:i + 1])
                ans.append(line[i + 1])
                story.append(story[-1].copy())
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
print('사용된 단어: {}종류\n'.format(dictionary.index('')), dictionary)

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

# Weights
W = tf.Variable(tf.random_normal([_WORD, _MEMORY], stddev=0.3), name='weight')
C = tf.transpose(W)  # Embedding C
A = tf.Variable(tf.random_normal([_MEMORY, _WORD], stddev=0.3), name='Embedding_A')
B = A  # Embedding B

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
hypothesis = softmax(tf.matmul(W, o + u))
# cost
cost = tf.reduce_mean(-tf.reduce_sum(Answer * tf.log(hypothesis), axis=1))
# cost = tf.reduce_sum(tf.square(hypothesis - Answer))

# Minimizing
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.005)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

print("\nTrain이 잘됐는지 다음 스토리로 테스트 해봅시다\n\n[STORY]")
test = 8   # TEST 할 문제 번호
for i in range(len(story[test])):
    print(" ".join(story[test][i]))
print("[Question] {}?".format(" ".join(question[test])))
print("[Answer] {}".format(ans[test]))

'''
# TEST
result = None
for step in range(1001):
    if step % 200 == 0:
        result = sess.run(tf.transpose(hypothesis), feed_dict={X: storyArr[2], Q: questionArr[2], Answer: ansArr[2]})
        i = np.argmax(result)
        print("[{}'s Output] {}".format(step, dictionary[i]))
        W_value, A_value, cost_value = sess.run([W, A, cost],
                                                feed_dict={X: storyArr[2], Q: questionArr[2], Answer: ansArr[2]})
        print(W_value[0, 0], cost_value)
    sess.run(train, feed_dict={X: storyArr[2], Q: questionArr[2], Answer: ansArr[2]})
'''

# 초기값설정
cost_mean = 1000
step = 0
cnt = 0
# Training
while cost_mean > 0.063:
    W_value, A_value, result = sess.run([W, A, tf.transpose(hypothesis)],
                                        feed_dict={X: storyArr[test], Q: questionArr[test], Answer: ansArr[test]})
    i = np.argmax(result)
    print("[After {} Loops, Output] {}".format(step, dictionary[i]))
    print("Cost", cost_mean, "W", W_value[0, 0], "A", A_value[0, 0])
    if dictionary[i] == ans[test]:  # Desired Answer 에 수렴하면 10 Loop 후 정지
        cnt = cnt + 1
        if cnt >= 10 and cost_mean < 0.066:
            break
    cost_mean = 0
    for index in range(len(ans)):
        # if index % 201 == 0:
        #     print(storyArr[index])
        #     input('%d Anything:' % index)
        sess.run(train, feed_dict={X: storyArr[index], Q: questionArr[index], Answer: ansArr[index]})
        cost_value = sess.run(cost, feed_dict={X: storyArr[index], Q: questionArr[index], Answer: ansArr[index]})
        str1 = '{}'.format(cost_value)
        if str1 == 'nan':
            print('[nan ERROR at %d]' % index)
            print(story[index])
            print(sess.run(W, feed_dict={X: storyArr[index], Q: questionArr[index], Answer: ansArr[index]}))
            exit()
        cost_mean = cost_mean + cost_value
    cost_mean = cost_mean / len(ans)
    step = step + 1

W_value, A_value, result = sess.run([W, A, tf.transpose(hypothesis)],
                                    feed_dict={X: storyArr[test], Q: questionArr[test], Answer: ansArr[test]})
i = np.argmax(result)
print("[After {} Loops, Output] {}".format(step, dictionary[i]))
print("Cost", cost_mean, "W", W_value[0, 0], "A", A_value[0, 0])

# 사용자 인터페이스
flag = True


def refineWord(list):
    for i, word in enumerate(list):
        if '.' in word:
            list[i] = list[i].replace('.', '')
        elif '?' in word:
            list[i] = list[i].replace('?', '')


while flag:
    print('\n\nStory를 입력해주세요')
    print('END 를 입력하면 Story 입력을 종료합니다')
    print('파일로도 가능합니다 [예: FILE test.txt]')
    user_story = []
    while True:
        user = input()
        if user == 'END':
            break
        elif 'FILE' in user:
            temp = user.split()
            file = temp[1]
            f = open(file, 'r')
            data = f.readlines()
            f.close()
            for i in range(len(data)):
                data[i] = data[i].split()
                refineWord(data[i])
            user_story = data
            break
        else:
            temp = user.split()
            refineWord(temp)
            user_story.append(temp)
    print('Question을 입력해주세요')
    user = input()
    user_question = user.split()
    refineWord(user_question)
    # print('Answer을 입력해주세요')
    # user_answer = input()

    user_storyArr = np.zeros([_WORD, len(user_story)])
    user_questionArr = np.zeros([_WORD, 1])
    # user_answerArr = np.zeros([_WORD, 1])

    try:
        for i, sent in enumerate(user_story):
            s = np.zeros([_WORD, 1])
            for word in sent:
                s[dictionary.index(word), 0] = 1
            user_storyArr[:, [i]] = s
        for word in user_question:
            user_questionArr[dictionary.index(word), 0] = 1
        # user_answerArr[dictionary.index(user_answer), 0] = 1
    except ValueError:
        print('사전에 없는 단어를 입력하였습니다')
        continue

    print("\n\n[STORY]")
    for story in user_story:
        print(" ".join(story))
    print("[Question] {}?".format(" ".join(user_question)))
    # print("[Answer] {}".format(user_answer))
    result = sess.run(tf.transpose(hypothesis),
                      feed_dict={X: user_storyArr, Q: user_questionArr})
    result = result * 100
    str1 = ""
    for _ in range(5):
        i = np.argmax(result)
        percent = int(result[0][i])
        result[0, i] = -1
        str1 = str1 + "%-12s" % dictionary[i]
        for i in range(int(percent / 2)):
            str1 = str1 + '|'
        str1 = str1 + " {}%\n".format(percent)

    print("[Output]\n{}".format(str1))
