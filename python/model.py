import tensorflow as tf
import numpy as np
import utils

class cMemN2N:
    def __init__(self, word, memory, dictionary, naturalConditions, matrixConditions):
        self._WORD = word
        self._MEMORY = memory
        self.dictionary = dictionary
        self.story = naturalConditions[0]
        self.question = naturalConditions[1]
        self.ans = naturalConditions[2]
        self.storyArr = matrixConditions[0]
        self.questionArr = matrixConditions[1]
        self.ansArr = matrixConditions[2]
        self.X, self.Q, self.Y, self.W, self.A, self.B, self.C = tuple(range(7))

    def set(self, stddev=0.1, learning_rate=0.01, test=1):
        # Input (Sentences)
        self.X = tf.placeholder(tf.float32, shape=[self._WORD, None])
        # Question q
        self.Q = tf.placeholder(tf.float32, shape=[self._WORD, 1])
        # Desired Answer
        self.Y = tf.placeholder(tf.float32, shape=[self._WORD, 1])
        # Predicted Answer ( 계산해서 나온것 )
        # Hypothesis

        # Weights
        self.W = tf.Variable(tf.random_normal([self._WORD, self._MEMORY], stddev=stddev), name='weight')
        self.C = tf.Variable(tf.random_normal([self._MEMORY, self._WORD], stddev=stddev), name='Embedding_C')
        # C = tf.transpose(W)  # Embedding C
        self.A = tf.Variable(tf.random_normal([self._MEMORY, self._WORD], stddev=stddev), name='Embedding_A')
        self.B = tf.Variable(tf.random_normal([self._MEMORY, self._WORD], stddev=stddev), name='Embedding_B')
        # B = A  # Embedding B

        Inputs = tf.matmul(self.A, self.X)
        u = tf.matmul(self.B, self.Q)
        # Weights = softmax(tf.matmul(tf.transpose(u), Inputs))
        Weights = tf.nn.softmax(tf.matmul(tf.transpose(u), Inputs))
        Outputs = tf.matmul(self.C, self.X)
        o = tf.reduce_sum(Outputs * Weights, axis=1, keep_dims=True)
        # hypothesis = softmax(tf.matmul(W, o + u))
        hypothesis = tf.nn.softmax(tf.transpose(tf.matmul(self.W, o + u)))
        hypothesis = tf.transpose(hypothesis)

        # cost function
        # # cross-entropy cost 함수
        cost = tf.reduce_mean(-tf.reduce_sum(self.Y * tf.log(hypothesis), axis=1))
        # # Classification 을 할때 사용 (값이 0, 1 정의)
        # cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
        # # Hypothesis 가 선형일때만 사용가능
        # cost = tf.reduce_sum(tf.square(hypothesis - Answer))

        # Checking
        temp1 = tf.argmax(hypothesis, 0)
        temp2 = tf.argmax(self.Y, 0)
        Answer = temp1[0]
        Correct = tf.equal(temp1[0], temp2[0])
        check_list = (Inputs, u, Weights, Outputs, o, hypothesis, Answer)

        # Minimizing
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train = optimizer.minimize(cost)
        # Launch the graph in a session.
        sess = tf.Session()
        # Initializes global variables in the graph.
        sess.run(tf.global_variables_initializer())

        print("\nTrain이 잘됐는지 다음 스토리로 테스트 해봅시다\n\n[STORY]")
        for i in range(len(self.story[test])):
            print(" ".join(self.story[test][i]))
        print("[Question] {}?".format(" ".join(self.question[test])))
        print("[Answer] {}".format(self.ans[test]))

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

        ##################################
        # 초기값설정
        cost_mean = 100
        accuracy = 0
        purpose = 90    # 목표 정확도 (95% 가 목표다)
        step = 0
        cnt = 0
        ##################################
        #           Training             #
        ##################################
        while True:
            if step % 10 == 0:
                W_value, Answer_value = sess.run([self.W, Answer],
                                                 feed_dict={self.X: self.storyArr[test],
                                                            self.Q: self.questionArr[test], self.Y: self.ansArr[test]})
                Answer_value = self.dictionary[Answer_value]
                print("[After {} Loops, Output] {}".format(step, Answer_value))
                print("Accuracy {}%".format(accuracy), "Cost", cost_mean, "W", W_value[0, 0])
            if accuracy > purpose:
                print("현재 Accuracy {}% 입니다. 계속하시겠습니까?".format(accuracy))
                user = input("[Y/N]")
                if user == 'Y':
                    purpose = purpose + 1
                else:
                    break
            previous_cost_mean = cost_mean
            cost_mean = 0
            accuracy = 0
            for index in range(len(self.ans)):
                sess.run(train, feed_dict={self.X: self.storyArr[index],
                                           self.Q: self.questionArr[index], self.Y: self.ansArr[index]})
                cost_value, Correct_value, check_str = sess.run([cost, Correct, check_list],
                                                                feed_dict={self.X: self.storyArr[index],
                                                                           self.Q: self.questionArr[index],
                                                                           self.Y: self.ansArr[index]})
                str1 = '{}'.format(cost_value)
                if str1 == 'nan':
                    print('[nan ERROR at %d]' % index)
                    print(self.story[index])
                    print(sess.run(self.W, feed_dict={self.X: self.storyArr[index], self.Q: self.questionArr[index],
                                                 self.Y: self.ansArr[index]}))
                    exit()
                ## 정상적으로 학습되는지 확인하는 부분
                # print('[Inputs]\n{}\n[u]\n{}\n[Weights]\n{}'
                #       '\n[Outputs]\n{}\n[o]\n{}\n[hypothesis]\n{}\n'.format(check_str[0], check_str[1], check_str[2],
                #                                                             check_str[3], check_str[4], check_str[5]))
                # input('check')
                cost_mean = cost_mean + cost_value
                if Correct_value:
                    accuracy = accuracy + 1
            cost_mean = cost_mean / len(self.ansArr)
            accuracy = accuracy * 100. / len(self.ansArr)
            step = step + 1
            if step % 100 == 0 and step <= 400:
                learning_rate = learning_rate / 2
                print("%d loops learning_rate : %f" % (step, learning_rate))
            if previous_cost_mean < cost_mean:
                print('!! 더이상 학습이 안됩니다 !!')
                break

        W_value, Answer_value = sess.run([self.W, Answer],
                                         feed_dict={self.X: self.storyArr[test], self.Q: self.questionArr[test],
                                                    self.Y: self.ansArr[test]})

        print("[After {} Loops, Output] {}".format(step, Answer_value))
        print("Accuracy {}%".format(accuracy), "Cost", cost_mean, "W", W_value[0, 0])
        # Save
        W0, A0, B0, C0 = sess.run([self.W, self.A, self.B, self.C],
                                  feed_dict={self.X: self.storyArr[test], self.Q: self.questionArr[test],
                                             self.Y: self.ansArr[test]})
        f = open("save_data.txt", 'w')
        data = '[Constant]\ntest: {}, stddev: {}, ' \
               'learning_rate: {}, WORD: {}, MEMORY: {}\n'.format(test, stddev, learning_rate, self._WORD, self._MEMORY)
        f.write(data)
        data = 'W\n{}\nA\n{}\nB\n{}\nC\n{}\n'.format(W0, A0, B0, C0)
        f.write(data)
        f.close()
        ###################################
        # TEST
        ###################################
        flag = True
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
                        utils.refineWord(data[i])
                    user_story = data
                    break
                else:
                    temp = user.split()
                    utils.refineWord(temp)
                    user_story.append(temp)
            print('Question을 입력해주세요')
            user = input()
            user_question = user.split()
            utils.refineWord(user_question)

            user_storyArr = np.zeros([self._WORD, len(user_story)])
            user_questionArr = np.zeros([self._WORD, 1])

            try:
                for i, sent in enumerate(user_story):
                    s = np.zeros([self._WORD, 1])
                    for word in sent:
                        s[self.dictionary.index(word), 0] = 1
                    user_storyArr[:, [i]] = s
                for word in user_question:
                    user_questionArr[self.dictionary.index(word), 0] = 1
                # user_answerArr[dictionary.index(user_answer), 0] = 1
            except ValueError:
                print('사전에 없는 단어를 입력하였습니다')
                continue

            print("\n\n[STORY]")
            for story in user_story:
                print(" ".join(story))
            print("[Question] {}?".format(" ".join(user_question)))
            result = sess.run(tf.transpose(hypothesis),
                              feed_dict={self.X: user_storyArr, self.Q: user_questionArr})
            result = result * 100
            str1 = ""
            for _ in range(5):
                i = np.argmax(result)
                percent = int(result[0][i])
                result[0, i] = -1
                str1 = str1 + "%-12s" % self.dictionary[i]
                for i in range(int(percent / 2)):
                    str1 = str1 + '|'
                str1 = str1 + " {}%\n".format(percent)
            print("[Output]\n{}".format(str1))
