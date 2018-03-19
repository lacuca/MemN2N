import tensorflow as tf
import numpy as np


class cData:
    def __init__(self, word, memory):
        self._WORD = word  # 사전의 크기 (사전이 저장할 수 있는 최대 WORD)
        self._MEMORY = memory  # Memory Vector 의 크기

    def read(self, file, test=False):
        f = open(file, 'r')
        read_data = f.readlines()
        f.close()

        story = []
        question = []
        ans = []

        story.append([])
        for l in read_data:
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
        for i in range(self._WORD):
            dictionary.append('')
        index = 0

        def isNumber(s):
            try:
                float(s)
                return True
            except ValueError:
                return False

        for l in read_data:
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

        sentence = np.zeros([self._WORD, 1])

        storyArr = []
        questionArr = []
        ansArr = []

        for sents in story:
            storyArr.append(np.zeros([self._WORD, len(sents)]))
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

        # 행렬로 잘 저장되는지 테스트 출력
        if test:
            print("\n")
            for i in range(len(ans)):
                print('{}\n{}\n'.format(story[i][0], story[i][1]),
                      '{}\n{}\n'.format(storyArr[i][0].T, storyArr[i][1].T))
            for i in range(len(ans)):
                print(question[i], '\n', questionArr[i].T)
            for i in range(len(ans)):
                print(ans[i], '\n', ansArr[i].T)

        return story, question, ans, storyArr, questionArr, ansArr, dictionary
