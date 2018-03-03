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
    storyArr.append([])
    for sent in sents:
        s = sentence.copy()
        for word in sent:
            s[dictionary.index(word), 0] = 1
        storyArr[-1].append(s)

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

