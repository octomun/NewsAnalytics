
#import JPype1
from konlpy.tag import Hannanum
from konlpy.utils import pprint
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf
from datetime import datetime, timedelta
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import pandas as pd
from sklearn.model_selection import train_test_split
import requests
import urllib.request
import urllib.parse
from urllib.request import urlopen
from bs4 import BeautifulSoup
# ## 주가 읽기

def read_file():
    stock = pd.read_csv("sk하이닉스.csv")
    stock['date'] = pd.to_datetime(stock['date'], format ='%Y-%m-%d')
    for i in range(len(stock)):
        stock['per'][i]=(float(stock['per'][i].replace("%",""))/100)

    # # 뉴스 url읽기
    url = pd.read_csv("crawling_news2.txt", header = None, names = ["date","href"])
    url['per']= "A"
    return url, stock

url, stock = read_file()
print("read_file")

#전처리
def Pretreatment(url,stock):
    for i in range(len(url)):
        url['date'][i] = url['date'][i][:8]
        url['date'][i] = pd.to_datetime(url['date'][i], format ='%y.%m.%d')
        url['date'][i] = pd.Timestamp(url['date'][i]).date()
        while url['date'][i] not in list(stock['date']):
            url['date'][i]=url['date'][i]+timedelta(days=1)
        for j in range(len(stock)):
            if url['date'][i] == stock['date'][j]:
                url['per'][i] = stock['per'][j-1] #다음날 주가

    te4 = pd.DataFrame(url)
    te4 =te4.drop_duplicates(["date"])
    q1 = te4["per"].quantile(.25)
    q3 = te4["per"].quantile(.75)
    for i in range(len(url)):
        if url['per'][i] > q3:
            url['per'][i] = 0
        elif url['per'][i] <= q3 and url['per'][i]>0:
            url['per'][i] = 1
        elif url['per'][i] <= 0 and url['per'][i]>q1:
            url['per'][i] = 2
        else:
            url['per'][i] = 3
    return url

    # ### 퍼센트 원 핫 인코딩

    # ##### 원핫 인코딩 y값 제거해야 한다 함

    # from tensorflow.keras.utils import to_categorical
    # #float(te4["per"][1])
    # url['per'] = list(to_categorical(url['per']))
    #

url = Pretreatment(url,stock)
print("Pretreatment")

#test와 train데이터 분리

# # 전처리 및 토큰화
def morpheme(data):
    news_group=[]
    j=0
    for i in range(len(data)):
        webpage = requests.get(data['href'][i])
        soup = BeautifulSoup(webpage.content, "html.parser")
        soup = soup.select_one('#newsViewArea').get_text()
        soup = re.sub('[^ㄱ-ㅎㅏ-ㅣ가-힣A-Za-z ]', '', soup)
        if i == 0 :
            news_group_date = data['date'][i]
            news_group_content = soup
            news_group = pd.DataFrame([news_group_date,news_group_content,data['per'][i]],['date','content','per']).transpose()
            continue
        if news_group['date'][j] == data['date'][i]:
            news_group['content'][j] = news_group['content'][j] + soup
        else :
            j = j + 1
            a = {'date':data['date'][i],'content':soup,'per':data['per'][i]}
            news_group = news_group.append(a,ignore_index=True)
    return news_group


all_data=morpheme(url)
print("morpheme")

def text_except_all(data):
    tokken = []
    hannanum=Hannanum()
    for i in range(len(data)):
        tokken.append(hannanum.pos(data['content'][i]))

    # ### 토큰 중 가장 긴 토큰을 기준으로 반복 및 형태소 중 명사 동사 선택

    lenA = []
    for i in range(len(tokken)):
        lenA.append(len(tokken[i]))
    max(lenA)

    all_tokken=[]
    Stopword = pd.read_csv("한국어불용어100.txt", header=None, names=['text','x','num'],delimiter = '\t')
    for i in range(len(tokken)):
        for j in range(lenA[i]):
            if tokken[i][j][1] == 'N' and tokken[i][j][0] not in Stopword['text'].values:
                all_tokken.append(tokken[i][j][0])
    return all_tokken

all_tokken = text_except_all(all_data)
print("text_except")

from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_tokken)

vocab_size = 1000  # 상위 500 단어만 사용
tokenizer = Tokenizer(num_words = vocab_size + 1)
tokenizer.fit_on_texts(all_tokken)

# print(tokenizer.word_index) #인덱스가 어떻게 부여됬는지(입력된 단어 순서)
# print(tokenizer.word_counts) #상위 몇개 단어를 했을 때 어떻게 부여됬는지(입력된 단어 순서)

def text_size(num):
    threshold = num
    total_cnt = len(tokenizer.word_index) # 단어의 수
    rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
    total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
    rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

    # 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
    for key, value in tokenizer.word_counts.items():
        total_freq = total_freq + value

        # 단어의 등장 빈도수가 threshold보다 작으면
        if(value < threshold):
            rare_cnt = rare_cnt + 1
            rare_freq = rare_freq + value

    print('단어 집합(vocabulary)의 크기 :',total_cnt)
    print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
    print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
    print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)

    #단어수가 2개인 단어의 빈도가 6.1%라 유의미한 영향을 줄 수 있어 제외하지 않는다

    # 전체 단어 개수 중 빈도수 2이하인 단어는 제거.
    # 0번 패딩 토큰을 고려하여 + 1
    vocab_size = total_cnt - rare_cnt + 1
    print('단어 집합의 크기 :',vocab_size)


    # ## 앞의 형태소분석을 붙여씀
    # ### 불필요하게 주가를 넣는 부분이 있고 href에서 본문을 따오는 부분 함수화 고려

text_size(2)

def train_test(data):
    train, test = train_test_split(data, test_size= 0.5, random_state=1234)

    #인덱스 초기화
    train = train.reset_index()
    test = test.reset_index()
    return train, test

train, test = train_test(all_data)
print("train_test")


# train_data = morpheme(train)
# test_data = morpheme(test)

# train_data['content'] = text_except(train_data['content'])
# test_data['content'] = text_except(test_data['content'])

X_train = train['content']
X_test = test['content']
# X_train = tokenizer.texts_to_sequences(train_data['content'])
# X_test = tokenizer.texts_to_sequences(test_data['content'])

y_train = train['per']
y_train = pd.DataFrame(y_train)
y_test = test['per']
y_test = pd.DataFrame(y_test)


drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]
# 빈 샘플들을 제거
X_train = np.delete(X_train, drop_train, axis=0)
y_train = np.delete(y_train, drop_train, axis=0)
print(len(X_train))
print(len(y_train))



# ## 패딩

def below_threshold_len(max_len, nested_list):
  cnt = 0
  for s in nested_list:
    if(len(s) <= max_len):
        cnt = cnt + 1
  print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (cnt / len(nested_list))*100))


print('뉴스의 최대 길이 :',max(len(l) for l in X_train))
print('뉴스의 평균 길이 :',sum(map(len, X_train))/len(X_train))
plt.hist([len(s) for s in X_train], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()
max_len = 270
below_threshold_len(max_len, X_train)


X_train = pad_sequences(X_train, maxlen = max_len)
X_test = pad_sequences(X_test, maxlen = max_len)



X_train2 = pd.DataFrame(X_train)
X_train2=X_train2.astype('float64')
X_test2 = pd.DataFrame(X_test)
X_test2=X_test2.astype('float64')



y_train=y_train.astype('float64')
y_test=y_test.astype('float64')



X_train2 = np.array(X_train2).reshape(X_train2.shape[0], X_train2.shape[1], 1)
y_train = np.array(y_train).reshape(y_train.shape[0], y_train.shape[1], 1)
X_test2 = np.array(X_test2).reshape(X_test2.shape[0], X_test2.shape[1], 1)
y_test = np.array(y_test).reshape(y_test.shape[0], y_test.shape[1], 1)


# #### loss = sparse_categorical_crossentropy은 y값을 원핫인코딩하지 않는다
# y_train = to_categorical(y_train, 4)


from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



model = Sequential()
#model.add(Embedding(vocab_size, 100)) #모델에 입력크기를 고정된 크기고 제한
#model.add(Dense(2, activation='softmax'))
model.add(LSTM(128, input_shape = (270,1)))#,return_sequences=True, input_shape = (300,1)
model.add(Dense(64, activation='relu'))
model.add(Dense(4, activation='softmax'))
# model.add(Dense(1, activation='sigmoid'))


es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc','categorical_accuracy'])
history = model.fit(X_train2, y_train, epochs=100, callbacks=[es, mc], batch_size=60, validation_split=0.2)
#validation_split 전체데이터(train)중 얼마를 학습할 것이냐
#batch_size 계산 후 가중치를 넘길 계산 단위

loaded_model = load_model('best_model.h5')
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test2, y_test)[1]))

# 학습 결과 그래프 그리기
'''
fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
acc_ax.plot(hist.history['val_accuracy'], 'b', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')
plt.show()
'''
# # 끝
