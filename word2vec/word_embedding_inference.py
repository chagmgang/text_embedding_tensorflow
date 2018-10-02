import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.font_manager
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

# matplot 에서 한글을 표시하기 위한 설정
rc('font', family='AppleGothic')

plt.rcParams['axes.unicode_minus'] = False


word_list = ['영화', '동물','생선','책','음악','우유','여자친구','싫다','게임','만화','좋다','나','눈','애니','고양이','강아지']

#########
# 옵션 설정
######
# 학습을 반복할 횟수
training_epoch = 300
# 학습률
learning_rate = 0.1
# 한 번에 학습할 데이터의 크기
batch_size = 48
# 단어 벡터를 구성할 임베딩 차원의 크기
# 이 예제에서는 x, y 그래프로 표현하기 쉽게 2 개의 값만 출력하도록 합니다.
embedding_size = 2
# word2vec 모델을 학습시키기 위한 nce_loss 함수에서 사용하기 위한 샘플링 크기
# batch_size 보다 작아야 합니다.
num_sampled = 15
# 총 단어 갯수
voc_size = len(word_list)
print(voc_size)

embeddings = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0), name='embeddings')

sess = tf.Session()

saver = tf.train.Saver()
saver.restore(sess, "./tmp/model.ckpt")

input_feature = []
output_feature = []

trained_embeddings = embeddings.eval(session=sess)

#########
# 임베딩된 Word2Vec 결과 확인
# 결과는 해당 단어들이 얼마나 다른 단어와 인접해 있는지를 보여줍니다.
######

for i, label in enumerate(word_list):
    x, y = trained_embeddings[i]
    print(i, label, x, y)
    plt.scatter(x, y)
    plt.annotate(label, xy=(x, y), xytext=(5, 2),
                 textcoords='offset points', ha='right', va='bottom')

plt.show()