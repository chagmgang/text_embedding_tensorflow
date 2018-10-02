import tensorflow as tf
import glob
import collections
from itertools import chain
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def build_dataset(sentences, vocabulary_size):
    words = []
    for sent in sentences:
        for splited_sent in sent.split(' '):
            words.append(splited_sent)
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)

    unk_count = 0
    sent_data = []
    for sentence in sentences:
        data = []
        for word  in sentence.split():
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0
                unk_count = unk_count + 1
            data.append(index)
        sent_data.append(data)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    # sent_data = 단어(key)에 대한 value값으로 구성된 하나의 리스트
    # count = 각 단어들이 총 문서에 몇 번씩 등장하였는지
    # dictionary = {단어: value}
    # reverse_dictionary = {value: 단어}
    return sent_data, count, dictionary, reverse_dictionary


def generate_batch(batch_size, instances, labels, doc, context):
    data_idx = 0    
    if data_idx+batch_size<instances:
        batch_labels = labels[data_idx:data_idx+batch_size]
        batch_doc_data = doc[data_idx:data_idx+batch_size]
        batch_word_data = context[data_idx:data_idx+batch_size]
        data_idx += batch_size
    else:
        overlay = batch_size - (instances-data_idx)
        batch_labels = np.vstack([labels[data_idx:instances],labels[:overlay]])
        batch_doc_data = np.vstack([doc[data_idx:instances],doc[:overlay]])
        batch_word_data = np.vstack([context[data_idx:instances],context[:overlay]])
        data_idx = overlay
    batch_word_data = np.reshape(batch_word_data,(-1,1))

    return batch_labels, batch_word_data, batch_doc_data