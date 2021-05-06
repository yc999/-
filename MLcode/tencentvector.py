# 共 8824330 词
from gensim.models import KeyedVectors
import numpy as np
from keras.layers import Embedding

EMBEDDING_DIM = 200  #词向量长度
EMBEDDING_length = 8824330
MAX_SEQUENCE_LENGTH = 10


filepath = 'Tencent_AILab_ChineseEmbedding.txt'
f = open(filepath)


embeddings_index = {}
embedding_matrix = np.zeros((EMBEDDING_length + 1, EMBEDDING_DIM))


i = 1
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = i
    embedding_matrix[i] = coefs
    print(i)
    print(word)
    print(coefs)
    i = i+1
    if i>=5:
        break

f.close()

# embedding_matrix = np.zeros((len(embeddings_index) + 1, EMBEDDING_DIM))
# for word, i in word_index.items():
#     print(i)
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None:
#         # words not found in embedding index will be all-zeros.
#         embedding_matrix[i] = embedding_vector




wv_from_file = KeyedVectors.load_word2vec_format(filepath,binary=False)

print(len(wv_from_file.vocab.keys()))
word_index = len(wv_from_file.vocab.keys())
embeddings_index = {}
# f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
# for line in f:
#     values = line.split()
#     word = values[0]
#     coefs = np.asarray(values[1:], dtype='float32')
#     embeddings_index[word] = coefs
# f.close()


embedding_layer = Embedding(word_index + 2,
                            EMBEDDING_DIM,
                            weights=[wv_from_file],
                            # input_length=MAX_SEQUENCE_LENGTH,
                            mask_zero = True,
                            trainable=False)
print(embedding_layer)











