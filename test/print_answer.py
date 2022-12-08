# library import
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('../data/train_df_embedding_small.csv')

print("user: ")
text = input()

# 학습한 모델 사용해서 user_input에 대한 embedding 생성
model = SentenceTransformer('jhgan/ko-sroberta-multitask')
embedding = model.encode(text)

# embedding 컬럼을 리스트로 변환 후 similarity 계산
sim = []
for i in range(len(df)):
    temp = df.loc[i]['embedding']
    temp = temp.strip("[""]")
    abc = temp.split()

    f = lambda x: cosine_similarity([embedding], [x])
    sim.append(f(abc).squeeze())

df['similarity'] = sim
df = df.astype({'similarity': 'float'})

answer = df.loc[df['similarity'].idxmax()]

print('구분: ', answer['sentiment'])
print('유사한 질문: ', answer['user'])
print('챗봇 답변: ', answer['chatbot'])
print('유사도: ', answer['similarity'])  # 1에 가까울수록 유사
