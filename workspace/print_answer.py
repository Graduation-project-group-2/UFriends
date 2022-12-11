# library import
import pandas as pd
import numpy as np
# from numpy import dot
# from numpy.linalg import norm
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def print_answer():
    df = pd.read_csv('../data/train_df_embedding.csv')

    print("user: ", end="")
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


# 피클로 저장 안 하는 게 나을 듯
# with open("print_answer.pkl", "wb") as f:
#     pickle.dump(print_answer(), f)

# function call
print_answer()

