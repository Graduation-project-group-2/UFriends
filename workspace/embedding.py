# library import
import pandas as pd
from numpy import dot
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# # 전처리한 데이터셋 불러오기
train_df = pd.read_csv('../data/train_df_preprocess.csv')

# 모델 사용해서 embedding 계산하고 컬럼으로 추가
model = SentenceTransformer('jhgan/ko-sroberta-multitask')
train_df['embedding'] = train_df.apply(lambda row: model.encode(row.user), axis=1)

# embedding 컬럼이 추가된 데이터셋 저장
train_df.to_csv('train_df_embedding.csv', index=False)