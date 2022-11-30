import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

train_df = pd.read_csv('train_df_preprocess.csv')

model = SentenceTransformer('jhgan/ko-sroberta-multitask')

# user를 embedding 하여 embedding column 생성 & csv 파일로 저장
train_df['embedding'] = train_df.apply(lambda row: model.encode(row.user), axis = 1)
train_df.to_csv('train_df_embedding.csv', index=False)

