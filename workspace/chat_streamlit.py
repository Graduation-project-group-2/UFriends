import streamlit as st
from streamlit_chat import message
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


@st.cache(allow_output_mutation=True)
def cached_model():
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model


@st.cache(allow_output_mutation=True)
def get_dataset():
    df = pd.read_csv('../data/train_df_embedding.csv')
    return df


model = cached_model()
df = get_dataset()

st.header('U-Friends 심리상담 챗봇')

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

with st.form('form', clear_on_submit=True):
    user_input = st.text_input('사용자: ', '')
    submitted = st.form_submit_button('전송')

if submitted and user_input:
    embedding = model.encode(user_input)

    sim = []
    for i in range(len(df)):
        temp = df.loc[i]['embedding']
        temp = temp.strip("[""]")
        abc = temp.split()

        f = lambda x: cosine_similarity([embedding], [x])
        sim.append(f(abc).squeeze())

    df['distance'] = sim
    df = df.astype({'distance': 'float'})

    answer = df.loc[df['distance'].idxmax()]

    st.session_state.past.append(user_input)
    st.session_state.generated.append(answer['chatbot'])

for i in range(len(st.session_state['past'])):
    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    if len(st.session_state['generated']) > i:
        message(st.session_state['generated'][i], key=str(i) + '_bot')