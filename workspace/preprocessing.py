import pandas as pd
import numpy as np

train_df = pd.read_excel('/Users/gimmingyeong/Documents/3-졸업작품/3-2/U-Friends/감성대화_데이터셋/Training/감성대화말뭉치(최종데이터)_Training/감성대화말뭉치(최종데이터)_Training.xlsx')
test_df = pd.read_excel('/Users/gimmingyeong/Documents/3-졸업작품/3-2/U-Friends/감성대화_데이터셋/Validation/감성대화말뭉치(최종데이터)_Validation/감성대화말뭉치(최종데이터)_Validation.xlsx')

train_df.set_index("번호", inplace=True)
test_df.set_index("번호", inplace=True)

# 필요없는 column drop
train_df = train_df.drop(["연령", "성별", "상황키워드", "신체질환", "감정_소분류",
                           "사람문장2", "시스템응답2", "사람문장3",
                          "시스템응답3", "사람문장2", "시스템응답4",  "사람문장4"], axis=1)
test_df = test_df.drop(["연령", "성별", "상황키워드", "신체질환", "감정_소분류",
                           "사람문장2", "시스템응답2", "사람문장3",
                          "시스템응답3", "사람문장2", "시스템응답4",  "사람문장4"], axis=1)

# 감정_대분류 : '불안 ', '기쁨 ' -> '불안', '기쁨'
train_df.loc[train_df['감정_대분류'] == '기쁨 ', '감정_대분류'] = '기쁨'
train_df.loc[train_df['감정_대분류'] == '불안 ', '감정_대분류'] = '불안'

test_df.loc[test_df['감정_대분류'] == '기쁨 ', '감정_대분류'] = '기쁨'
test_df.loc[test_df['감정_대분류'] == '불안 ', '감정_대분류'] = '불안'

# document 열과 label 열의 중복을 제외한 값의 개수
train_df['사람문장1'].nunique(), train_df['감정_대분류'].nunique()
test_df['사람문장1'].nunique(), test_df['감정_대분류'].nunique()

# 총 40,879개의 샘플이 존재하는데 '사람문장1'열에서 중복을 제거한 샘플의 개수가 39,415개라는 것은 약 6,000개의 중복 샘플이 존재한다는 의미
# '감정_대분류'열은 6이 출력됨(기쁨, 당황, 분노, 불안, 상처, 슬픔). 중복 샘플을 제거
# document 열의 중복 제거
train_df.drop_duplicates(subset=['사람문장1'], inplace=True)
test_df.drop_duplicates(subset = ['사람문장1'], inplace=True) # 사람문장1 열에서 중복인 내용이 있다면 중복 제거

train_df.drop_duplicates(subset=['시스템응답1'], inplace=True)
test_df.drop_duplicates(subset = ['시스템응답1'], inplace=True)

print('총 샘플의 수 :',len(train_df))
print('총 샘플의 수 :',len(test_df))

print(train_df.groupby('감정_대분류').size().reset_index(name = 'count'))
print(test_df.groupby('감정_대분류').size().reset_index(name = 'count'))

print(len(train_df))
print(len(test_df))

# 한글과 공백을 제외하고 모두 제거
train_df['사람문장1'] = train_df['사람문장1'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
test_df['사람문장1'] = test_df['사람문장1'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","") # 정규 표현식 수행
train_df['사람문장1'] = train_df['사람문장1'].str.replace('^ +', "") # white space 데이터를 empty value로 변경
test_df['사람문장1'] = test_df['사람문장1'].str.replace('^ +', "") # 공백은 empty 값으로 변경

train_df['시스템응답1'] = train_df['시스템응답1'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
test_df['시스템응답1'] = test_df['시스템응답1'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","") # 정규 표현식 수행
train_df['시스템응답1'] = train_df['시스템응답1'].str.replace('^ +', "") # white space 데이터를 empty value로 변경
test_df['시스템응답1'] = test_df['시스템응답1'].str.replace('^ +', "") # 공백은 empty 값으로 변경

# 한글이 없는 리뷰였다면 더 이상 아무런 값도 없는 빈(empty) 값이 되었을 것
# 그런 샘플이 있나 확인
train_df['사람문장1'].replace('', np.nan, inplace=True)
test_df['사람문장1'].replace('', np.nan, inplace=True) # 공백은 Null 값으로 변경

train_df['시스템응답1'].replace('', np.nan, inplace=True)
test_df['시스템응답1'].replace('', np.nan, inplace=True)

print(train_df.isnull().sum())
print(test_df.isnull().sum())

# 감정_대분류 labeling
train_df.loc[(train_df['감정_대분류']=="분노"), '감정_대분류'] = 0
train_df.loc[(train_df['감정_대분류']=="슬픔"), '감정_대분류'] = 1
train_df.loc[(train_df['감정_대분류']=="불안"), '감정_대분류'] = 2
train_df.loc[(train_df['감정_대분류']=="상처"), '감정_대분류'] = 3
train_df.loc[(train_df['감정_대분류']=="당황"), '감정_대분류'] = 4
train_df.loc[(train_df['감정_대분류']=="기쁨"), '감정_대분류'] = 5

test_df.loc[(test_df['감정_대분류']=="분노"), '감정_대분류'] = 0
test_df.loc[(test_df['감정_대분류']=="슬픔"), '감정_대분류'] = 1
test_df.loc[(test_df['감정_대분류']=="불안"), '감정_대분류'] = 2
test_df.loc[(test_df['감정_대분류']=="상처"), '감정_대분류'] = 3
test_df.loc[(test_df['감정_대분류']=="당황"), '감정_대분류'] = 4
test_df.loc[(test_df['감정_대분류']=="기쁨"), '감정_대분류'] = 5

train_df.rename(columns={'감정_대분류':'sentiment', '사람문장1':'user', '시스템응답1':'chatbot'}, inplace=True)
test_df.rename(columns={'감정_대분류':'sentiment', '사람문장1':'user', '시스템응답1':'chatbot'}, inplace=True)

# preprocessing한 dataframe을 csv 파일로 저장
train_df.to_csv('train_df_preprocess.csv', index=False)
test_df.to_csv('test_df_preprocess.csv', index=False)









