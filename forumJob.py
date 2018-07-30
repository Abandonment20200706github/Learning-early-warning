import requests
import json
import pandas as pd
import threading
import Constants

# 因为这部分逻辑相对相见，同时很适合多线程
# 于是小试了下多线程
df_forum = pd.read_excel("forum/stage0/forum.xlsx")
list_all = []
def f(row):
    comment = row[1]
    dict_ = {
        "query": str(comment)
    }
    postData = json.dumps(dict_)
    response = requests.post('http://39.107.98.156:5353/v1/emotion/query',data=postData)
    response = response.json()
    list_ = [0, 0, 0]
    if response['polar'] == 0:
        list_ = [0,1,0]
    elif response['polar'] == 1:
        list_ = [0, 0, 1]
    elif response['polar'] == -1:
        list_ = [1, 0, 0]
    row = row+list_
    list_all.append(row)

threads = []
len_ = len(df_forum)
for i in range(0, len(df_forum)):
    list_ = [df_forum.iloc[i]['name'],df_forum.iloc[i]['description']]
    t = threading.Thread(target=f, args=(list_,))
    threads.append(t)

for i in range(0,len_):
    threads[i].start()
for i in range(0, len_):
    threads[i].join()

df_forum = pd.DataFrame(list_all, columns=['name']+Constants.list_forum)
del df_forum["description"]
def f(row):
    feature = pd.Series()
    feature['name'] = list(row['name'])[0]
    len_ = len(row)
    feature['comment_negative'] = row['negative'].sum()/len_
    feature['comment_neutral'] = row['neutral'].sum()/len_
    feature['comment_positive'] = row['positive'].sum()/len_
    feature['comment_num_all'] = len_
    return feature
df_forum = df_forum.groupby('name').apply(f)

df_base = pd.read_csv("baseInformation/stage0/stu_base.csv")
df_ = pd.merge(df_base, df_forum, how='left', on='name')
df_ = df_.fillna(-1)
df_forum = df_[['stu_id'] + Constants.list_comment]
df_forum.to_csv("forum/stage1/forum.csv", index=False)
df_.to_csv("baseInformation/stage1/stu_base.csv", index=False)



