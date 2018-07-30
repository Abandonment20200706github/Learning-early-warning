import pandas as pd
import pymysql
import os
import Constants


'''
业务1: 最主要是学号 姓名 邮箱的串联 
'''
def f1():
    ##########################################
    # 获取姓名和邮箱的映射
    ##########################################
    df = pd.read_csv('rawData/pretest.csv')
    df['name'] = df['姓']+df['名']
    del df['姓']
    del df['名']
    df = df.drop([0,1,135])
    df = df[['name','Email地址']]

    df.loc[110]['name'] = "侯莎"
    df.loc[58]['name'] = "李宜"

    df.columns=['name','email']
    df_name2email = df
    ####################################################
    # 获取学号和姓名的映射
    ####################################################
    sql = "select id,name from student where id LIKE '17%'"
    df_id2name = pd.read_sql(sql, Constants.conn)
    df_id2name.columns = ['stu_id','name']
    ####################################################
    # 拼接学号 姓名 邮箱
    # 保存
    ####################################################
    df = pd.read_csv('rawData/preUser.csv')
    df = df.loc[df['stu_id'].drop_duplicates().index]
    df['stu_id'] = df[['stu_id']].astype(str)
    # print(pd.merge(df_id2name,df, on='stu_id'))
    df = pd.merge(df_name2email,pd.merge(df_id2name,df, on='stu_id'),on='name')
    df =df.loc[df['name'].drop_duplicates().index]
    df.to_csv('result_data/preUserFinished.csv', index=False, encoding='utf8')
    df = pd.read_csv('result_data/preUserFinished.csv')
    print(df)

'''
业务2: 
'''
prefixPath = "testQuestions/stage0/testQuestion/"
allFile = os.listdir('testQuestions/stage0/testQuestion')
#########################################################
# 单个文件的操作
# csv没哟统一的编码格式
# 其中涉及数据清洗的策略函数
#########################################################
def f1_judgeScore(tmp_, score_):
    if tmp_ == float(0):
        return 0
    elif tmp_ < score_:
        return 0.5
    else:
        return 1

def f2_clean(str_,score_):
    if str_=='-' or str_=='-\n':
        return -1
    elif str_[-1]=="n":
        tmp_ = float(str_[:-2])
        return f1_judgeScore(tmp_, score_)
    else:
        tmp_ = float(str_)
        return f1_judgeScore(tmp_, score_)

def f2_timeFormate(str_):
    str_ = str_.split("日")[0]
    return str_[1:5]+"-"+str_[6:8]+"-"+str_[9:]

def f3_solo(fPath):
    with open(fPath, "r", encoding='utf-8', errors='ignore') as fr:
        passFlag = False
        list_res = []
        len_ = 0
        for line in fr:
            if passFlag==False:
                passFlag = True
            else:
                list_ = line.split(",")
                key = list_[4]
                timeStamp = f2_timeFormate(list_[6])
                len_ = len(list_)-10
                score_ = 10/len_
                try:
                    for i in range(10,len(list_)):
                        list_res.append([key,f2_clean(list_[i],score_),timeStamp])
                except:
                    print("------->",fPath)
        return pd.DataFrame(list_res[:-len_],columns=['email','score','timestamp'])

def f3_expalinName(str_):
    list_str_ = str_[9:].split(" ")
    chapter = list_str_[0].split(".")
    chapter = [int(i) for i in chapter]
    name = list_str_[1].split(".")[0].split("-")[0]
    if len(chapter)==2:
        chapter = chapter+[0]
    return chapter+[name]

df_score = pd.DataFrame(columns= ['email','score','chapter','chapter_son','chapter_grandson','description','timestamp'])
for fName in allFile:
    information = f3_expalinName(fName)
    fPath = 'testQuestions/stage0/testQuestion/'+ fName
    df = f3_solo(fPath)
    df['chapter'] = information[0]
    df['chapter_son'] = information[1]
    df['chapter_grandson'] = information[2]
    df['description'] = information[3]
    df_score = pd.concat([df_score,df])

# 和学号连接上
sql = "select stu_id, email from stu_base"
df_email2stuid = pd.read_sql(sql, Constants.myConn)
df_toDB = pd.merge(df_email2stuid, df_score, how='inner', on='email')

####################################################
# 写入数据库
####################################################
def f_toDB(df_toDB):
    from sqlalchemy import create_engine
    user = 'root'
    password = "Thisisit..0316"
    host = "localhost"
    db = "stu_analysis"
    engine = create_engine(str( r"mysql+mysqldb://%s:" + '%s' + "@%s/%s?charset=utf8") % (user, password, host, db))
    df_toDB.to_sql('test_question', engine, index= False)

##################################################
# 筛选统计
# f_Statistics 整体特征：比如总记录数等
# f1_stuSolo 单体特征：比如正确率等
##################################################
def f_Statistics(df_toDB):
    print("total record:", len(df_toDB))
    print("total stu:", len(df_toDB['stu_id'].drop_duplicates()), " ----->ave:", len(df_toDB)/len(df_toDB['stu_id'].drop_duplicates()))
    print("chapter:", len(df_toDB['chapter'].drop_duplicates()), " ----->ave:", len(df_toDB)/len(df_toDB['chapter'].drop_duplicates()))
    print("chapter_grandson:", len(df_toDB['description'].drop_duplicates())+3, " ----->ave:", len(df_toDB)/(len(df_toDB['description'].drop_duplicates())+3))
    print("density: <", 100*len(df_toDB)/((len(df_toDB['description'].drop_duplicates())+3)*95),"%")

    print("total date:", len(df_toDB['timestamp'].drop_duplicates()), " ----->ave:", len(df_toDB)/len(df_toDB['timestamp'].drop_duplicates()))
    print("April:", len(df_toDB[df_toDB['timestamp']<'2018-04']))
    print("May:", len(df_toDB[df_toDB['timestamp']<'2018-05'])-len(df_toDB[df_toDB['timestamp']<'2018-04']))
    print("June:", len(df_toDB[df_toDB['timestamp']<'2018-06'])-len(df_toDB[df_toDB['timestamp']<'2018-05']))
    print("July:", len(df_toDB)-len(df_toDB[df_toDB['timestamp']<'2018-06']))

def f1_stuSolo(row):
    feature = pd.Series()
    feature['stu_id'] = list(row['stu_id'])[0]
    feature['count_record'] = len(row)
    feature['count_chapter'] = len(row['chapter'].drop_duplicates())
    feature['count_chapterSon'] = len((row['chapter']+row['chapter_son']).drop_duplicates())
    feature['count_chapterGrandson'] = len((row['chapter'] + row['chapter_son']+row['chapter_grandson']).drop_duplicates())
    feature['correct'] = row['score'].sum()/len(row)
    feature['test_date_delta'] = row['testQ_date_delta'].sum()/len(row)
    return feature

def f_stu(df_toDB):
    def f(row):
        feature = pd.Series()
        feature['testQ_date_delta'] = (pd.Timestamp("2018-6-22") - pd.Timestamp(str(row))).days
        return feature
    df_toDB['testQ_date_delta'] = df_toDB['timestamp'].apply(f)
    # df_toDB = df_toDB[df_toDB['testQ_date_delta'] > 56]
    df_stu = df_toDB.groupby('stu_id').apply(f1_stuSolo)
    return df_stu


if __name__ == "__main__":
    df_toDB = df_toDB[df_toDB['score']!=-1]
    df = f_stu(df_toDB)
    df = pd.merge(df_email2stuid, df, how='left', on='stu_id')
    df = df.fillna(-1)
    del df['email']
    print(df)
    if df.empty==False:
        df.to_csv("testQuestions/stage1/testQuestion.csv", index=False)
