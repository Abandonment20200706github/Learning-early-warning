import pymysql
import pandas as pd

'''
定义连接
'''
conn = pymysql.connect(host='*****',
                           port=3306,
                           user='****',
                           passwd='*****',
                           db='******',
                           charset='utf8')

myConn = pymysql.connect(host='******',
                      port=3306,
                      user='****',
                      passwd='*****',
                      db='******',
                      charset='utf8')

'''
业务1：获取小节课程名称
stu_id
|- base knowledge -> sections
|- chapter knowledge ->  sections
|- follow knowledge -> sections 
|- parallel knowledge -> sections 
'''
def f0_Chapter2knowledgePoint(df_knowledgePoint):
    '''
    章对应的节
    :param df_knowledgePoint:
    :return:
    '''
    df = pd.DataFrame()
    def f(str):
        return str.split(".")[0]
    df['key'] = df_knowledgePoint['subchapter'].apply(f)
    df['subchapter'] = df_knowledgePoint['subchapter']
    def f(row):
        feature = pd.Series()
        feature['handler'] = list(row['key'])[0]
        feature['res'] = list(row['subchapter'])
        return feature
    df = df.groupby('key').apply(f)
    return df

def f1_knowledgePoint():
    '''
    获得具体知识点
    :return: df --- chapter [1.1, 1.2, 1.3, 1.4]……
    '''
    sql = "select * from subchapter_index"
    df_knowledgePoint = pd.read_sql(sql, conn)
    chapter2sections = f0_Chapter2knowledgePoint(df_knowledgePoint)
    list_spacial = df_knowledgePoint['subchapter'].tolist()
    def f(row):
        return chapter2sections[chapter2sections['handler']==str(row)]['res'].values[0]
    df_knowledgePoint['chapter'] = df_knowledgePoint['chapter'].apply(f)
    def f(row):
        if str(row)=="None":
            return []
        else:
            return str(row).split("+")
    df_knowledgePoint['follow_subchapter'] = df_knowledgePoint['follow_subchapter'].apply(f)
    df_knowledgePoint['parallel_subchapter'] = df_knowledgePoint['parallel_subchapter'].apply(f)
    return df_knowledgePoint

###############################################
# 清洗映射表
##############################################
sql = "select sectionid, lessonid from section"
df_sectionAndKnowledge = pd.read_sql(sql,conn)
def f(lessonid):
    if len(lessonid.split('.'))==3:
        return lessonid.split('.')[0]+"."+lessonid.split('.')[1]
    else:
        return lessonid
df_sectionAndKnowledge['lessonid'] = df_sectionAndKnowledge['lessonid'].apply(f)
def f1_knowledgePoint2SectionSolo(sections):
    '''
    单个知识点转化为具体章节视频
    :param section:单个知识点
    :return:
    '''
    if sections==[]:
        return []
    else:
        return df_sectionAndKnowledge[df_sectionAndKnowledge['lessonid'].isin(sections)]['sectionid'].tolist()

def f2_knowledgePoint2sections():
    '''
    知识点转化为具体章节视频
    :return:
    '''
    df_nowledgePoint = f1_knowledgePoint()
    df_nowledgePoint['key'] = df_nowledgePoint['subchapter']
    def f(subchapter):
        return [subchapter]

    df_nowledgePoint['subchapter'] = df_nowledgePoint['subchapter'].apply(f)
    df_nowledgePoint['subchapter'] = df_nowledgePoint['subchapter'].apply(f1_knowledgePoint2SectionSolo)
    df_nowledgePoint['chapter'] = df_nowledgePoint['chapter'].apply(f1_knowledgePoint2SectionSolo)
    df_nowledgePoint['follow_subchapter'] = df_nowledgePoint['follow_subchapter'].apply(f1_knowledgePoint2SectionSolo)
    df_nowledgePoint['parallel_subchapter'] = df_nowledgePoint['parallel_subchapter'].apply(f1_knowledgePoint2SectionSolo)
    return df_nowledgePoint










