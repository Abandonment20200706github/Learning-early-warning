import pandas as pd
import Constants


def f_extractObjective(Q_type, col_key, max):
    '''
    合并同一知识点可能对应多题的情况
    原始知识点可能是三级的，统一为两级
    :param Q_type: 题型
    :param col_key: 知识点列表
    :return:
    '''
    df = pd.read_excel("examination/stage0/scoreMaterial"+Q_type+".xlsx")
    df_stuId = df['stu_id']
    del df['stu_id']
    def f(x):
        print(x, max)
        if pd.isnull(x)==True:
            return 1
        elif x==max:
            return 1
        elif str(x)<str(max) and str(x)>str('0.0'):
            return 0.5
        else:
            return 0
    df = df.applymap(f)
    print(df)
    df['stu_id'] = df_stuId
    df_stuId = pd.read_sql("select stu_id from stu_base", Constants.myConn).apply(pd.to_numeric)
    df = pd.merge(df_stuId, df)
    ##################
    # 合并相同题目
    ##################
    index_stuId = df['stu_id']
    del df['stu_id']

    df = df.T
    df['key'] = col_key
    def f(row):
        key = list(row['key'])[0]
        del row['key']
        feature = row.sum()/len(row)
        feature['key'] = key
        return feature
    df = df.groupby('key').apply(f)
    col_key = df['key']
    del df['key']
    df = df.T
    df.columns = col_key
    def f(x):
        if x==1:
            return 1
        elif x==0:
            return 0
        else:
            return 0.5
    df = df.applymap(f)
    df['stu_id'] = index_stuId
    df.to_csv('examination/stage1/knowledge'+Q_type+'Finished.csv', index=False)
    print(pd.read_csv('examination/stage1/knowledge'+Q_type+'Finished.csv').columns)

if __name__ == "__main__":
    f_extractObjective("Objective", Constants.col_keyObjective, 1)
    f_extractObjective("Brief", Constants.col_keyBrief, 5)
    f_extractObjective("Comprehensive", Constants.col_keyComprehensive, 3)