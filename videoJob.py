import pandas as pd
import os
import Constants
import myUtils

'''
该组特征的提取是一个三级结构：
整体时间阶段（期中、期末）
    |- 单时段特征（基于基础特征）
        |- 基础特征
'''

'''
业务1：基本特征的提取
比如：无效---会话次数  有效---观看天数、观看时长(min)、交互次数、暂停|快进|静音次数、会话次数
'''
def get_base_feature(key, userid, startTime, endTime, date_delta,conn):
    '''
    提取基础特征
    :param key: 小节列表
    :param userid: 用户标识
    :param conn: 数据库连接
    :return: 特征字典
    '''
    if key == []:
        return Constants.dict_feature
    elif len(key)==1:
        sql = "select userid, behave, sectionid,happentime  from behavior where userid = "\
              +userid+" and sectionid in " + str(tuple(key))[:-2]+")" \
              + 'and happentime between \' ' + startTime + "\' and \'" + endTime + "\'"
        df = pd.read_sql(sql, conn)
    else:
        sql = "select userid, behave, sectionid,happentime  from behavior where userid = "\
              +userid+" and sectionid in " + str(tuple(key)) \
              + 'and happentime between \' '+ startTime + "\' and \'" + endTime + "\'"
        df = pd.read_sql(sql, conn)
        # 因为df的behave 不是从1开始的
    for index in df.index:
        if df.loc[index]['behave'] == 1:
            break
        else:
            df = df.drop([index])
    df = df.reset_index(drop=True).reindex()
    if df['happentime'].count() > 0:
        # 添加会话字段
        list_session = []
        df_m = df[df['behave']==1]
        index_ = df_m.index
        len_ = len(df)
        count_i = index_[0]
        for i in index_[1:]:
            for j in range(count_i, i):
                list_session = list_session + [count_i]
            count_i = i
        for j in range(count_i, len_):
            list_session = list_session + [count_i]
        df['session'] = list_session
        def f(row):
            feature = pd.Series()
            row = row.reset_index(drop=True).reindex()
            feature['stu_id'] = list(row['userid'])[0]
            feature['session'] = list(row['session'])[0]
            len_watch = (row['happentime'][len(row['happentime'])-1] - row['happentime'][0]).seconds
            if  len_watch< 6 * 3600 and len_watch>2 * 60:
                feature['section'] = list(row['sectionid'])[0]
                feature['len_watch'] = len_watch
                feature['len_session'] = len(row['happentime'])
                feature['count_pause'] = len(row[row['behave'] == 3])
                feature['count_speed'] = len(row[row['behave'] == 5])
                feature['count_mute'] = len(row[row['behave'] == 12])
                feature['date'] = str(list(row['happentime'])[0])[:-9]
                feature['date_delta'] = (pd.Timestamp("2018-6-22")-row['happentime'][0]).days
            else:
                feature['section'] = 'null'
                feature['len_watch'] = 0
                feature['len_session'] = 0
                feature['count_pause'] = 0
                feature['count_speed'] = 0
                feature['count_mute'] = 0
                feature['date'] = '0'
                feature['date_delta'] = 0
            return feature
        df = df.groupby('session').apply(f)
        ####################################
        df = df[df['date_delta'] > date_delta]
        ###################################
        count_session_total = len(df['count_pause'])

        if len(df)==0:
            return Constants.dict_feature
        df = df[df['section']!='null']
        len_watch = df['len_watch'].sum()
        len_watch_max = df['len_watch'].max()
        len_watch_min = df['len_watch'].min()
        len_watch_ave = df['len_watch'].sum() / len(df)
        len_session = df['len_session'].sum()
        len_session_max = df['len_session'].max()
        len_session_min = df['len_session'].min()
        len_session_ave = df['len_session'].sum() / len(df)
        count_pause = df['count_pause'].sum()
        count_speed = df['count_speed'].sum()
        count_mute = df['count_mute'].sum()
        count_session = len(df['count_pause'])
        count_date = len(df['date'].drop_duplicates())
        if len_session ==0:
            len_dataDelta = 0
        else:
            len_dataDelta = df['date_delta'].sum()/len_session
        # len_watch(s)
        return {"len_watch": len_watch, "len_session": len_session, "count_pause": count_pause,
                "count_speed": count_speed, "count_mute": count_mute, "count_session": count_session,
                "count_date": count_date, "count_session_total" : count_session_total,
                'len_dataDelta':len_dataDelta, 'len_watch_max':len_watch_max,
                'len_watch_min':len_watch_min, 'len_watch_ave':len_watch_ave,
                'len_session_max':len_session_max, 'len_session_min':len_session_min,
                'len_session_ave':len_session_ave}
    else:
        return Constants.dict_feature

# print(get_base_feature(['beidawlf_01_02_02'], '1700022701', "2018-01-01", "2018-07-01", conn))

'''
业务2：单时段特征（基于基础特征）
'''
def f_Time(Q_type, stage, date_delta=0):
    '''

    :param Q_type:
    :param stage:
    :param date_delta: 距离考试时间，0为距离考试大于0天的
    :return:
    '''
    videoPath = "video/" + stage
    examinationPath = "examination/stage1"
    testQuestionsPath = "testQuestions/stage1"
    if os.path.exists(videoPath):
        pass
    else:
        os.makedirs(videoPath)
        os.makedirs(videoPath + "knowledgePoint")
        os.makedirs(videoPath +"knowledgePoint"+ Constants.list_type[0])
        os.makedirs(videoPath +"knowledgePoint"+  Constants.list_type[1])
        os.makedirs(videoPath +"knowledgePoint"+  Constants.list_type[2])
    df_knowledgePoint2sections = myUtils.f2_knowledgePoint2sections()
    df_base = pd.read_sql("select * from stu_base", Constants.myConn)
    list_stuId = df_base['stu_id'].tolist()
    df = pd.read_csv(examinationPath+'/knowledge'+Q_type+'Finished.csv')
    list_key = df.columns.tolist()[:-1]
    df_all = pd.DataFrame()
    for key in list_key:
        print(key)
        list_res = []
        for stu_id in list_stuId:
            list_ = [stu_id, key]
            for type in Constants.list_KP_type:
                dict_feature = get_base_feature(list(df_knowledgePoint2sections[df_knowledgePoint2sections['key']==key][type])[0],
                                                stu_id , Constants.startTime, Constants.endTime, date_delta, Constants.conn)
                for key_, value_ in dict_feature.items():
                    list_ = list_ + [value_]
            list_res.append(list_)
        df_key = pd.DataFrame(list_res, columns=['stu_id','key_name']+Constants.subchapter_feature
                              +Constants.chapter_feature
                              +Constants.follow_subchapter_feature
                              +Constants.parallel_subchapter_feature
                              +Constants.front_subchapter_feature)
        df_base['stu_id']= df_base['stu_id'].apply(pd.to_numeric)
        df['stu_id'] = df['stu_id'].apply(pd.to_numeric)
        df_key['stu_id'] = df_key['stu_id'].apply(pd.to_numeric)
        df_res = pd.merge(df_key, df_base[Constants.list_base], on='stu_id')
        df_res = pd.merge(df_res,df[['stu_id',key]], on='stu_id')
        df_ = df_res[key]
        del df_res[key]
        df_res['label'] = df_
        # 粘合测试题特征
        df_testQuestion = pd.read_csv(testQuestionsPath+"/testQuestion.csv")
        df_res = pd.merge(df_res, df_testQuestion)
        print(df_res.shape)
        df_res.to_csv(videoPath+'/knowledgePoint/'+Q_type+'/'+key+'.csv',index=False)
        df_all = pd.concat([df_all,df_res])
    df_all.to_csv(videoPath+"/knowledgePoint/"+Q_type+"/all.csv", index=False)

if __name__ == "__main__":
    for Q_type in Constants.list_type:
        f_Time(Q_type, "stage1")
    print("======>base stage finished<======")
    import featureProject
    featureProject.merge_forum()
    print("======>merge stage finished<======")
    import analysis
    analysis.stage1_train("RF")
    print("======>RF train stage finished<======")
    analysis.stage1_train("GDBT")
    print("======>GDBT train stage finished<======")




