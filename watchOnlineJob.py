import pandas as pd
import pymysql
import os
from myUtils import conn,myConn
import myUtils

'''
该组特征的提取是一个三级结构：
整体时间阶段（期中、期末）
    |- 时序特征（基于基础特征）
        |- 基础特征
'''

'''
业务1：基本特征的提取
比如：无效---会话次数  有效---观看天数、观看时长(min)、交互次数、暂停|快进|静音次数、会话次数
'''
def get_base_feature(key, userid, conn):
    '''
    提取基础特征
    :param key: 小节列表
    :param userid: 用户标识
    :param conn: 数据库连接
    :return: 特征字典
    '''
    if key == []:
        return {"len_watch": 0, "len_session": 0, "count_pause": 0,
                "count_speed": 0, "count_mute": 0, "count_session": 0,
                "count_date": 0, "count_session_total" : 0}
    elif len(key)==1:
        sql = "select userid, behave, sectionid,happentime  from behavior where userid = "\
              +userid+" and sectionid in " + str(tuple(key))[:-2]+")"
        df = pd.read_sql(sql, conn)
    else:
        sql = "select userid, behave, sectionid,happentime  from behavior where userid = "\
              +userid+" and sectionid in " + str(tuple(key))
        df = pd.read_sql(sql, conn)
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
            else:
                feature['section'] = 'null'
                feature['len_watch'] = 0
                feature['len_session'] = 0
                feature['count_pause'] = 0
                feature['count_speed'] = 0
                feature['count_mute'] = 0
                feature['date'] = '0'
            return feature
        df = df.groupby('session').apply(f)
        count_session_total = len(df['count_pause'])

        df = df[df['section']!='null']
        len_watch = df['len_watch'].sum()
        len_session = df['len_session'].sum()
        count_pause = df['count_pause'].sum()
        count_speed = df['count_speed'].sum()
        count_mute = df['count_mute'].sum()
        count_session = len(df['count_pause'])
        count_date = len(df['date'].drop_duplicates())

        return {"len_watch": len_watch, "len_session": len_session, "count_pause": count_pause,
                "count_speed": count_speed, "count_mute": count_mute, "count_session": count_session,
                "count_date": count_date, "count_session_total" : count_session_total}
    else:
        return {"len_watch": 0, "len_session": 0, "count_pause": 0,
                "count_speed": 0, "count_mute": 0, "count_session": 0,
                "count_date": 0, "count_session_total" : 0}

get_base_feature(['beidawlf_01_02_02'], '1700022701', conn)

'''
业务2：时序特征（基于基础特征）
'''







'''
业务3：整体时间阶段（期中、期末）
'''




