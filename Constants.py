import pymysql

# 连接moodle的数据库
conn = pymysql.connect(******)
# 连接自己的数据库
myConn = pymysql.connect(*******)

# 起始和结束日期
startTime = "2018-01-01"
endTime = "2018-06-22"

# 题型类别
list_type = ['Objective','Brief','Comprehensive']
list_base = ['stu_id','sex','subject','active','understanding','vision','sequence','duration','score']
list_KP_type = ['subchapter','chapter', 'follow_subchapter', 'parallel_subchapter', 'front_subchapter']

# 不同类型题目的知识点列表
col_keyObjective = ['1.1', '1.1', '1.4', '2.1', '2.3', '3.3', '3.4', '2.3', '2.5', '2.4', '6.1', '6.3',
               '2.3', '5.4', '8.4', '5.1', '6.2', '6.2', '5.2', '5.2', '5.5', '4.1', '5.2', '5.2',
               '7.1', '7.1', '7.2', '7.7', '7.4', '8.1', '8.1', '8.1', '8.4', '8.6', '6.4', '2.6',
               '7.1', '3.4', '2.5', '1.2', '1.2', '7.5']
col_keyBrief = ['2.5','7.9','1.2','1.4','12.3','3.8']
col_keyComprehensive = ['6.3','6.3','6.3','2.4','3.4']

# video基础特征
list_videoFeature = ["len_watch", "len_session", "count_pause",
                "count_speed", "count_mute", "count_session",
                "count_date", "count_session_total",
                'len_dataDelta','len_watch_max', 'len_watch_min',
                     'len_watch_ave','len_session_max','len_session_min',
                     'len_session_ave']
dict_feature = {}
subchapter_feature = []
chapter_feature = []
follow_subchapter_feature = []
parallel_subchapter_feature = []
front_subchapter_feature = []
for feature in list_videoFeature:
    dict_feature[feature] = 0
    subchapter_feature = subchapter_feature+["subchapter_"+feature]
    chapter_feature = chapter_feature+["chapter_" + feature]
    follow_subchapter_feature = follow_subchapter_feature+["follow_subchapter_" + feature]
    parallel_subchapter_feature = parallel_subchapter_feature+["parallel_subchapter_" + feature]
    front_subchapter_feature = front_subchapter_feature+["front_subchapter_" + feature]


# 评论情感识别的表头
list_forum = [ 'description', 'negative', 'neutral', 'positive']
list_comment = ['comment_negative', 'comment_neutral', 'comment_positive', 'comment_num_all']

# 两个无效学习者
list_dirtyStu = [1700022751, 1700022786]

# 经常要用到到的一个文件路径
path_baseInformation = "baseInformation/stage1/stu_base.csv"
