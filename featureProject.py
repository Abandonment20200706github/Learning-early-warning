import Constants
import pandas as pd

'''
业务：将数据库的数据取出来
'''
# # 符合基本要求的95名学生
# # name email stu_id active ……
# sql = "select * from stu_base"
# df = pd.read_sql(sql, Constants.myConn)
# path = "baseInformation/stage0/stu_base.csv"
# df.to_csv(path, index=False)
#
# # 测试题
# # stu_id email chapter chapter_son chapter_grandson  description……
# sql = "select * from test_question"
# df = pd.read_sql(sql, Constants.myConn)
# path = "testQuestions/stage0/test_question.csv"
# df.to_csv(path, index=False)

'''
业务：将数据与后来添加的评论成绩合并
'''
def merge_forum():
    path_forum = "forum/stage1/forum.csv"
    for type in Constants.list_type:
        path_0 = "video/stage1/knowledgePoint/"+type+"/all.csv"
        df = pd.read_csv(path_0)
        df_forum = pd.read_csv(path_forum)
        df_all = pd.merge(df, df_forum)
        path_1 = "video/stage2/knowledgePoint/" + type + "/all.csv"
        df_all.to_csv(path_1, index=False)