import pandas as pd
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import preprocessing
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import json
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score,mean_absolute_error
import Constants
from sklearn.model_selection import LeaveOneOut

'''
独立业务：分数分布图
'''
def draw_distribute():
    path_ = "examination/stage0/scoreTotal.xlsx"
    df = pd.read_excel(path_)
    df_baseInformation = pd.read_csv(Constants.path_baseInformation)
    df = pd.merge(df_baseInformation, df, on='stu_id').sort_values('score_y')
    df_y = df.copy()
    start = 35
    end = 105
    interval = 5
    X = np.arange(start, end, interval)
    # for i in range(int((end-start)/interval)-1):
    #     if len(df_y[(df_y['score_y']>X[i]) & (df_y['score_y']<=X[i+1]) ]['score_y'])==0:
    #         pass
    #     else:
    #         df_y[(df_y['score_y'] > X[i]) & (df_y['score_y'] <= X[i + 1])]['score_y']=X[i]
    Y = df_y[ 'score_y'].tolist()

    plt.figure()
    plt.hist(Y, bins=10,  rwidth=0.9)
    X = np.arange(36, 100, 6.3)
    plt.xticks(X)
    plt.yticks([0, 3, 6, 9, 12, 15, 18])
    plt.ylabel("Frequency")
    plt.xlabel("Score")
    plt.show()

'''
业务： Stage1 拟合模型、预测
'''
def analysis_feature_improtance(X_data, y_score, list_all, clf):
    """
    根据模型的特征重要性
    :param data: 分数经过排序的原始数据
    :param list_all: 特征列表
    :return: 降序排序的特征列表
    """
    X_data = preprocessing.StandardScaler().fit_transform(X_data)
    clf = clf.fit(X_data, y_score)
    df_imo = pd.DataFrame()
    df_imo['name'] = list_all
    df_imo["feature_importances_"] = clf.feature_importances_
    df_imo = df_imo.sort_values(by="feature_importances_", ascending=False)
    print(df_imo)
    return df_imo['name'].tolist()

def stage1_clf_machine(data, list_all, clf):
    '''
    对单体进行预测，然后返回预测的结果
    :param data:
    :param list_all:
    :param clf:
    :return:
    '''
    X_data = data.loc[:, list_all]
    y_score = data['label']
    # 保证每次数据集的划分是确定的
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_score,train_size= 0.8,test_size = 0.2, random_state=42)
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    mse = mean_squared_error(y_test, y_predict)
    msle = mean_squared_log_error(y_test, y_predict)
    return msle, mse, list(y_predict), list(y_test), clf

def stage1_balance_label(data):
    def f(row):
        feature = pd.Series()
        feature['label'] = list(row['label'])[0]
        feature['num'] = len(row)
        return feature
    df = (data.groupby('label').apply(f))[['label','num']]
    df = df.sort_values(by='num',ascending=False).reset_index(drop=True).reindex()
    a = df.iloc[0]
    b = df.iloc[1]
    proportion = b['num']/a['num']
    df_a = data[data['label'] == a['label']]
    a_train, a_test = train_test_split(df_a, train_size=proportion,test_size=1-proportion, random_state=42)
    data = data[data['label']!=a['label']]
    data = pd.concat([data,a_train])
    return data

def stage1_trainSolo(data, list_all, Q_type, clfName):
    """
    训练模型
    *******************************************
    这部分调整分类器
    *******************************************
   :param start_time:开始时间
    :param timedelta: 时间段长度
    :param n: 时间段个数
    :param fp_in:输入文件名称
    :return:
    """
    data = data.sort_values(by="label").reset_index(drop=True).reindex()  # 按照分数从小到大排序，为了更清晰用肉眼表征现象
    len1 = len(data[data['label']==1])
    len0 = len(data[data['label'] == 0])
    len05 = len(data[data['label'] == 0.5])
    max_ = max(len1, len0)
    max_ = max(max_, len05)
    # def stage1_balance_label(data):
    #     def f(row):
    #         return row.sample(n=max_, random_state=2, replace=True)
    #     data = data.groupby('label').apply(f)
    #     return data
    data = stage1_balance_label(data)  # 正负例数量分布更均衡
    X_data = data.loc[:, list_all]
    y_score = data['label']
    ##############################################################
    clf = 0
    if clfName == "GDBT":
        params = {'n_estimators': 500, 'max_depth': 2, 'min_samples_split': 2,
                  'learning_rate': 0.01, 'loss': 'ls'}
        clf = GradientBoostingRegressor(**params)
    elif clfName == "RF":
        clf = RandomForestRegressor(max_depth=2, random_state=0)
    else:
        print("no clf")
    ##############################################################
    list_all = analysis_feature_improtance(X_data, y_score, list_all,clf)
    list_fea=[]
    min_msle = 1
    min_list_fea = []
    min_mse = []
    min_list_predict = []
    min_list_test = []
    min_clf = 0
    list_stu = data['stu_id'].tolist()
    for p in list_all:
        list_fea.append(p)
        ########################################################
        if clfName=="GDBT":
            params = {'n_estimators': 500, 'max_depth': 2, 'min_samples_split': 2,
                      'learning_rate': 0.01, 'loss': 'ls'}
            clf = GradientBoostingRegressor(**params)
        elif clfName=="RF":
            clf = RandomForestRegressor(max_depth=2, random_state=0)
        else:
            print("no clf")
        ########################################################
        tmp_msle, tmp_mse, tmp_list_predict, tmp_list_test, clf = stage1_clf_machine(data, list_fea, clf)
        if tmp_msle<min_msle:
            min_msle = tmp_msle
            min_list_fea = list_fea.copy()
            min_mse = tmp_mse
            min_list_predict = tmp_list_predict.copy()
            min_list_test = tmp_list_test.copy()
            min_clf = clf
        del tmp_list_predict
        del tmp_list_test
    min_list_test = [float(x) for x in min_list_test]
    res_dict = {'min_msle': min_msle.tolist(),
                'min_mse': min_mse.tolist(),
                'min_list_fea': min_list_fea,
                'min_list_predict': min_list_predict,
                'min_list_test':min_list_test,
                'list_stu':list_stu}
    joblib.dump(min_clf, "output/model/"+clfName+"_model_stage1" + Q_type + ".m")
    print("===========>", len(res_dict['min_list_fea']))
    return res_dict

def stage1_train(clfName):
    '''
    针对三个题型的三个模型
    :return:
    '''
    list_type = Constants.list_type
    for Q_type in list_type:
        data = pd.read_csv('video/stage2/knowledgePoint/'+Q_type+'/all.csv')
        print(data.shape)
        list_all = data.columns.tolist()[2:]
        list_all.remove("label")
        res_dict = stage1_trainSolo(data, list_all, Q_type, clfName)
        with open('output/result/'+clfName+'_result'+Q_type+'_stage1.json','w') as fw:
            json.dump(res_dict, fw)

# stage1_train("GDBT")
# stage1_train("RF")

'''
业务： Stage2 拟合模型、预测
'''
def stage2_clf_machine(data, list_all, clf):
    X_data = data.loc[:, list_all]
    y_score = data['label']
    X = np.arange(len(data))
    loo = LeaveOneOut()
    list_test = []
    for train_index, test_index in loo.split(X):
        X_train = X_data.iloc[train_index]
        X_test = X_data.iloc[test_index]
        y_train = y_score.iloc[train_index]
        y_test = y_score.iloc[test_index]
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        list_test.append([y_test.values[0], y_predict[0]])
    df = pd.DataFrame(list_test)
    mse = mean_squared_error(df[0], df[1])
    msle = mean_squared_log_error(df[0], df[1])
    return msle, mse, list(df[1]), list(df[0]), clf

def stage2_balance_label(data):
    def f(x):
        return int((int(x)-36)/5)
    data['flag'] = data['label'].apply(f)
    def f(row):
        return row.sample(n=17, random_state=2, replace=True)
    data = data.groupby('flag').apply(f)
    return data

def stage2_train(clfName):
    list_type = ['Objective','Brief','Comprehensive']
    df_res = pd.DataFrame()
    for Q_type in list_type:
        clf = joblib.load("output/model/"+clfName+"_model_stage1"+Q_type+".m")
        with open('output/result/'+clfName+'_result'+Q_type+'_stage1.json','r') as fr:
            feature = (json.load(fr))['min_list_fea']
        df_ = pd.DataFrame()
        df_forum = pd.read_csv("forum/stage1/forum.csv")
        for root, dir, files in os.walk("video/stage1/knowledgePoint/"+Q_type):
            for file in files:
                if file=="all.csv":
                    pass
                else:
                    df = pd.read_csv(os.path.join(root,file))
                    df = pd.merge(df_forum, df)
                    df_res['stu_id'] = df['stu_id']
                    df_train = df[feature]
                    df_[file[:-4]] = clf.predict(df_train)
        len_ = len(df_.T)
        df_res[Q_type] = (df_.T.sum().T)/len_
    df_score = pd.read_excel('examination/stage0/scoreTotal.xlsx')
    df_res = pd.merge(df_res, df_score)
    df_res['label'] = df_res['score']

    clf = 0
    ########################################################
    if clfName == "GDBT":
        params = {'n_estimators': 500, 'max_depth': 2, 'min_samples_split': 2,
                  'learning_rate': 0.01, 'loss': 'ls'}
        clf = GradientBoostingRegressor(**params)
    elif clfName == "RF":
        clf = RandomForestRegressor(max_depth=2, random_state=0)
    else:
        print("no clf")
    ########################################################
    df_res = stage2_balance_label(df_res)
    del df_res['flag']
    tmp_msle, tmp_mse, tmp_list_predict, tmp_list_test, clf = stage2_clf_machine(df_res, list_type, clf)

    # print("mp_msle, tmp_mse =========> ", tmp_msle, tmp_mse)
    df_ = pd.DataFrame([tmp_list_predict,tmp_list_test]).T
    df_ = df_.sort_values(0)

    return list(df_[0]), list(df_[1])

# print("++++++++++++++++++++++++++++")
# b2, b1 = stage2_train('RF')
# a2, a1 = stage2_train('GDBT')

def make_confusion_matrix(y_test, y_pred, class_names, str_):
    from sklearn.metrics import confusion_matrix
    import itertools
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title=str_+ ':Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title=str_+ ':Normalized confusion matrix')

    plt.show()

def f_():
    b2, b1 = stage2_train('RF')
    a2, a1 = stage2_train('GDBT')

    df_a = pd.DataFrame([a1, a2])
    df_a = df_a.T.sort_values(0)

    df_b = pd.DataFrame([b1, b2])
    df_b = df_b.T.sort_values(0)

    df_a[2] = df_b[1]
    # print(df_a)

    print(df_a.corr())

    fig, ax = plt.subplots()
    x = np.arange(1,len(df_a)+1)
    ax.plot(x,list(df_a[0]), 'r',label='True')
    ax.plot(x,list(df_a[2]), 'g', label='RF')
    ax.plot(x,list(df_a[1]), 'b', label='GDBT')
    legend = ax.legend( shadow=True, fontsize='x-large')
    plt.xticks([])
    ax.grid(True)
    plt.ylabel("Score")
    plt.xlabel("Student")
    plt.show()

    def f(x):
        if x>60:
            return 1
        else:
            return 0
    df_ = df_a.applymap(f)
    make_confusion_matrix(df_[0], df_[2], ['fail', 'pass'], "RF")
    make_confusion_matrix(df_[0], df_[1], ['fail', 'pass'], "GDBT")

    mse2 = mean_squared_error(list(df_a[0]), list(df_a[2]))
    msle2 = mean_squared_log_error(list(df_a[0]), list(df_a[2]))
    mae2 = mean_absolute_error(list(df_a[0]), list(df_a[2]))

    mse1 = mean_squared_error(list(df_a[0]), list(df_a[1]))
    msle1 = mean_squared_log_error(list(df_a[0]), list(df_a[1]))
    mae1 = mean_absolute_error(list(df_a[0]), list(df_a[1]))
    print("============")
    print(mse2, msle2, mae2)
    print(mse1, msle1, mae1)
    print("============")

    tmp = np.arange(0,100,5)
    list_res = []
    list_ = []
    df = df_a
    for i in tmp:
        df_a = df.copy()
        def f(x):
            if x>i:
                return 1
            else:
                return 0
        df_a = df_a.applymap(f)
        p_a = precision_score(y_true=list(df_a[0]), y_pred=list(df_a[2]))
        p_b = precision_score(y_true=list(df_a[0]), y_pred=list(df_a[1]))
        r_a = recall_score(y_true=list(df_a[0]), y_pred=list(df_a[2]))
        r_b = recall_score(y_true=list(df_a[0]), y_pred=list(df_a[1]))

        a_a = accuracy_score(y_true=list(df_a[0]), y_pred=list(df_a[2]))
        a_b = accuracy_score(y_true=list(df_a[0]), y_pred=list(df_a[1]))
        f1_a = f1_score(y_true=list(df_a[0]), y_pred=list(df_a[2]))
        f1_b = f1_score(y_true=list(df_a[0]), y_pred=list(df_a[1]))
        list_ = [i, a_a, a_b, f1_a, f1_b]
        print(list_)
        list_res.append(list_)
    print(pd.DataFrame(list_res))

    def f(mae, clfName):
        down = 60-mae
        up = 60+mae
        a = 0
        b = 0
        c = 0
        len_a = 0
        len_b = 0
        len_c = 0
        df_ = df[df[0]<60-mae1]
        if df_.empty==True:
            a=0
        else:
            len_a = len(df_)
            if clfName =="GDBT":
                a = len(df_[df_[1]<down])/len(df_)
            elif clfName =="RF":
                a = len(df_[df_[2] < down]) / len(df_)
            else:
                print("no clfName")
        df_ = df[(df[0]>=60-mae) & (df[0]<=60+mae)]
        if df_.empty==True:
            b=0
        else:
            len_b = len(df_)
            if clfName =="GDBT":
                b = len(df_[(df_[1] >= down) & (df_[1] <= up)]) / len(df_)
            elif clfName =="RF":
                b = len(df_[(df_[2] >= down) & (df_[2] <= up)]) / len(df_)
            else:
                print("no clfName")
        df_ = df[df[0]>60+mae]
        if df_.empty==True:
            c=0
        else:
            len_c = len(df_)
            if clfName =="GDBT":
                c = len(df_[df_[1] > up]) / len(df_)
            elif clfName =="RF":
                c = len(df_[df_[2]>up])/len(df_)
            else:
                print("no clfName")
        all = (len_a * a + len_b * b + len_c * c)/len(df)
        # print(len_a, len_b, len_c, len(df))
        # print(a, b, c, all)

    f(mae2, "RF")
    f(mae1, "GDBT")

f_()