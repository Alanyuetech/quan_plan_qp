# #打开csv文件
# import pandas as pd
# data = pd.read_csv('data.csv')

# #dataframe的一列，随机赋值每一行数0或1
# import numpy as np
# data['label'] = np.random.randint(0,2,size=len(data))


##################################################################################################################################
# #随机正整数
# import random
# a=random.randint(0,100)
# #小数取整
# import math
# math.floor(1.2)

# #dataframe 导出为csv


##################################################################################################################################
# #dataframe 删除'time_label','c_time_label'列
# data.drop(['time_label','c_time_label'],axis=1,inplace=True)

##################################################################################################################################
# #读取excel文件，提供文件名的前半段，通过模糊查询读取文件
# import glob
# import pandas as pd
# data = pd.read_excel(glob.glob('data*.xlsx')[0])

##################################################################################################################################
# #两个dataframe，根绝ID列合并，找到不公有的ID
# import pandas as pd
# df1 = pd.DataFrame({'ID':[1,2,3,4,5],'A':[1,2,3,4,5]})
# df2 = pd.DataFrame({'ID':[1,2,3,4,5,6],'B':[1,2,3,4,5,6]})
# df3 = pd.merge(df1,df2,on='ID',how='outer')
# df3[df3['A'].isnull()|df3['B'].isnull()]


##################################################################################################################################
# import argparse  # 命令行参数解析包


# # 将命令行中输入的数字组合成一个四位数
# parser = argparse.ArgumentParser(description='命令行中输入若干个数字')
# parser.add_argument('n1', type=int, help='输入第一个数字')
# parser.add_argument('n2', type=int, help='输入第二个个数字')
# parser.add_argument('-n3', type=int, help='输入一个数字')
# parser.add_argument('--n4', type=int, help='输入一个数字')
# args = parser.parse_args()
# ans = 0
# ans = 1000 * args.n1 + 100 * args.n2 + 10 * args.n3 + args.n4
# print(ans)

# import argparsecd
# parser = argparse.ArgumentParser(description='test')

# parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
# parser.add_argument('--seed', type=int, default=72, help='Random seed.')
# parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')

# args = parser.parse_args()

# print(args.sparse)
# print(args.seed)
# print(args.epochs)

# print("当文件是被调用时为模块名,当文件被执行时为 __main__",__name__)

##################################################################################################################################
# #写一个函数，输入一个dataframe,一个列名sum_col,和不定数量的列名，按不定数量的列名分组，对sum_col求和，返回一个dataframe
# import pandas as pd
# def group_and_sum(df, sum_col, *group_cols):
#     return df.groupby(list(group_cols))[sum_col].sum().reset_index()
# # 使用方法示例：
# # df = pd.DataFrame(...)
# # result_df = group_and_sum(df, 'column_to_sum', 'group_col1', 'group_col2', 'group_col3')

##################################################################################################################################
#dataframe一列的积
# dff['a'].prod()

##################################################################################################################################
#多个dataframe merge
# from functools import reduce
# dfs = [df1, df2, df3]
# df_final = reduce(lambda left,right: pd.merge(left,right,on='id'), dfs)

##################################################################################################################################
#numpy.float64的nan 转为np.nan


##################################################################################################################################
# #监控代码运行时间
# start_time = time.time()
# end_time = time.time()
# print("运行时间：",end_time-start_time) 

# import warnings
# def get_fields(file_key, fields=[], add=[]):  #file_key='STOCK_VALUATION_INDICATOR', fields=['free_turnover'], add=['trade_date']，运行结果是：['trade_date', 'free_turnover']
#     if fields == '*' or fields ==['*']:
#         return []
#     if fields:
#         fields = list(set(add + [f for f in fields if f in getattr(FIELDS, file_key) if f not in add and 'symbol' not in f]))
#         if not fields:
#             warnings.warn("No fields selected.")
#     return fields

##################################################################################################################################
#将dataframe一列中 . 后面的部分提取出来并去重
# df['a'].str.split('.').str[1].unique()

##################################################################################################################################
#dataframe一列中等于 a 或者 b 的行提取出来
# df[df['a'].isin(['a','b'])]

##################################################################################################################################
# #运行 qdw-data-validation 下的main.py文件 ，main中有需要输入的参数
# import os
# os.system('python qdw-data-validation/main.py --file_name=STOCK_VALUATION_INDICATOR --fields=free_turnover --add=trade_date')

##################################################################################################################################
# #使用os 运行 qdw-data-validation  下 的result.py文件中的 CompareResult 类,CompareResult 类中有需要输入的参数 name=symbol_name_mapping
# import os
# #运行  CompareResult  类

##################################################################################################################################
#dataframe 类型为 Decimal 的一列    保留4位小数

# #定义一个两列的df，其中一列a为 Decimal 类型
# import pandas as pd
# from decimal import Decimal
# df = pd.DataFrame({'a':[Decimal('1.23456789'),Decimal('2.5456789')],'b':[1,2]})
# #将a列不保留小数
# df['a'] = df['a'].apply(lambda x:round(x,0))
# print(df)
# print(type(df['a'][0]))

# import numpy as np
# a=np.nan
# b=3
# c=a*b
# print(c)
# print(c is not np.nan)
# print(np.isnan(c))

# #list_a 是一个list,有1020 个元素，每次取700个 赋值给 a
# list_a = [i for i in range(1020)]
# a = [list_a[i:i+700] for i in range(0, len(list_a), 700)]

##################################################################################################################################
#测试不同方法rollingdataframe一列的速度

# #生成1000个随机数作为dataframe的一列1
# import pandas as pd
# import numpy as np
# import time
# from functools import reduce 
# df = pd.DataFrame({'a':np.random.uniform(1,2,10**6)})

# df_1,df_2,df_3,df_4,df_5,df_6=df.copy(),df.copy(),df.copy(),df.copy(),df.copy(),df.copy()

# num=100
# expresion = '*'.join([f'df_6["a"].shift({i})' for i in range(num)])

# start_dt = time.time()
# df_1['a_prod5'] = df_1['a'].rolling(num).apply(lambda x:np.exp(np.log(x).sum()),raw=True)
# end_dt = time.time()
# print("rolling 运行时间：",end_dt-start_dt)

# start_dt = time.time()
# for i in range(1,num):
#     df_2[f'a_{i}'] = df_2['a'].shift(i)
# df_2['a_prod5'] = df_2.loc[:,'a':'a_{}'.format(num-1)].prod(min_count=num,axis=1)
# end_dt = time.time()
# print("shift add columns 运行时间：",end_dt-start_dt)

# start_dt = time.time()
# df_3['a_prod5'] = df_6["a"]*df_6["a"].shift(1)*df_6["a"].shift(2)*df_6["a"].shift(3)*df_6["a"].shift(4)*df_6["a"].shift(5)*df_6["a"].shift(6)*df_6["a"].shift(7)*df_6["a"].shift(8)*df_6["a"].shift(9)*df_6["a"].shift(10)*df_6["a"].shift(11)*df_6["a"].shift(12)*df_6["a"].shift(13)*df_6["a"].shift(14)*df_6["a"].shift(15)*df_6["a"].shift(16)*df_6["a"].shift(17)*df_6["a"].shift(18)*df_6["a"].shift(19)*df_6["a"].shift(20)*df_6["a"].shift(21)*df_6["a"].shift(22)*df_6["a"].shift(23)*df_6["a"].shift(24)*df_6["a"].shift(25)*df_6["a"].shift(26)*df_6["a"].shift(27)*df_6["a"].shift(28)*df_6["a"].shift(29)*df_6["a"].shift(30)*df_6["a"].shift(31)*df_6["a"].shift(32)*df_6["a"].shift(33)*df_6["a"].shift(34)*df_6["a"].shift(35)*df_6["a"].shift(36)*df_6["a"].shift(37)*df_6["a"].shift(38)*df_6["a"].shift(39)*df_6["a"].shift(40)*df_6["a"].shift(41)*df_6["a"].shift(42)*df_6["a"].shift(43)*df_6["a"].shift(44)*df_6["a"].shift(45)*df_6["a"].shift(46)*df_6["a"].shift(47)*df_6["a"].shift(48)*df_6["a"].shift(49)*df_6["a"].shift(50)*df_6["a"].shift(51)*df_6["a"].shift(52)*df_6["a"].shift(53)*df_6["a"].shift(54)*df_6["a"].shift(55)*df_6["a"].shift(56)*df_6["a"].shift(57)*df_6["a"].shift(58)*df_6["a"].shift(59)*df_6["a"].shift(60)*df_6["a"].shift(61)*df_6["a"].shift(62)*df_6["a"].shift(63)*df_6["a"].shift(64)*df_6["a"].shift(65)*df_6["a"].shift(66)*df_6["a"].shift(67)*df_6["a"].shift(68)*df_6["a"].shift(69)*df_6["a"].shift(70)*df_6["a"].shift(71)*df_6["a"].shift(72)*df_6["a"].shift(73)*df_6["a"].shift(74)*df_6["a"].shift(75)*df_6["a"].shift(76)*df_6["a"].shift(77)*df_6["a"].shift(78)*df_6["a"].shift(79)*df_6["a"].shift(80)*df_6["a"].shift(81)*df_6["a"].shift(82)*df_6["a"].shift(83)*df_6["a"].shift(84)*df_6["a"].shift(85)*df_6["a"].shift(86)*df_6["a"].shift(87)*df_6["a"].shift(88)*df_6["a"].shift(89)*df_6["a"].shift(90)*df_6["a"].shift(91)*df_6["a"].shift(92)*df_6["a"].shift(93)*df_6["a"].shift(94)*df_6["a"].shift(95)*df_6["a"].shift(96)*df_6["a"].shift(97)*df_6["a"].shift(98)*df_6["a"].shift(99)
# end_dt = time.time()
# print("shift single column 运行时间：",end_dt-start_dt)

# start_dt = time.time()
# df_4['a_prod5'] = 1
# for i in range(num):
#     df_4['a_prod5'] = df_4['a_prod5'].mul(df_4['a'].shift(i))
# end_dt = time.time()
# print("shift single column + for 运行时间：",end_dt-start_dt)

# strat_dt = time.time()
# shifted_cols = [df_5['a'].shift(i) for i in range(num)]
# df_5['a_prod5'] = reduce(lambda x,y:x*y,shifted_cols)
# end_dt = time.time()
# print("shift single column + reduce 运行时间：",end_dt-start_dt)

# strat_dt = time.time()
# df_6['a_prod5'] = eval(expresion)
# end_dt = time.time()
# print("shift single column + eval 运行时间：",end_dt-start_dt)
# print(expresion)

##################################################################################################################################
# #生成一个含有null 的dataframe
# import pandas as pd
# import numpy as np
# df = pd.DataFrame({'a':[1,2,3,np.nan,5],'b':[1,2,3,4,5]})

##################################################################################################################################
# import numpy_financial as npf   
# print(npf.irr([-250000, 100000, 150000, 200000, 250000, 300000]))

##################################################################################################################################
# #看一个变量占用内存的大小,单位是MB
# import sys
# a = 1
# print(sys.getsizeof(a)/1024/1024)

##################################################################################################################################
# #if判断一个变量是否有数据，这个变量可能没有被定义
# # a=1
# if 'a' not in locals().keys():
#     print('a is not defined')
# else:
#     print('a is ',a)

##################################################################################################################################
#调用当前目录下的py文件
#import 

##################################################################################################################################

# #重写下方代码，因为以下代码可能导致result占用内存很大，需要输入一个参数，控制result是否  分块输出写入数据库
# if batch_write:
#     #自动确定每一次循环的start和end，最初跑一年的数据，根据一年的数据占据内存的大小，确定每一次循环的start和end
#     #enddt初始化为start一年后的日期：start，enddt为字符串
#     enddt_days = 365
#     enddt = (datetime.datetime.strptime(start, '%Y%m%d') + datetime.timedelta(days=enddt_days)).strftime('%Y%m%d')
#     #设定result内存占用阈值为1G
#     memory_threshold = 1    
#     update_enddt = 0
#     while update_enddt==0:
#         result = func_loader.user_funcs.run(start, enddt)
#         if (sys.getsizeof(result)/(1024**3))>memory_threshold: 
#             enddt_days = enddt_days//2  ##重新设定enddt  每次减半
#             enddt = (datetime.datetime.strptime(start, '%Y%m%d') + datetime.timedelta(days=enddt_days)).strftime('%Y%m%d')
#             continue  
#         else:
#             enddt_days = int(enddt_days / ((sys.getsizeof(result)/(1024**3)) / memory_threshold) )
#             enddt = (datetime.datetime.strptime(start, '%Y%m%d') + datetime.timedelta(days=enddt_days)).strftime('%Y%m%d')
#             update_enddt = 1


#     startdt = start
#     while enddt <= end:
#         result = func_loader.user_funcs.run(startdt, enddt)
#         logger.info("Write data to database.")
#         persistence = SimpleFactorPersistence()
#         persistence.factor_write(df=result,
#                             group=factor_base_info.group_name,
#                             factor=factor_base_info.factor_name,
#                             override=override,
#                             universe=universe,
#                             dry_run=dry_run)
#         startdt = (datetime.datetime.strptime(startdt, '%Y%m%d') + datetime.timedelta(days=1)).strftime('%Y%m%d')
#         enddt = (datetime.datetime.strptime(startdt, '%Y%m%d') + datetime.timedelta(days=enddt_days)).strftime('%Y%m%d')
        
#     logger.info("{} {} finish".format(factor_base_info.group_name, factor_base_info.factor_name))
        
# else:
#     result = func_loader.user_funcs.run(start, end) 

#     # if dry_run:
#     #     logger.info("Dry run finish")
#     #     return result

#     logger.info("Write data to database.")
#     persistence = SimpleFactorPersistence()
#     persistence.factor_write(df=result,
#                             group=factor_base_info.group_name,
#                             factor=factor_base_info.factor_name,
#                             override=override,
#                             universe=universe,
#                             dry_run=dry_run)
#     logger.info("{} {} finish".format(factor_base_info.group_name, factor_base_info.factor_name))

##################################################################################################################################
# #写一个生成器，迭代器的样例
# def gen():
#     for i in range(10):
#         yield i  #yield是生成器的关键字，每次循环返回一个值，但是不会终止循环
# for i in gen():
#     print(i)


# # 只有当我们调用next()方法时，才会真正的从内存中取出数据，并且这个数据只能被取出一次
# def get_num():
#     num = 0
#     while num < 10:
#         num += 1
#         yield num
 
# for g in get_num():
#     print(g)

##################################################################################################################################
# #结合客户代码样式，写一个分批次落地数据的方案，避免客户run函数中的result占用内存过大导致爆内存
# import pandas as pd
# import numpy as np
# import time
# import inspect
# import sys

# def run(start, end):

#     aa =pd.DataFrame(columns=['1','2','3','4','5','6','7','8','9','10'])
#     def get_data(start_dt,end_dt):
#         return round(np.random.uniform(start_dt,end_dt,1).tolist()[0],3)
    
#     for i in range(start,end):
#         aa.loc[i]=get_data(i,i+1)

#     return aa


# def runy(start, end):

#     aa =pd.DataFrame(columns=['1','2','3','4','5','6','7','8','9','10'])
#     def get_data(start_dt,end_dt):
#         return round(np.random.uniform(start_dt,end_dt,1).tolist()[0],3)
    
#     for i in range(start,end):
#         aa.loc[i]=get_data(i,i+1)

#         # 替换 return
#         if (sys.getsizeof(aa)/1024/1024 > 0.1) or (i == end-1) :
#             yield aa
#             aa =pd.DataFrame(columns=['1','2','3','4','5','6','7','8','9','10'])
#         else:
#             continue


# if inspect.isgeneratorfunction(runy):  
#     st=time.time()
#     for res in runy(1,10000):
#         print(res)
#         print('写入数据库')
#         print(sys.getsizeof(res)/1024/1024,'MB')
#     et=time.time()-st
#     print('运行时间：',et)
# else:
#     st=time.time()
#     res = run(1,10000)
#     print(res)
#     print('写入数据库')
#     print(sys.getsizeof(res)/1024/1024,'MB')
#     et=time.time()-st
#     print('运行时间：',et)


# print(res.info(memory_usage='deep'))
# print('运行时间：',et)


##################################################################################################################################
# list 中模糊查询
# import re
# list_a = ['ab','cfdf','dhy','ebg','fyj','gfbgf','hd','ifgfg','lgj']
# list_b = [i for i in list_a if re.search('bg',i)]
# print(list_b)

##################################################################################################################################
# #判断在列表a中存在，但是在列表b中不存在的元素
# a = [1,2,3,4,5,6,7,8,9]
# b = [4,5,6,7,8,10,11,12,13]
# c = [i for i in a if i not in b]
# print(c)

##################################################################################################################################
# #for循环改造为并行计算
# from multiprocessing import Pool
# import pandas as pd

# def worker_process(i):  #函数执行任务
#     # Do some work ...
#     df = pd.DataFrame({"a": [i], "b": [i * 2]})  # Create a dataframe with one row
#     return df

# with Pool(processes=4) as pool:
#     result_objects = [pool.apply_async(worker_process, args=(i,)) for i in range(10)]

# # Do some other work ...

# result_dfs = [res.get() for res in result_objects]  # 此行代码运行时间过长的原因：

# # Concatenate all dataframes
# result = pd.concat(result_dfs, ignore_index=True)

##################################################################################################################################
# #循环一个list，以n为间隔，每次获取n个元素的第一个和最后一个，例如：n=5,第一次获取list的第一个和第5个元素，第二次获取第六个和第10个元素。
# list_a = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
# n=5
# for i in range(0,len(list_a),n):
#     print(list_a[i],list_a[min(i+n-1,len(list_a)-1)])


# ##################################################################################################################################
# #可以调用函数内部函数的并行解决方案
# from joblib import Parallel, delayed

# def run(start_date,end_date):
#     ...

#     def cal_run(trading_list):
#         ...

#     results = Parallel(n_jobs=4)(delayed(cal_run)(trading_list[i:min(i+t_interval-1,len(trading_list)-1)]) for i in range(0,len(trading_list),t_interval))
    
#     for rts in results:
#         pre_5_std_mean = pd.concat([pre_5_std_mean, rts]) 
#     return pre_5_std_mean

# ##################################################################################################################################
# #查询列表中元素含有 SSE 的元素
# import re  #
# list_a = ['SSE.000001','SSE.000002','SSE.000003','SSE.000004','SSE.000005','SSE.000006','SSE.000007','SSE.000008','SSE.000009','SSE.000010']
# list_b = [i for i in list_a if re.search('SSE',i)]

# ##################################################################################################################################
# #连接mongo数据库
# import pymongo
# client = pymongo.MongoClient(hf.DEFAULT_PERSIST['config'])
# db = client['hf']


# ##################################################################################################################################
# #mongodb 连接和查询数据--file_id下的具体数据
# from algoqi.factor.loader import tsdata_hf_factor as hf
# import pymongo
# import gridfs
# from bson.objectid import ObjectId
# from io import BytesIO
# from pandas import DataFrame, concat, read_csv

# hf.DEFAULT_PERSIST

# client = pymongo.MongoClient(hf.DEFAULT_PERSIST['config'])

# db = client['dev7_hf_data_db']
# fs = gridfs.GridFS(db,'hf_Financial_Futures_SF_volume_pct_Volume_hf_')   # hf_Financial_Futures_SF_volume_pct_Volume_hf_  是集合名
# db.list_collection_names()  #查询所有集合（表）

# collection=db['hf_Financial_Futures_SF_volume_pct_Volume_hf_']     #   hf_Futures_SF_Overlap_BBANDS_MIDDLEBAND_Overlap_hf_     hf_Financial_Futures_SF_volume_pct_Volume_hf_
# results=collection.find({'symbol':'IF.CFFEX','date':{'$gte':'20150105','$lte':'20150205'}})    #  {'symbol':'IF.CFFEX','date':{'$gte':'20150105','$lte':'20150205'}}
# dfs = []
# for doc in results:
#     buf = fs.get(doc['file_id']).read()
#     if not buf: continue
#     buff = BytesIO(buf)
#     df = read_csv(buff, compression="gzip")
#     df.insert(0, column='symbol', value=doc['symbol'])
#     dfs.append(df)   
# print(dfs)

# ##################################################################################################################################
# import numpy as np
# # np.where   #条件为列表时，出现了报错 The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all(). 
# #解决方案：将条件改为np.array(条件)

# import numpy as np
# import pandas as pd

# print('1',np.array([np.nan]).astype(bool))  # 输出：[ True]
# print('2',pd.Series([np.nan]).astype(bool))  # 输出：0    True


# print(bool(np.nan))


# ##################################################################################################################################
# week_sign = -1 if common.trading_days(TODAY_STR,TODAY_STR is None) else 0


# ##################################################################################################################################
# import pandas as pd
# from decimal import Decimal
# df = pd.DataFrame({'a':[Decimal('1.23456789'),Decimal('2.5456789')],'b':[1,2]})
# #输出a列的第一个元素和最后一个元素
# print(df['a'].iloc[0],df['a'].iloc[-1])


# ##################################################################################################################################
# #定义一个 dict ，有嵌套结构
# dict_a = {'a':1,'b':2,'c':{'d':True,'e':4}}
# print(dict_a)
# print(dict_a['d'].get('d',True))
# try:
#     lasdf = dict_a['d'].get('d',True)
# except Exception as e:
#     lasdf = True
# print(lasdf)
# print(type(str(True)))
# print(str('true'))

##################################################################################################################################
#调用comm_logging模块
# from ..comm_logging import comm_logging

##################################################################################################################################
# #子类初始化是 调用父类的初始化方法
# class Parent:
#     def __init__(self):
#         print("Parent __init__")

# class Child(Parent):
#     def __init__(self):
#         super().__init__()  # Call Parent __init__
#         print("Child __init__")

# c = Child()

##################################################################################################################################
# # 字符串 换行
# str_a = '可以可以\n下一行展示'
# print(str_a)

##################################################################################################################################
# #创建一个list,将list中的字符串提取出来，组成一个新的字符串，中间用逗号隔开
# columns = ['*']
# columns_str = ','.join(columns)
# table_name = 'basedata'
# sql = "select %s from %s" % (columns_str,table_name)
# print(sql)

##################################################################################################################################
# #insert 数据
# import pandas as pd
# table_name = 'basedata'
# data = [[1,4],[2,5],[3,6]]
# columns = ['a','b']
# #将data中的数据插入到table_name表中，如 (1,4),(2,5),(3,6)
# sql = "insert into %s (%s) values (%s)" % (table_name,','.join(columns),','.join(['%s']*len(data[0])))
# print(sql)

##################################################################################################################################
# # update 数据
# table_name = 'basedata'
# columns = ['a','b']
# condition = 'a=1'
# columns_str = ','.join([column+'=%s' for column in columns])
# sql = "update %s set %s where %s" % (table_name,columns_str,condition)
# print(sql)

##################################################################################################################################
# #测试建表
# db_name = 'hf'
# table_name = 'basedata'
# sql = "CREATE %s.TABLE %s "%(db_name,table_name)
# print(sql)


##################################################################################################################################
# def get_conf_data(self):  #
#     if self.start and self.end:
#         return self.start, self.end, self.override, self.universe, self.data_restrict
#     else:
#         raise TemplateError("CommonAshareFactorConfSerializer", "Config ERROR.")  # 抛出异常,config error的作用是：在日志中打印出来

##################################################################################################################################
# config = add_config(os.path.join(BASEDIR, env))  #此代码含义是：将env文件中的配置信息，添加到config中
# config['meta_path'] = os.path.join(BASEDIR, 'meta')  #此代码含义是：将meta文件夹的路径，添加到config中


##################################################################################################################################
# print("Mysql: Please connect to databases first! \n or use 'USE DATABASE' to specify a database! \n or Assign a value to db_name")


##################################################################################################################################
# # #series 转dataframe后有三列 symbol	datetime	value，另symbol为index,datetime中的数据作为列名

# #构建一个 MultiIndex 的 series
# import pandas as pd
# import numpy as np
# arrays = [['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
#             ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']]
# tuples = list(zip(*arrays))
# index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
# s = pd.Series([1,np.nan,1,1,1,np.nan,1,np.nan], index=index)
# print(s)


##################################################################################################################################
# #构建一个dataframe,有三列 symbol	datetime	value,每一个symbol的行数不同
# import pandas as pd
# import numpy as np
# df_all = pd.DataFrame({'symbol':['000001','000001','000001','000002','000002','000002','000003','000003'],
#                     'datetime':['2019-01-01','2019-01-02','2019-01-03','2019-01-01','2019-01-02','2019-01-03','2019-01-01','2019-01-02'],
#                     'value':[1,1,1,1,1,1,1,1]})
# df_all = df_all.set_index(['datetime'])
# def tt(df):
#     return df['value']
# out = df_all.groupby('symbol').apply(tt)
# print(out)
# print(type(out))
# print('~~~~~~~~~~~~~~~')
# out = df_all.groupby('symbol').apply(tt).unstack()
# print(out)
# print(type(out))


# #判断是否为dataframe
# import pandas as pd
# import numpy as np
# df_all = pd.DataFrame({'symbol':['000001','000001','000001','000002','000002','000002','000003','000003'],
#                     'datetime':['2019-01-01','2019-01-02','2019-01-03','2019-01-01','2019-01-02','2019-01-03','2019-01-01','2019-01-02'],
#                     'value':[1,1,1,1,1,1,1,1]})
# #判断df_all是否是 series
# print(isinstance(df_all,pd.Series))




# import pandas as pd

# # 创建两个Series，其中的索引部分重叠部分不重叠
# s1 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
# s2 = pd.Series([4, 5, 6], index=['b', 'c', 'd'])

# # 将Series转换为DataFrame
# df1 = s1.reset_index()
# df2 = s2.reset_index()

# # 合并DataFrame
# df = pd.merge(df1, df2, how='outer', on='index')
# print(df)
# # 'outer'参数表示我们想要保留所有的键，即使它们只在一个DataFrame中存在。
# # 如果你只想保留两个DataFrame都有的键，可以使用'inner'参数。
# # 如果你想保留一个DataFrame的键，可以使用'left'或'right'参数。


# ##################################################################################################################################
# #构建一个series，有n个元素，所有元素求和为1
# import pandas as pd
# import numpy as np
# n = 5
# s = pd.Series(np.random.rand(n))
# s = s/s.sum()
# print(s.sum())


# ##################################################################################################################################
# #随机获取dataframe中的n行
# import pandas as pd
# import numpy as np
# df = pd.DataFrame({'symbol':['000001','000001','000001','000002','000002','000002','000003','000003'],
#                     'datetime':['2019-01-01','2019-01-02','2019-01-03','2019-01-01','2019-01-02','2019-01-03','2019-01-01','2019-01-02'],
#                     'value':[1,1,1,1,1,1,1,1]})
# n = 2
# df = df.sample(n)
# print(df)


# ##################################################################################################################################
# #拼接三个dataframe
# import pandas as pd
# import numpy as np
# df1 = pd.DataFrame({'symbol':['000001','000001','000001','000002','000002','000002','000003','000003'],
#                     'datetime':['2019-01-01','2019-01-02','2019-01-03','2019-01-01','2019-01-02','2019-01-03','2019-01-01','2019-01-02'],
#                     'value':[1,1,1,1,1,1,1,1]})
# df2 = pd.DataFrame({'symbol':['000001','000001','000001','000002','000002','000002','000003','000003'],
#                     'datetime':['2019-01-01','2019-01-02','2019-01-03','2019-01-01','2019-01-02','2019-01-03','2019-01-01','2019-01-02'],
#                     'value':[1,1,1,1,1,1,1,1]})
# df3 = pd.DataFrame({'symbol':['000001','000001','000001','000002','000002','000002','000003','000003'],
#                     'datetime':['2019-01-01','2019-01-02','2019-01-03','2019-01-01','2019-01-02','2019-01-03','2019-01-01','2019-01-02'],
#                     'value':[1,1,1,1,1,1,1,1]})
# df = pd.concat([df1,df2,df3])
# print(df)


##################################################################################################################################
# #列表中小于某个日期的所有日期
# import pandas as pd
# import numpy as np  
# list1 = ['20200101','20200102','20200103','20200104','20200105','20200106','20200107','20200108','20200109','20200110']
# #获取list1中小于20200105的所有日期
# list2 = [x for x in list1 if x < '20200105']

##################################################################################################################################
# #按dataframe的名称获取不同的dataframe
# import pandas as pd
# import numpy as np
# df1 = pd.DataFrame({'symbol':['000001','000001','000001','000002','000002','000002','000003','000003'],
#                     'datetime':['2019-01-01','2019-01-02','2019-01-03','2019-01-01','2019-01-02','2019-01-03','2019-01-01','2019-01-02'],
#                     'value':[1,1,1,1,1,1,1,1]})
# df2 = pd.DataFrame({'symbol':['000001','000001'],
#                     'datetime':['2019-01-01','2019-01-02'],
#                     'value':[1,1]})
# df3 = pd.DataFrame({'symbol':['000001','000001','000001'],
#                     'datetime':['2019-01-01','2019-01-02','2019-01-03'],
#                     'value':[1,1,1]})
# for df_n in [df1,df2,df3]:
#     print(df_n)
#     print('~~~~~~~~~~~~~~~~~~~~~~~')




##################################################################################################################################
# #将dataframe的数据可视化为折线图
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# df = pd.DataFrame({'datetime':['2019-01-01','2019-01-02','2019-01-03','2019-01-01','2019-01-02','2019-01-03','2019-01-01','2019-01-02'],
#                     'value':[1,2,3,4,5,6,7,8]})
# df = df.set_index(['datetime'])
# #每十个数据显示横轴的日期
# df = df.reset_index()
# df = df.iloc[::10,:]
# df = df.set_index(['datetime'])
# df.plot()
# plt.show()

# #定义个dataframe，有3列，date和，stock,weight   
# import pandas as pd
# import numpy as np

# df = pd.DataFrame({'date':['20190101','20190102','20190102','20190103','20190104','20190105','20190105','20190106','20190107'],
#                      'stock':[np.nan,'000001','000002',np.nan,np.nan,'000001','000002',np.nan,np.nan],
#                         'weight':[np.nan,0.3,0.7,np.nan,np.nan,0.5,0.5,np.nan,np.nan]})
# print(df)
# # 先创建一个空的 DataFrame 用于保存最终的结果
# result = pd.DataFrame(columns=['date', 'stock', 'weight'])

# # 找出所有非空的日期（即有数据的日期）
# non_empty_dates = df.loc[~df['stock'].isna(), 'date'].unique()

# # 对于每一个非空的日期
# for i in range(len(non_empty_dates)):
#     # 获取当前日期
#     current_date = non_empty_dates[i]
    
#     # 如果当前日期不是最后一个非空的日期
#     if i < len(non_empty_dates) - 1:
#         # 获取下一个非空的日期
#         next_date = non_empty_dates[i+1]
        
#         # 获取当前日期和下一个非空日期之间的所有日期（包含当前日期，不包含下一个非空日期）
#         dates_to_fill = pd.date_range(start=current_date, end=next_date, closed='left').strftime('%Y%m%d')
#     else:
#         # 如果当前日期是最后一个非空的日期，则获取从当前日期到 df 的最后一个日期的所有日期
#         dates_to_fill = pd.date_range(start=current_date, end=df['date'].max()).strftime('%Y%m%d')
    
#     # 对于每一个需要填充的日期
#     for date in dates_to_fill:
#         # 复制当前日期的所有数据
#         data_to_fill = df[df['date'] == current_date].copy()
        
#         # 将复制的数据的日期设置为需要填充的日期
#         data_to_fill['date'] = date
        
#         # 将复制的数据添加到结果 DataFrame 中
#         result = result.append(data_to_fill, ignore_index=True)

# print(result)


##################################################################################################################################
#绘图
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates

# # 将date列转换为datetime类型，以便在图表上正确显示
# hz_df_plt['date'] = pd.to_datetime(hz_df_plt['date'])

# # 设置date为索引
# # hz_df_plt.set_index('date', inplace=True)

# # 创建一个新的figure和axes
# fig, ax = plt.subplots(figsize=(15,10))

# # 对于df中的每一列，绘制一条折线图
# for column in hz_df_plt.columns[1:]:
#     hz_df_plt[column].plot(ax=ax, label=column)

    
# # 设置x轴的主要刻度为每个月的第一天
# # ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
# # 设置x轴的主要刻度为每一天
# # ax.xaxis.set_major_locator(mdates.DayLocator())

# # 设置x轴的主要刻度格式为年月日
# # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))


# ax.set_xticks(hz_df_plt.index)
# # ax.set_xticks(hz_df_plt['date'])

# # ax.set_xticks(range(len(hz_df_plt)))
# # ax.set_xticklabels(hz_df_plt.index)
# ax.set_xticklabels(hz_df_plt['date'].dt.strftime('%Y-%m-%d'))

# ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

# # 自动旋转日期标签以防止重叠
# plt.gcf().autofmt_xdate()

# #控制横轴
# # ax.set_xlim([hz_df_plt.index.min(),hz_df_plt.index.max()])

# # 添加图例
# ax.legend()

# plt.text(-0.05,1,'单位：%',transform = ax.transAxes,verticalalignment='top')

# #先保存再显示
# plt.savefig("华证指数换手率.png")
# # 显示图表
# plt.show()

##################################################################################################################################
# # to_datetime  控制输出的格式为 '%Y%m%d'
# import pandas as pd
# import numpy as np
# df = pd.DataFrame({'date':['20190101','20190102','20190103','20190104','20190105','20190106','20190107','20190108','20190109','20190110'],
#                     'value':[1,2,3,4,5,6,7,8,9,10]})
# df['date'] = pd.to_datetime(df['date'],format='%Y%m%d')
# print(df)

##################################################################################################################################
# import pandas as pd

# # 创建示例数据框
# df = pd.DataFrame({'日期':['20190101','20190102','20190102','20190104','20190104','20190106','20190107'],
#                    '数值':[None,2,3,4,5,None,None]})

# # 将日期列转换为日期时间类型
# df['日期'] = pd.to_datetime(df['日期'])

# # 创建一个包含所有日期的数据框
# all_dates = pd.DataFrame({'日期': pd.date_range(start=df['日期'].min(), end=df['日期'].max())})

# # 将原始数据框与所有日期的数据框进行合并
# merged_df = pd.merge(all_dates, df, how='left', on='日期')

# # 使用ffill()函数向前填充数据
# merged_df['数值'] = merged_df['数值'].ffill()

# # 按日期排序
# merged_df = merged_df.sort_values('日期')

# # 打印结果
# print(merged_df.to_string(index=False))

##################################################################################################################################
#定义一个元组
a = (1,2,3)