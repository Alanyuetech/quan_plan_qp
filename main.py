#主函数作用待定(可能通过主函数调用不同的任务主题：因子研究，因子任务，交易等)
import logging
from comm_logging import common_logging
from database import conn_mysql
import pandas as pd

#获取日志对象
logger = logging.getLogger('quant_main')   
#连接数据库,初始化数据库对象
conmysql = conn_mysql.ConnectMysql(host='127.0.0.1',port=3306,user='root',password='123456',db_name='quan_plan')




#关闭数据库连接
conmysql.close_database()





