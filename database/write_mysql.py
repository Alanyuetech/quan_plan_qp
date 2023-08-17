#写入mysql中的数据
import pandas as pd
import pymysql
import logging 
from conn_mysql import ConnectMysql

logger = logging.getLogger('quant_main')

class WriteMysql(ConnectMysql):
    def __init__(self,host:str,port:str,user:str,password:str,db_name:str =''):
        super().__init__(host,port,user,password,db_name)
        if self.db_name=='':
            logger.info("Mysql: The current connected database is: {}   Please connect to database first!".format(self.db_name))        
    
    def write_table(self,table_name:str,columns:list,data:list,db_name:str =''):
        '''将数据写入数据表中,写入单条数据,data为list or tuple,data样例:(1,2,3) 或 [1,2,3]'''
        if db_name=='':
            db_name=self.db_name
        if db_name=='':
            logger.info("Mysql: The current connected database is: {}   Please connect to database first!".format(self.db_name))

        try:
            sql = "insert into %s.%s (%s) values (%s)" % (db_name,table_name,','.join(columns),','.join(['%s']*len(data)))
            self.cursor.execute(sql,data)
            self.conn.commit()
            logger.info("Mysql: Successfully insert table: %s.%s ; len: %s"%(db_name,table_name,len(data)))
        except Exception as e:
            logger.exception("Mysql: Failed to insert table: %s.%s ; \n  sql: %s"%(db_name,table_name,sql))
        
    def write_table_many(self,table_name:str,columns:list,data:list,db_name:str =''):
        '''将数据写入数据表中，写入多条数据,data为list or tuple,data样例:[(1,2,3),(4,5,6)] 或 [[1,2,3],[4,5,6]]'''
        if db_name=='':
            db_name=self.db_name
        if db_name=='':
            logger.info("Mysql: The current connected database is: {}   Please connect to database first!".format(self.db_name))
        try:
            sql = "insert into  %s.%s  (%s) values (%s)"% (db_name,table_name,','.join(columns),','.join(['%s']*len(data[0])))
            self.cursor.executemany(sql,data)
            self.conn.commit()
            logger.info("Mysql: Successfully insert table: %s.%s ; len: %s"%(db_name,table_name,len(data)))
        except Exception as e:
            logger.exception("Mysql: Failed to insert table: %s.%s ; \n  sql: %s"%(db_name,table_name,sql))
    
    def update_table(self,table_name:str,columns:list,data:list,condition:str,db_name:str =''):
        '''更新数据表中的数据'''
        if db_name=='':
            db_name=self.db_name
        if db_name=='':
            logger.info("Mysql: The current connected database is: {}   Please connect to database first!".format(self.db_name))
        try:
            columns_str = ','.join([column+'=%s' for column in columns])
            sql = "update %s.%s set %s where %s" % (db_name,table_name,columns_str,condition)
            self.cursor.execute(sql,data)
            self.conn.commit()
            logger.info("Mysql: Successfully update table: %s.%s ; len: %s"%(db_name,table_name,len(data)))
        except Exception as e:
            logger.exception("Mysql: Failed to update table: %s.%s ; \n  sql: %s"%(db_name,table_name,sql))
