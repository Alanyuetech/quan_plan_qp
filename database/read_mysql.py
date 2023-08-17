#读取mysql中的数据
import pymysql
import logging 
from conn_mysql import ConnectMysql

logger = logging.getLogger('quant_main')

class ReadMysql(ConnectMysql):
    def __init__(self,host:str,port:str,user:str,password:str,db_name:str =''):
        super().__init__(host,port,user,password,db_name)
        if self.db_name=='':
            logger.info("Mysql: The current connected database is: {}   Please connect to database first!".format(self.db_name))
            
    def read_table_all(self,table_name:str,db_name:str =''):
        '''读取数据表中的全部数据'''
        if db_name=='':
            db_name=self.db_name
        if db_name=='':
            logger.info("Mysql: The current connected database is: {}   Please connect to database first!".format(self.db_name))

        try:
            sql="select * from %s.%s"%(db_name,table_name)
            self.cursor.execute(sql)
            data=self.cursor.fetchall()
            logger.info("Mysql: Successfully read table: %s.%s ; len: %s"%(db_name,table_name,len(data)))
            return data
        except Exception as e:
            logger.exception("Mysql: Failed to read table: %s.%s ; \n  sql: %s"%(db_name,table_name,sql))
    
    def read_table_part(self,table_name:str,columns:list=['*'],condition:str=''):
        '''读取数据表中的部分字段,以及满足条件的数据'''
        if db_name=='':
            db_name=self.db_name
        if db_name=='':
            logger.info("Mysql: The current connected database is: {}   Please connect to database first!".format(self.db_name))

        try:
            columns_str = ','.join(columns)
            sql="select %s from %s.%s "%(columns_str,db_name,table_name)
            if condition!='':
                sql+=" where %s"%condition
            self.cursor.execute(sql)
            data=self.cursor.fetchall()
            logger.info("Mysql: Successfully read table: %s.%s ; len: %s"%(db_name,table_name,len(data)))
            return data
        except Exception as e:
            logger.exception("Mysql: Failed to read table: %s.%s ; \n  sql: %s"%(db_name,table_name,sql))

    def read_table_sql(self,sql):
        '''生产中可能有比较复杂的获取数据的逻辑,此函数提供sql接口,可直接使用sql进行查询'''
        # 判断sql中是否含有select语句,防止其他sql语句误操作
        if 'select' not in sql.lower():
            logger.error("Mysql: Please check the sql statement, it must be a select statement!")
            return None
        try:
            self.cursor.execute(sql)
            data=self.cursor.fetchall()
            logger.info("Mysql: Successfully read table by sql; len: {}".format(len(data)))
            return data
        except Exception as e:
            logger.exception("Mysql: Failed to read table by sql; \n  sql: %s"%sql)
