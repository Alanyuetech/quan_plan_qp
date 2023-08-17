#连接mysql 
import pymysql
import logging 

logger = logging.getLogger('quant_main')

class ConnectMysql:
    '''一般只初始化和关闭数据库，  创建原始数据库、数据表，删除等操作直接在数据库中操作,  涉及日常操作中的表备份，可以适当使用建表和删表操作 '''
    def __init__(self,host:str,port:str,user:str,password:str,db_name:str =''):
        self.host=host
        self.port=port
        self.user=user
        self.password=password
        self.db_name=db_name
        if self.db_name=='':
            self.db=pymysql.connect(host=self.host,port=self.port,user=self.user,password=self.password)
            self.cursor=self.db.cursor()
            logger.info("Mysql: Successfully connect to mysql server: %s"%self.db)
        else:
            self.db=pymysql.connect(host=self.host,port=self.port,user=self.user,password=self.password,db=self.db_name)
            self.cursor=self.db.cursor()
            logger.info("Mysql: Successfully connect to database: %s"%self.db_name)

    def create_database(self,db_name:str):
        '''创建数据库'''
        if self.db_name=='':
            try:
                sql="create database if not exists %s"%db_name
                self.cursor.execute(sql)
                self.db.commit()
                logger.info("Mysql: Successfully create database: %s"%db_name)
            except Exception as e:
                logger.exception("Mysql: Failed to create database: %s"%db_name)
        else:
            logger.info("Mysql: The current connected database is: {}   Please connect to mysql server first!".format(self.db_name))
    
    def create_table(self,table_name:str,columns_dict:dict,db_name:str=''):
        '''创建数据表'''
        if db_name=='':
            db_name=self.db_name
        if db_name=='':
            logger.info("Mysql: The current connected database is: {}   Please connect to database first!".format(self.db_name))
        try:
            #通过columns字典确定列名和列类型，拼接成sql语句
            sql = "CREATE TABLE %s.%s ("%(db_name,table_name)
            for column_name, column_type in columns_dict.items():
                sql += f"{column_name} {column_type}, "
            sql = sql.strip(", ") + ");"
            self.cursor.execute(sql)
            self.db.commit()
            logger.info("Mysql: Successfully create table: %s.%s"%(db_name,table_name))
        except Exception as e:
            logger.exception("Mysql: Failed to create table: %s.%s"%(db_name,table_name))

    def copy_table(self,table_name:str,new_table_name:str,db_name:str='',new_db_name:str=''):
        '''复制数据表,多用于日常操作中备份数据表'''
        # 如果db_name和new_db_name为空，则默认为self.db_name
        if db_name=='':
            db_name=self.db_name
        if new_db_name=='':
            new_db_name=self.db_name
        # 如果db_name和new_db_name都为空--即self.db_name为空，则日志记录
        if (db_name=='') or (new_db_name==''):
            logger.info("Mysql: db_name: {}  new_db_name: {} ; The database involved in the operation is not explicitly specified !".format(db_name,new_db_name))

        try:
            sql_create="create table %s.%s like %s.%s"%(new_db_name,new_table_name,db_name,table_name)
            sql_insert="insert into %s.%s select * from %s.%s"%(new_db_name,new_table_name,db_name,table_name)
            self.cursor.execute(sql_create)
            self.cursor.execute(sql_insert)
            self.db.commit()
            logger.info("Mysql: Successfully copy table: %s.%s to %s.%s"%(new_db_name,new_table_name,db_name,table_name))
        except Exception as e:
            logger.exception("Mysql: Failed to copy table: %s.%s to %s.%s"%(new_db_name,new_table_name,db_name,table_name))

    def delete_table(self,table_name:str,db_name:str=''):
        '''删除数据表'''
        if db_name=='':
            db_name=self.db_name
        if db_name=='':
            logger.info("Mysql: The current connected database is: {} ; Can't delete a table without a specified database!".format(db_name))

        try:
            sql="drop table if exists %s.%s"%(db_name,table_name)
            self.cursor.execute(sql)
            self.db.commit()
            logger.info("Mysql: Successfully delete table: %s.%s"%(db_name,table_name))
        except Exception as e:
            logger.exception("Mysql: Failed to delete table: %s.%s"%(db_name,table_name))

    def delete_database(self,db_name:str):
        '''删除数据库,不需要具体的上下文'''
        try:
            sql="drop database if exists %s"%db_name
            self.cursor.execute(sql)
            logger.info("Mysql: Successfully delete database: %s"%db_name)
        except Exception as e:
            logger.exception("Mysql: Failed to delete database: %s"%db_name)

    def use_database(self,db_name:str):
        '''切换数据库'''
        try:
            sql="use %s"%db_name
            self.cursor.execute(sql)
            self.db.commit()
            logger.info("Mysql: Successfully use database: %s"%db_name)
        except Exception as e:
            logger.exception("Mysql: Failed to use database: %s"%db_name)

    def show_databases(self):
        '''显示当前mysql中的所有数据库'''
        try:
            sql="show databases"
            self.cursor.execute(sql)
            data=self.cursor.fetchall()
            logger.info("Mysql: Successfully show databases")
            return data
        except Exception as e:
            logger.exception("Mysql: Failed to show databases")
        
    def show_tables(self,db_name:str):
        '''显示当前数据库中的所有数据表'''
        if db_name=='':
            db_name = self.db_name 
        if db_name=='':
            logger.info("Mysql: Please connect to databases first! \n or use 'USE DATABASE' to specify a database! \n or Assign a value to db_name/self.db_name")

        try:
            sql="show tables in %s"%db_name
            self.cursor.execute(sql)
            data=self.cursor.fetchall()
            logger.info("Mysql: Successfully show tables in database: %s"%db_name)
            return data
        except Exception as e:
            logger.exception("Mysql: Failed to show tables in database: %s"%db_name)

    def close_database(self):
        '''关闭数据库'''
        try:
            self.cursor.close()
            self.db.close()
            logger.info("Mysql: Successfully close database : %s"%self.db)
        except Exception as e:
            logger.exception("Mysql: Failed to close database : %s"%self.db)





        