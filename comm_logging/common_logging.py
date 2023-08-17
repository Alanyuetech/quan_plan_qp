# 此函数作用是：定义一个通用的日志类，可以被其他模块调用
import logging
import time 
import os

# class Singleton(type):
#     _instances = {}
#     def __call__(cls, *args, **kwargs):
#         if cls not in cls._instances:
#             cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
#         return cls._instances[cls]

class Common_logging(object):
    def __init__(self,log_name:str):
        self.log_name=log_name
        self.logger=logging.getLogger(self.log_name)  #创建一个logger
        self.logger.setLevel(logging.INFO)  #设置日志级别
        self.log_path=os.path.join(os.path.dirname(os.path.dirname(__file__)),'comm_logging')  #获取日志路径
        self.log_file=self.log_name+'_'+time.strftime('%Y%m%d_%H{}%M{}%S{}'.format('h','m','s'),time.localtime(time.time()))+'.log'  #获取日志文件名
        self.log_file_path=os.path.join(self.log_path,self.log_file)  #获取日志文件路径
        self.fh=logging.FileHandler(self.log_file_path)  #创建一个handler，用于写入日志文件
        self.fh.setLevel(logging.INFO)  # 设置 handler 的级别
        # 创建一个formatter，并将它添加到handler中
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.fh.setFormatter(formatter)   
        self.logger.addHandler(self.fh)  #给logger添加handler



    def get_logger(self):
        return self.logger


# 创建日志对象，供其他模块调用，下方quant_main为主函数的log文件，其他如有需求，可以使用同样的方法创建
quant_main_logger = Common_logging('quant_main').get_logger()
# database_logger = Common_logging('database').get_logger()