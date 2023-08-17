# Cumulative returns and derivatives
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'database'))
import conn_mysql
import importlib
print('get_basedata')

class GetBaseData(object):
    def __init__(self):
        #调用数据获取模块
        print('GetBaseData')
        pass
    def get_base_data(self,module_name):
        self.module =  importlib.import_module(f'{module_name}')
        print(self.module)