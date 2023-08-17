# Cumulative volume and derivatives
import os
import sys
# sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'get_data'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'select_research_model'))
# import base_data,derived_data
import importlib

class Volume(object):
    def __init__(self):
        #调用数据获取模块
        # self.base_data = base_data
        pass
    def traditional_volume(self,model_name):
        #通过model_name 匹配具体的模型
        model = importlib.import_module(f'{model_name}')
        # cal factor volume
        return