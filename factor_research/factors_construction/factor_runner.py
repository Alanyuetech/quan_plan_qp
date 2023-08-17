# 同样的包只会加载一次
from liquidity_fatcor_group  import volume
from momentum_factor_group import cumulative_return

print('new')
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'get_data'))  

from base_data import get_basedata

print('new2')
getbasedata = get_basedata.GetBaseData()
getbasedata.get_base_data('read_mysql')





print('sll')
aa = volume.Volume()
aa.traditional_volume('dl_model')

print('sll 2')
bb = cumulative_return.CumulativeReturn()
bb.traditional_cum_return('lr_model')

cc = cumulative_return.CumulativeReturn()
cc.traditional_cum_return('ml_model')

#主程序
if __name__ == '__main__':
    pass