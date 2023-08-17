#定义一个类,用于存放所有的策略,并且可以根据策略的名字,返回对应的策略
import numpy as np
class StrategyGroup:
    def __init__(self,data):
        self.data=data
    
    def ma_5_10(self):
        #此函数为5日均线上穿10日均线策略，5日均线上穿10日均线就买入，5日均线下穿10日均线就卖出
        self.strategy=self.data.data
        self.strategy['ma5']=self.strategy['close'].rolling(5).mean()
        self.strategy['ma10']=self.strategy['close'].rolling(10).mean()
        self.strategy['signal']=np.where(self.strategy['ma5']>self.strategy['ma10'],1,0)
        self.strategy['signal']=self.strategy['signal'].shift(1)
        self.strategy['volume']=np.where(self.strategy['signal']==1,100,0)
        return self.strategy
    def xxx(self):
        #此函数为 过去5个交易日波动率小于过去10个交易日波动率且过去5个交易日每天最大值和最小值的差的波动率小于过去10个交易日每天最大值和最小值的差的波动率，就买入，买入后持有到10个交易日波动率小于过去5个交易日波动率，就卖出，波动率用收盘价计算。
        self.strategy=self.data.data
        self.strategy['vol5']=self.strategy['close'].rolling(5).std()
        self.strategy['vol10']=self.strategy['close'].rolling(10).std()
        self.strategy['vol5_10']=self.strategy['vol5']/self.strategy['vol10']
        self.strategy['vol5_10']=self.strategy['vol5_10'].shift(1)
        self.strategy['max_min_5']=(self.strategy['high']-self.strategy['low'].min()).rolling(5).std()
        self.strategy['max_min_10']=(self.strategy['high']-self.strategy['low'].min()).rolling(10).std()
        self.strategy['max_min_5_10']=self.strategy['max_min_5']/self.strategy['max_min_10']
        self.strategy['max_min_5_10']=self.strategy['max_min_5_10'].shift(1)
        self.strategy['signal']=np.where((self.strategy['vol5_10']<1)&(self.strategy['max_min_5_10']<1),1,0)
        self.strategy['signal']=self.strategy['signal'].shift(1)
        self.strategy['volume']=np.where(self.strategy['signal']==1,100,0)
        return self.strategy
    

