#构建一个回测框架，输入参数为代码列表，开始时间，结束时间，手续费
#输出为回测结果，包括收益率，最大回撤，夏普比率，年化收益率，交易次数，胜率，盈亏比，最大连续盈利次数，最大连续亏损次数
#回测框架包括一个回测类，一个回测结果类，一个回测指标类，一个回测数据类，一个回测数据处理类，一个回测策略类，一个回测策略处理类
#回测类包括回测参数，回测数据，回测策略，回测结果，回测指标，回测数据处理，回测策略处理
import pandas as pd
import tushare as ts
import numpy as np
from stratety_group import StrategyGroup
class BackTest:
    #此类为回测类，输入参数为代码，开始时间，结束时间，手续费,初始资金，策略名称
    def __init__(self,code,start,end,fee,initial_capital,strategy_name):
        self.code=code
        self.start=start
        self.end=end
        self.fee=fee
        self.strategy_name=strategy_name
        self.initial_capital=initial_capital
        self.data=BackTestData(self.code,self.start,self.end)
        self.strategy=BackTestStrategy(self.data,self.strategy_name)
        self.result=BackTestResult(self.data,self.strategy,self.fee)
        self.indicator=BackTestIndicator(self.result)
        self.dataProcess=BackTestDataProcess(self.data)
        self.strategyProcess=BackTestStrategyProcess(self.strategy)
    def run(self):
        self.dataProcess.run()
        self.strategyProcess.run()
        self.result.run()
        self.indicator.run()
        return self.indicator
class BackTestData:
    #此类为获取数据类，输入参数为代码，开始时间，结束时间
    def __init__(self,code,start,end):
        self.code=code
        self.start=start
        self.end=end
        self.data=pd.DataFrame()
    def run(self):
        #输入代码为列表,其中包含多个代码时，返回的数据为多个代码的数据，此时需要对数据进行处理，将数据合并成一个数据
        if type(self.code)==list:
            for i in self.code:
                self.data=self.data.append(ts.get_k_data(i,self.start,self.end))
        else:
            self.data=ts.get_k_data(self.code,self.start,self.end)
        return self.data
class BackTestStrategy(StrategyGroup):
    #此类为策略类，输入参数为数据，策略名称,继承自策略组类
    def __init__(self,data,strategy_name):
        self.data=data
        self.strategy_name=strategy_name
        self.strategy=pd.DataFrame()
    def run(self):
        #通过strategy_name动态调用策略
        self.strategy=eval('self.'+self.strategy_name+'()')
        return self.strategy 
class BackTestResult:
    #此类为回测结果类，输入参数为数据，策略，手续费
    def __init__(self,data,strategy,fee):
        self.data=data
        self.strategy=strategy
        self.fee=fee
        self.result=pd.DataFrame()
    def run(self):
        self.result=self.strategy.strategy
        self.result['fee']=self.result['volume']*self.result['close']*self.fee
        self.result['profit']=self.result['volume']*(self.result['close']-self.result['open'])-self.result['fee']
        self.result['profit'].iloc[0]=0
        self.result['profit']=self.result['profit'].cumsum()
        self.result['capital']=self.result['profit']+10000
        return self.result
class BackTestIndicator:
    #此类为回测指标类，输入参数为回测结果
    def __init__(self,result):
        self.result=result
        self.indicator=pd.DataFrame()
    def run(self):
        self.indicator['return']=self.result.result['capital']/10000-1
        self.indicator['maxdrawdown']=(self.indicator['return'].cummax()-self.indicator['return'])/self.indicator['return'].cummax()
        self.indicator['maxdrawdown'].iloc[0]=0
        self.indicator['annualreturn']=self.indicator['return']/(self.result.result.shape[0]/252)
        self.indicator['sharp']=self.indicator['annualreturn']/self.indicator['return'].std()
        self.indicator['winrate']=self.result.result[self.result.result['profit']>0].shape[0]/self.result.result.shape[0]
        self.indicator['profitlossratio']=-self.result.result[self.result.result['profit']<0]['profit'].mean()/self.result.result[self.result.result['profit']>0]['profit'].mean()
        self.indicator['maxwin']=self.result.result[self.result.result['profit']>0]['profit'].max()
        self.indicator['maxloss']=self.result.result[self.result.result['profit']<0]['profit'].min()
        self.indicator['maxwincount']=self.result.result[self.result.result['profit']>0]['profit'].count()
        self.indicator['maxlosscount']=self.result.result[self.result.result['profit']<0]['profit'].count()
        return self.indicator
class BackTestDataProcess:
    #此类为数据处理类，输入参数为数据
    def __init__(self,data):
        self.data=data
    def run(self):
        self.data.run()
        self.data.data=self.data.data.set_index('date')
        return self.data.data
class BackTestStrategyProcess:
    #此类为策略处理类，输入参数为策略
    def __init__(self,strategy):
        self.strategy=strategy
    def run(self):
        self.strategy.run()
        #将策略索引设置为日期，如果已经设为日期则不需要此步骤
        if self.strategy.strategy.index.name!='date':
            self.strategy.strategy=self.strategy.strategy.set_index('date')
        return self.strategy.strategy
if __name__=='__main__':
    #股票代码，开始日期，结束日期，手续费单独输入，以备最后输出时调用
    stock_code='600000'
    start_date='2017-01-01'
    end_date='2017-12-31'
    fee=0.0003
    initial_capital=10000000
    strategy_name='xxx'
    bt=BackTest(stock_code,start_date,end_date,fee,initial_capital,strategy_name)
    indicator=bt.run()
    print(indicator.indicator)
    #输出为csv文件，文件名有股票，开始时间，结束时间，手续费组成，策略名称组成
    indicator.indicator.to_csv(stock_code+'_'+start_date+'_'+end_date+'_'+str(fee)+'_'+strategy_name+'.csv')
