from algoqi.data import bond, common, mutual_fund, risk_model, stock
from collections import defaultdict
import numpy as np
import pandas as pd
import statsmodels.api as sm
from os import listdir

from algoqi.factor.components.file_downloader import download_file_from_next_cloud

download_file_from_next_cloud(url="http://file.orientlab.orientsec.com.cn:30080/s/qySaYwRsx8cLz88",
                              file_path='/tmp/wind_fund_class_data.xls')

def run(start_date: str, end_date: str):

    # fund_num 测试用 调节计算的基金数量
    fund_num = 100000

    # 一年中基金净值数据少于threshold_lenth则不采用
    threshold_lenth = 150

    # 股票仓位极差阈值
    threshold_range = 40.0

    # 债券仓位阈值
    threshold_bondtotot = 50.0

    threshold_stocktotot = 50.0

    # 市场指数的STD代码 中证全指：'000985.SSE' 
    Rm_index = '000985.SSE'

    # 跟踪基金beta的窗口长度
    track_slide_window = 60

    # 择时能力估计窗口长度
    TM_window_len = 250

    # 择时能力估计的模型，用60天后的数据
    TM_Rm_days = 60

    # 万德外部数据的相对路径
    wind_fund_data_xls_path = '/tmp/wind_fund_class_data.xls'

    # 需要计算的基金类别
    fund_type = ['普通股票型基金', '偏股混合型基金', '灵活配置型基金']

    bond_fund_type = ['中长期纯债型基金', '短期纯债型基金']

    currency_fund_type = ['货币市场型基金']

    def get_fund_pool(date='20200102', opt=1):
        """ 获取对应日期的基金池
        :param date<String>: 日期
        :param opt<Int>: 选项 1：主动权益 ,2：中长债和短债, 3：货基
        :return: 返回基金List
        """
        fund_class_data = pd.read_excel(wind_fund_data_xls_path)
        fund_class_data.columns = ['wind_code', 'fund_name', 'list_date', 
                                   'list_year', 'class_1', 'class_2', 
                                   'isinitial', 'dingkai', 'type_change']

        # 通过外部数据找到指定类型的基金，除去上市不足一年的基金，除去曾经转型的基金
        fund_class_data = fund_class_data[(fund_class_data['list_date'] + pd.Timedelta(days=365) 
                                           <= pd.to_datetime(date)) & 
                                          (fund_class_data['type_change'].isnull())]
        # 筛选三类基金
        if opt == 1:
            fund_class_data = fund_class_data[fund_class_data['class_2'].str.contains('|'.join(fund_type))]
        elif opt == 2:
            fund_class_data = fund_class_data[fund_class_data['class_2'].str.contains('|'.join(bond_fund_type))]
        elif opt == 3:
            fund_class_data = fund_class_data[fund_class_data['class_2'].str.contains('|'.join(currency_fund_type))]
        # 代码转换
        fund_pool = list(map(lambda x: x.replace('OF', 'MF'), 
                             fund_class_data['wind_code'].to_list()))
        if opt == 1:
            fund_pool = fund_filter(fund_pool, date)
#             print(f"fund numbers: {len(fund_pool)}")
        return fund_pool

    def fund_filter(all_fund, date):
        """ 过滤非主动权益性基金
        :param all_fund<Iterable>: 待过滤的基金池
        :param year<Int>: 年份
        :return: 返回主动权益性基金List
        """
        # 后续每日跟踪用这行
        df = mutual_fund.symbol_detail(all_fund)

        # 生成历史数据用这行
#         df = all_fund_data_memory.copy(deep=True)
#         df = df[df['symbol_standard'].isin(all_fund)]


        df['setup_date'] = pd.to_datetime(df['setup_date'])

        # 筛除停止运作的基金(df['maturity_date']字段的'nan'是字符串,2333)
        df = df[(df['maturity_date'] == 'nan') | 
                (pd.to_datetime(date) <= pd.to_datetime(df['maturity_date']))]

        # 计算股票历史仓位极差
        range_df = {'code': df['symbol_standard'].to_list(), 'range': [], 'drop': []}
        for fund in range_df['code']:
            # 获取全部历史年报, 剔除前4次建仓期的报告
            stktotot = mutual_fund.mutual_fund_asset_portfolio(symbols=fund, 
                                                               start_dt='19971231', 
                                                               end_dt=date)[['symbol', 'stocktotot', 'bondtotot']].iloc[5: , :].dropna()
            # 过去8个季度的债券持仓超过达到阈值，则筛除
            range_df['drop'].append(True if np.mean(stktotot['bondtotot'][-8: ]) > threshold_bondtotot else False)
            # 计算极差
            range_df['range'].append(np.abs(np.max(stktotot['stocktotot']) - np.min(stktotot['stocktotot'])))

        range_df = pd.DataFrame(range_df).dropna()
        range_df = range_df[(~range_df['drop']) & (range_df['range'] <= threshold_range)]

        return range_df['code'].to_list()


    def get_all_fund_data():
        """根据wind外部基金表，调取全体基金数据
        :return: 返回基金数据<DataFrame>
        """
        fund_class_data = pd.read_excel(wind_fund_data_xls_path)
        fund_class_data.columns = ['wind_code', 'fund_name', 'list_date', 
                                   'list_year', 'class_1', 'class_2', 
                                   'isinitial', 'dingkai', 'type_change']
        fund_class_data = fund_class_data[
            fund_class_data['class_2'].str.contains(
                '|'.join(fund_type + bond_fund_type + currency_fund_type)
            )
        ]

        # 代码转换 wind->std
        fund_pool = list(
            map(
                lambda x: x.replace('OF', 'MF'),
                fund_class_data['wind_code'].to_list()
            )
        )
        return mutual_fund.symbol_detail(fund_pool)
    # all_fund_data_memory = get_all_fund_data()


    def calc_bond_curr_idx_data():
        from datetime import datetime
        from scipy.stats.mstats import winsorize

        today = datetime.now().strftime('%Y%m%d')

        # 计算窗口的起始日期
        start_date = '20110901'
        end_date = common.prev_trading_day(
            date=today, 
            minus_num=TM_Rm_days)

        # 风险因子
        risk_factors = risk_model.risk_factor_return(
            start_dt=start_date, 
            end_dt=end_date, 
            risk_factors=['btop', 'sizenl', 'momentum']
        )
        risk_factors.columns = ['smb', 'hml', 'umd']

        # 无风险利率，十年期国债利率/365
        Rf = bond.yield_curve(start_dt=start_date, end_dt=today, curve_names = ['m3'])
        Rf.columns = ['Rf']
        
        Rf.loc[:, 'Rf'] = pd.to_numeric(Rf['Rf'],errors='coerce')
        Rf['Rf'] /= 365*100
        # 系统中
        Rf = Rf[~Rf.index.duplicated()]

        # 市场风险Rm
        Rm = stock.index_day_bars(
            index_symbol=Rm_index, 
            start_dt=start_date, 
            end_dt=today, 
            symbol_schema='STD'
        )[['close', 'pct_change']]
        Rm.columns = ['price', 'Rm']
        Rm['Rm'] /= 100
        Rm['price_after_60'] = Rm['price'].shift(-TM_Rm_days)
        Rm = Rm.dropna()
        Rm['Rm_1_60'] = Rm['price_after_60'] / Rm['price'] - 1

        bond_fund = get_fund_pool(end_date, opt=2)
        curr_fund = get_fund_pool(end_date, opt=3)

        yield_table = pd.DataFrame(
            index=pd.to_datetime(
                common.trading_days(start_date, end_date)
            )
        )

        # 债券利率因子
        for fund in bond_fund:
            nav_data = mutual_fund.fund_nav(symbols=fund, 
                                            start_dt=start_date, 
                                            end_dt=end_date)
            # 没有数据或者数据不足，则不计算
            if nav_data is None or len(nav_data) < threshold_lenth:
                continue
            # 计算收益率
            nav_data = nav_data[['adjusted_net_value']]
            nav_data['val_lag_1'] = nav_data['adjusted_net_value'].shift(1)
            nav_data = nav_data.dropna()
            nav_data[fund] = nav_data['adjusted_net_value'] / nav_data['val_lag_1'] - 1
            nav_data.index = pd.to_datetime(nav_data.index, format='%Y%m%d')

            # 左外连接
            yield_table = pd.merge(yield_table, nav_data[[fund]], how='left', left_index=True, right_index=True)
        yield_table['bund_factor'] = np.nanmean(yield_table, axis=1)
        interst_factor = yield_table['bund_factor']

        # 货基利率因子
        for fund in curr_fund:
            nav_data = mutual_fund.fund_nav(symbols=fund, 
                                            start_dt=start_date, 
                                            end_dt=end_date)
            # 没有数据或者数据不足，则不计算
            if nav_data is None or len(nav_data) < threshold_lenth:
                continue
            # 计算收益率
            nav_data = nav_data[['adjusted_net_value']]
            nav_data['val_lag_1'] = nav_data['adjusted_net_value'].shift(1)
            nav_data = nav_data.dropna()
            nav_data[fund] = nav_data['adjusted_net_value'] / nav_data['val_lag_1'] - 1
            nav_data.index = pd.to_datetime(nav_data.index, format='%Y%m%d')

            # 左外连接
            yield_table = pd.merge(yield_table, nav_data[[fund]], how='left', left_index=True, right_index=True)
        yield_table['curr_factor'] = np.nanmean(yield_table, axis=1)
        curr_fund_factor = yield_table['curr_factor']

        # 整合
        reg_data = Rf.merge(Rm[['Rm', 'Rm_1_60']], how='inner', 
                            left_index=True, 
                            right_index=True)
        reg_data = reg_data.merge(risk_factors, 
                                  how='inner', 
                                  left_index=True, 
                                  right_index=True)
        reg_data = reg_data.merge(interst_factor, 
                                  how='inner', 
                                  left_index=True, 
                                  right_index=True)
        reg_data = reg_data.merge(curr_fund_factor, 
                                  how='inner', 
                                  left_index=True, 
                                  right_index=True)
        reg_data = reg_data.astype('float')
        reg_data.dropna(inplace=True)
        reg_data['bund_factor'] = winsorize(reg_data['bund_factor'], limits=[0.001, 0.001])
        reg_data['curr_factor'] = winsorize(reg_data['curr_factor'], limits=[0.001, 0.001])

        reg_data['Rm-Rf'] = reg_data['Rm'] - reg_data['Rf']
        reg_data['Rm_1_60*Rm-Rf'] = reg_data['Rm-Rf'] * Rm['Rm_1_60']
        reg_data = reg_data[['Rm', 'Rm_1_60', 'smb', 'hml', 'umd', 'Rm-Rf', 'Rm_1_60*Rm-Rf', 'bund_factor', 'curr_factor']]

        return reg_data
    # reg_data_memory = calc_bond_curr_idx_data()
    # reg_data_memory


    def get_all_fund_nav():
        from datetime import datetime

        start_date = '20111231'
        end_date = datetime.now().strftime('%Y%m%d')

        tmp = pd.read_excel(wind_fund_data_xls_path)
        tmp.columns = ['wind_code', 'fund_name', 'list_date', 
                       'list_year', 'class_1', 'class_2', 
                       'isinitial', 'dingkai', 'type_change']
        tmp = tmp.loc[tmp['class_2'].str.contains('|'.join(fund_type)), 'wind_code'].to_list()
        # 代码转换 wind->std
        fund_pool = list(
            map(
                lambda x: x.replace('OF', 'MF'),
                tmp
            )
        )
        ret = pd.DataFrame(index=common.trading_days(start_date, end_date))

        for fund in fund_pool:
            nav_data = mutual_fund.fund_nav(symbols=fund, 
                                            start_dt=start_date, 
                                            end_dt=end_date)
            if nav_data is None:
                continue
            nav_data = nav_data[['adjusted_net_value']]
            nav_data['val_lag_1'] = nav_data['adjusted_net_value'].shift(1)
            nav_data = nav_data.dropna()
            nav_data[fund] = nav_data['adjusted_net_value'] / nav_data['val_lag_1'] - 1
            nav_data.index = pd.to_datetime(nav_data.index, format='%Y%m%d')

            ret = pd.merge(ret, nav_data[[fund]], how='left', left_index=True, right_index=True)
        ret_dic = dict()
        for fund in ret.columns:
            ret_dic[fund] = ret[fund]
        return ret_dic

    # fund_nav_data_memory = get_all_fund_nav()
    # fund_nav_data_memory

    # if 'fund_nav_data.csv' not in listdir('./'):
    #      pd.DataFrame(fund_nav_data_memory).to_csv('fund_nav_data.csv')

    # from sys import getsizeof
    # print(getsizeof(fund_nav_data_memory)/1024/2024, 'MB')

    def calc_bund_factor(date, opt=2, minus_num=0):
        '''计算给定日期的fund_factor或curr_factor

        '''
        # 获取基金池
        bond_fund = get_fund_pool(date, opt)

        # 初始化
        st_dt = common.prev_trading_day(date=date, minus_num=TM_window_len)
        yield_table = pd.DataFrame(
            index=pd.to_datetime(
                common.trading_days(st_dt, date)
            )
        )
        for fund in bond_fund:
            # 当年1月1日前一个交易日
            nav_data = mutual_fund.fund_nav(symbols=fund, 
                                            start_dt=st_dt, 
                                            end_dt=date)
            # 没有数据或者数据不足，则不计算
            if nav_data is None or len(nav_data) < threshold_lenth:
                continue
            # 计算收益率
            nav_data = nav_data[['adjusted_net_value']]
            nav_data['val_lag_1'] = nav_data['adjusted_net_value'].shift(1)
            nav_data = nav_data.dropna()
            nav_data[fund] = nav_data['adjusted_net_value'] / nav_data['val_lag_1'] - 1
            nav_data.index = pd.to_datetime(nav_data.index, format='%Y%m%d')

            # 左外连接
            yield_table = pd.merge(yield_table, nav_data[[fund]], how='left', left_index=True, right_index=True)
        yield_table = yield_table.fillna(yield_table.mean())

        if opt == 2:
            yield_table['bund_factor'] = yield_table.mean(axis=1)
            return yield_table[['bund_factor']]
        elif opt == 3:
            yield_table['curr_factor'] = yield_table.mean(axis=1)
            return yield_table[['curr_factor']]
        else :
            return None


    def get_variables(date):
        """ 获取对应年份的自变量
        :param date<String>: 日期
        :return: 返回自变量<DataFrame>

        """
        # 计算窗口的起始日期
        start_date = common.prev_trading_day(date=date, minus_num=TM_window_len)
        end_date = date

        # 风险因子
        risk_factors = risk_model.risk_factor_return(
            start_dt=start_date, 
            end_dt=end_date, 
            risk_factors=['btop', 'sizenl', 'momentum']
        )
        risk_factors.columns = ['smb', 'hml', 'umd']

        # 无风险利率，十年期国债利率/365
        Rf = bond.yield_curve(start_dt=start_date, end_dt=end_date, curve_names = ['y10'])
        Rf.columns = ['Rf']
        Rf = Rf.astype(np.float64)
        Rf['Rf'] /= 365*100

        # 市场风险Rm
        Rm = stock.index_day_bars(
            index_symbol=Rm_index, 
            start_dt=start_date, 
            end_dt=common.next_trading_day(
                date=date, 
                plus_num=TM_Rm_days), 
            symbol_schema='STD'
        )[['close', 'pct_change']]
        Rm.columns = ['price', 'Rm']
        Rm['Rm'] /= 100
        Rm['price_after_60'] = Rm['price'].shift(-TM_Rm_days)
        Rm = Rm.dropna()
        Rm['Rm_1_60'] = Rm['price_after_60'] / Rm['price'] - 1

        # 债券利率因子
        interst_factor = calc_bund_factor(date, opt=2, minus_num=0)

        # 货基利率因子
        curr_fund_factor = calc_bund_factor(date, opt=3, minus_num=0)

        # 整合
        reg_data = Rf.merge(Rm, how='inner', 
                            left_index=True, 
                            right_index=True)
        reg_data = reg_data.merge(risk_factors, 
                                  how='inner', 
                                  left_index=True, 
                                  right_index=True)
        reg_data = reg_data.merge(interst_factor, 
                                  how='inner', 
                                  left_index=True, 
                                  right_index=True)
        reg_data = reg_data.merge(curr_fund_factor, 
                                  how='inner', 
                                  left_index=True, 
                                  right_index=True)
        reg_data = reg_data.astype('float')
        reg_data['Rm-Rf'] = reg_data['Rm'] - reg_data['Rf']
        reg_data['Rm_1_60*Rm-Rf'] = reg_data['Rm-Rf'] * Rm['Rm_1_60']
        reg_data = reg_data[['Rf', 'Rm', 'Rm_1_60', 'smb', 'hml', 'umd', 'Rm-Rf', 'Rm_1_60*Rm-Rf', 'bund_factor', 'curr_factor']]

        return reg_data


    def get_timing_data(funds, data, date):
        """ 通过类似T-M模型的方式，得到基金经理择时能力数据
        :param funds<Iterable>: 基金池
        :param data<DataFrame>: 模型回归使用的数据
        :param date<String>: 日期
        :return: 返回择时能力数据<DataFrame>
        """

        # 计算窗口的起始日期
        start_date = common.prev_trading_day(date=date, minus_num=TM_window_len)
        end_date = date

        timing_data = []

        for fund_code in funds:
        #     print(fund_code)
            # 获取复权净值
            nav_data = mutual_fund.fund_nav(
                symbols=fund_code, 
                start_dt=start_date, 
                end_dt=end_date)
            # 如果没有净值数据，或者净值数据过少，则不予采用
            if nav_data is None or len(nav_data) <= threshold_lenth:
                continue
            nav_data = nav_data[['adjusted_net_value']]

            # 计算收益率
            nav_data['val_lag_1'] = nav_data['adjusted_net_value'].shift(1)
            nav_data = nav_data.dropna()
            nav_data['Ri'] = nav_data['adjusted_net_value'] / nav_data['val_lag_1'] - 1

            # 整合数据
            nav_data.index = pd.to_datetime(nav_data.index, format='%Y%m%d')
            merged_data = nav_data[['Ri']].merge(data, how='inner', left_index=True, right_index=True)
            # 回归
            Y = merged_data['Ri']
            X = merged_data[['Rm-Rf', 'Rm_1_60*Rm-Rf', 'smb', 'hml', 'umd', 'bund_factor', 'curr_factor']]
            X = sm.add_constant(X)
            reg_model = sm.OLS(Y, X)
            result = reg_model.fit() 
            timing_data.append({'code': fund_code, 'timing': result.params[2]})

        timing_df = pd.DataFrame(timing_data)
        # 计算分位数
        fractile = list(timing_df['timing'].quantile([.2, .4, .6, .8]))
    #     print(f'fractile of timing data: {fractile}')
        timing_df['group'] = -1
        timing_df['group'].astype(int)
        # 按分位数分组
        timing_df.loc[timing_df['timing'] <= fractile[0], 'group']= 1
        timing_df.loc[(timing_df['timing'] > fractile[0]) & (timing_df['timing'] <= fractile[1]), 'group'] = 2
        timing_df.loc[(timing_df['timing'] > fractile[1]) & (timing_df['timing'] <= fractile[2]), 'group'] = 3
        timing_df.loc[(timing_df['timing'] > fractile[2]) & (timing_df['timing'] <= fractile[3]), 'group'] = 4
        timing_df.loc[timing_df['timing'] > fractile[3], 'group'] = 5

    #     for i in range(1, 6):
    #         print(f"funds in group{i}: {len(timing_df[timing_df['group']==i])}")
    #         print(timing_df.loc[timing_df['group']==i, 'code'].to_list())
        return timing_df

    def track_funds(funds, group_id, date):
        """ 跟踪对应分组的基金，滑动窗口60日回归得到beta
        :param funds<Iterable>: 基金列表
        :param group_id<Int>: 分组id
        :param date<String>: 日期
        :return: beta数据
        """

        st_dt = common.prev_trading_day(date=date, minus_num=track_slide_window)

        # 按天更新用这个
        reg_factor = stock.index_day_bars(
            index_symbol=Rm_index, 
            start_dt=st_dt, 
            end_dt=date, 
            symbol_schema='STD'
        )[['pct_change']]
        reg_factor.columns = ['Rm']
        reg_factor['Rm'] /= 100

        # 债券利率因子
        interst_fct = calc_bund_factor(date=date, opt=2, minus_num=track_slide_window)

        # 货基利率因子
        curr_fund_fct = calc_bund_factor(date=date, opt=3, minus_num=track_slide_window)

        reg_factor = reg_factor.merge(interst_fct, 
                                      how='inner', 
                                      left_index=True, 
                                      right_index=True)
        reg_factor = reg_factor.merge(curr_fund_fct, 
                                      how='inner', 
                                      left_index=True, 
                                      right_index=True)
        risk_factors = risk_model.risk_factor_return(
            start_dt=st_dt, 
            end_dt=date,
            risk_factors=['btop', 'sizenl', 'momentum']
        )
        risk_factors.columns = ['smb', 'hml', 'umd']
        risk_factors = risk_factors.astype(np.float64)

        reg_factor = reg_factor.merge(risk_factors, 
                                      how='inner', 
                                      left_index=True, 
                                      right_index=True)
    #     # 集中计算用这行
    #     reg_factor = reg_data_memory.loc[st_dt: date, :]

        beta_data = dict()
        factor_name_list = ['Rm', 'smb', 'hml', 'umd', 'bund_factor', 'curr_factor']
        beta_data['col'] = factor_name_list
        for i in factor_name_list:
            beta_data[i] = pd.DataFrame(index=pd.to_datetime([date]))

        for fund_code in funds:
            for i in factor_name_list:
                beta_data[i][fund_code] = np.nan
            # 获取复权净值
    #         print(fund_code)
            nav_data = mutual_fund.fund_nav(symbols=fund_code, 
                                            start_dt=st_dt, 
                                            end_dt=date)
            # 如果没有净值数据，或者净值数据过少，则不予采用
            if nav_data is None or len(nav_data) <= track_slide_window * 0.6:
                continue

            nav_data = nav_data[['adjusted_net_value']]
            nav_data['val_lag_1'] = nav_data['adjusted_net_value'].shift(1)
            nav_data['Ri'] = nav_data['adjusted_net_value'] / nav_data['val_lag_1'] -1
            nav_data.index = pd.to_datetime(nav_data.index, format='%Y%m%d')
            # 整合数据
            data = nav_data[['Ri']].merge(reg_factor, how='inner', left_index=True, right_index=True).dropna()

            Y = data['Ri']
            X = data[['Rm', 'smb', 'hml', 'umd', 'bund_factor', 'curr_factor']]
            X = sm.add_constant(X)
            try:
                reg_model = sm.OLS(Y, X)
                result = reg_model.fit()
            except Exception as e:
                print((len(X), len(Y)))
                print(f"{fund_code} 数据缺失!")
                # 数据不足则删除
                del beta_data[fund_code]
            # 记录beta

            for j in range(len(factor_name_list)):
                beta_data[factor_name_list[j]].loc[beta_data[factor_name_list[j]].index[0], fund_code] = result.params[j+1]
        return beta_data


    def track_funds_with_memory(funds, group_id, date):
        """ 统一计算跟踪对应分组的基金，滑动窗口60日回归得到beta
        :param funds<Iterable>: 基金列表
        :param group_id<Int>: 分组id
        :param date<String>: 日期
        :return: beta数据
        """

        st_dt = common.prev_trading_day(date=date, minus_num=track_slide_window)

        # 集中计算用这行
        reg_factor = reg_data_memory.loc[st_dt: date, :]

        beta_data = dict()
        factor_name_list = ['Rm', 'smb', 'hml', 'umd', 'bund_factor', 'curr_factor']
        beta_data['col'] = factor_name_list
        for i in factor_name_list:
            beta_data[i] = pd.DataFrame(index=pd.to_datetime([date]))
    #     temp_data = fund_nav_data_memory.loc[st_dt: date, :]
        for fund_code in funds:
            for i in factor_name_list:
                beta_data[i][fund_code] = np.nan
            # 获取复权净值
            nav_data = fund_nav_data_memory[fund_code][st_dt: date].dropna()
            # 如果没有净值数据，或者净值数据过少，则不予采用
            if nav_data is None or len(nav_data) <= track_slide_window * 0.6:
                continue

            # 整合数据
            data = reg_factor.merge(nav_data, how='inner', left_index=True, right_index=True).dropna()

            Y = data[fund_code]
            X = data[['Rm', 'smb', 'hml', 'umd', 'bund_factor', 'curr_factor']]
            X = sm.add_constant(X)
            try:
                reg_model = sm.OLS(Y, X)
                result = reg_model.fit()
            except Exception as e:
                print((len(X), len(Y)))
                print(f"{fund_code} 数据缺失!")
                # 数据不足则删除
                del beta_data[fund_code]
            # 记录beta
            for j in range(len(factor_name_list)):
                beta_data[factor_name_list[j]].loc[beta_data[factor_name_list[j]].index[0], fund_code] = result.params[j+1]
        return beta_data


    def calc_weighted_beta(beta_lst, date):
        """ 计算加权beta
        :param funds<DataFrame>: 各个基金Beta时间序列数据
        :param date<String>: 日期
        :return<dict>: 分组的基金市值加权beta因子
        """
        ret = defaultdict(dict)
        for i in range(5):
            for factor in ['Rm', 'smb', 'hml', 'umd', 'bund_factor', 'curr_factor']:
                # 获取基金规模数据
                df = pd.DataFrame(index=beta_lst[i][factor].columns)
                for fund in beta_lst[i][factor].columns:
                    temp = mutual_fund.mutual_fund_asset_portfolio(symbols=fund, 
                                                                   start_dt=f'20101231', 
                                                                   end_dt=date)[['symbol', 'totalasset']].dropna().iloc[-1]
                    df.loc[fund, 'totalasset'] = temp['totalasset']
                df['weight'] = df['totalasset'] / np.sum(df['totalasset'])
                # 计算加权beta因子
                try:
                    ret[factor][i+1] = beta_lst[i][factor].fillna(value=0.0).dot(df['weight'].values)
                except ValueError as e:
                    print('Matrice ERROR!!!!!!!!!!!!')
                    print(e.args)
        #         print(ret[i+1])
        return ret

    def calc_average_beta(beta_lst, year):
        """ 计算等权beta
        :param funds<DataFrame>: 各个基金Beta时间序列数据
        :param date<String>: 日期
        :return<dict>: 分组的基金等权beta因子
        """
        ret = defaultdict(dict)
        for i in range(5):
            for factor in ['Rm', 'smb', 'hml', 'umd', 'bund_factor', 'curr_factor']:
                # 计算等权beta因子
                ret[factor][i+1] = beta_lst[i][factor].fillna(value=0.0).dot(np.ones(len(beta_lst[i][factor].columns)))/len(beta_lst[i][factor].columns)
    #             ret[factor][i+1] = np.nanmean(beta_lst[i][factor], axis=1).item()
        return ret


    def get_daily_beta_idx(date='20180102'):
        """计算给定日期的beta指数，每天计算60天前的基金池，计算择时能力并分组；再跟踪分组的beta

        """
        # 获取基金池
        fund_pool = get_fund_pool(date=date, opt=1)

        # 分组
        prev_60_date = common.prev_trading_day(date=date, minus_num=TM_Rm_days)
        prev_250_60_date = common.prev_trading_day(date=prev_60_date, minus_num=TM_window_len)

        # 按日更新，动态跟踪用这行
        data = get_variables(date=prev_60_date) # 获取估计窗口的数据

        # 集中计算用这个
    #     data = reg_data_memory.loc[prev_250_60_date: prev_60_date, :]

        timing_df = get_timing_data(fund_pool, data, prev_60_date)

        all_beta_data = []
            # 对5组基金进行跟踪
        for i in range(1, 6):
            all_beta_data.append(
    #             track_funds_with_memory(
    #                 timing_df.loc[timing_df['group']==i, 'code'].to_list(), i, date
    #             )
                track_funds(
                    timing_df.loc[timing_df['group']==i, 'code'].to_list(), i, date
                )
            )
    #         print(f"No.{i} group finished!")


        # 计算规模加权Betafactors
        weighted_beta_factor = calc_weighted_beta(all_beta_data, date)

        # 计算等权Betafactors
        average_beta_factor = calc_average_beta(all_beta_data, date)

        return weighted_beta_factor, average_beta_factor
    
    
    
    #先创建数据表
    ret_df = pd.DataFrame(columns=['date', 'symbol', 'value'])
    for trade_date in common.trading_days(start_date, end_date):
        
        try:
            w, a = get_daily_beta_idx(trade_date)
        except:
            continue
        for f in ['Rm', 'smb', 'hml', 'umd']:
            for i in range(1, 6):
                tmp = pd.DataFrame(
                    {
                        'date': trade_date, 
                        'symbol': f + f"_{i}", 
                        'value': a[f][i].item()
                    }, 
                    columns=['date', 'symbol', 'value'], 
                    index=[0])
                ret_df = ret_df.append(tmp)
                
    return ret_df



        