start = '2016-01-01'                       # 回测起始时间
end = '2017-01-01'                         # 回测结束时间
universe = DynamicUniverse('HS300')        # 证券池，支持股票和基金、期货
benchmark = 'HS300'                        # 策略参考基准
freq = 'd'                                 # 'd'表示使用日频率回测，'m'表示使用分钟频率回测
refresh_rate = 1                           # 执行handle_data的时间间隔

accounts = {
    'fantasy_account': AccountConfig(account_type='security', capital_base=10000000)
}

def initialize(context):                   # 初始化策略运行环境
    pass

def handle_data(context):                  # 核心策略逻辑
    account = context.get_account('fantasy_account')

 


start = '2015-01-01'
end = '2015-12-31'
benchmark = 'HS300'
universe = set_universe('A')    #股票池为 A 股所有股票。
captal_base = 100000
freq = 'd'
refresh_rate = 1

def initialize(account):
    pass

def handle_data(account):
    hist = account.get_attribute_history('closePrice', 3)  #取得股票池中所有股票前 3 天的收盘价（closePrice）。
    for s in account.universe:
        if hist[s][2] - hist[s][0] > 0.1 and s not in account.valid_secpos:     #hist[s][2] - hist[s][0]得到 1 天前和 3 天前收盘价的差值。
            order_pct(s, 0.05)                              #表示按账户当前总价值的百分比买入股票。
        elif hist[s][2] - hist[s][0] < -0.1 and s not in account.valid_secpos:   #account.valid_secpos是账户当前所持有的证券信息。
            order_to(s, 0)    #如果满足卖出条件则执行