# encoding: UTF-8

from copy import copy
from collections import OrderedDict

from vnpy.trader.vtConstant import *
from vnpy.trader.vtObject import VtTickData

from .omDate import getTimeToMaturity


# 常量定义
CALL = 1
PUT = -1


########################################################################
class OmInstrument(VtTickData):
    """交易合约对象"""

    #----------------------------------------------------------------------
    def __init__(self, contract, detail):
        """Constructor"""
        super(OmInstrument, self).__init__()
        
        # 初始化合约信息
        self.exchange = contract.exchange
        self.vtSymbol = contract.vtSymbol
    
        self.size = contract.size
        self.priceTick = contract.priceTick
        self.gatewayName = contract.gatewayName
        
        # 中间价
        self.midPrice = EMPTY_FLOAT
        
        # 持仓数据
        self.longPos = detail.longPos
        self.shortPos = detail.shortPos
        self.netPos = self.longPos - self.shortPos
        
    #----------------------------------------------------------------------
    def newTick(self, tick):
        """行情更新"""
        if not self.tickInited:
            self.date = tick.date
            self.openPrice = tick.openPrice
            self.upperLimit = tick.upperLimit
            self.lowerLimit = tick.lowerLimit
            self.tickInited = True
            
        self.lastPrice = tick.lastPrice
        self.volume = tick.volume
        self.openInterest = tick.openInterest
        self.time = tick.time
        
        self.bidPrice1 = tick.bidPrice1
        self.askPrice1 = tick.askPrice1
        self.bidVolume1 = tick.bidVolume1
        self.askVolume1 = tick.askVolume1

    #----------------------------------------------------------------------
    def newTrade(self, trade):
        """成交更新"""
        if trade.direction is DIRECTION_LONG:
            if trade.offset is OFFSET_OPEN:
                self.longPos += trade.volume
            else:
                self.shortPos -= trade.volume
        else:
            if trade.offset is OFFSET_OPEN:
                self.shortPos += trade.volume
            else:
                self.longPos -= trade.volume
                
        self.calculateNetPos()
       
    #----------------------------------------------------------------------
    def calculateNetPos(self):
        """计算净持仓"""
        newNetPos = self.longPos - self.shortPos
        
        # 检查净持仓是否发生变化
        if newNetPos != self.netPos:
            netPosChanged = True
            self.netPos = newNetPos
        else:
            netPosChanged = False
            
        return netPosChanged


########################################################################
class OmUnderlying(OmInstrument):
    """标的物"""

    #----------------------------------------------------------------------
    def __init__(self, contract, chainList):
        """Constructor"""
        super(OmUnderlying, self).__init__(contract)
        
        # 以该合约为标的物的期权链字典
        self.chainDict = OrderedDict((chain.symbol, chain) for chain in chainList)
        
        # 希腊值
        self.theoDelta = EMPTY_FLOAT    # 理论delta值
        self.posDelta = EMPTY_FLOAT     # 持仓delta值
        
    #----------------------------------------------------------------------
    def newTick(self, tick):
        """行情更新"""
        super(OmUnderlying, self).newTick(tick)
        
        self.theoDelta = self.size * self.midPrice / 100
        
        # 遍历推送自己的行情到期权链中
        for chain in self.chainList:
            chain.newUnderlyingTick()

    #----------------------------------------------------------------------
    def newTrade(self, trade):
        """成交更新"""
        super(OmUnderlying, self).newTrade(trade)
        self.calculatePosGreeks()
    
    #----------------------------------------------------------------------
    def calculatePosGreeks(self):
        """计算持仓希腊值"""
        self.posDelta = self.theoDelta * self.netPos    


########################################################################
class OmOption(OmInstrument):
    """期权"""

    #----------------------------------------------------------------------
    def __init__(self, contract, underlying, model, r):
        """Constructor"""
        super(OmOption, self).__init__(contract)
        
        # 期权属性
        self.underlying = underlying    # 标的物对象
        self.k = contract.strikePrice   # 行权价
        self.r = r                      # 利率
        
        if contract.optionType == OPTION_CALL:
            self.cp = CALL              # 期权类型        
        else:
            self.cp = PUT
            
        self.expiryDate = contract.expiryDate       # 到期日（字符串）
        self.t = getTimeToMaturity(self.expiryDate) # 剩余时间
    
        # 波动率属性
        self.bidImpv = EMPTY_FLOAT
        self.askImpv = EMPTY_FLOAT
        self.midImpv = EMPTY_FLOAT
    
        # 定价公式
        self.calculatePrice = model.calculatePrice
        self.calculateGreeks = model.calculateGreeks
        self.calculateImpv = model.calculateImpv
    
        # 模型定价
        self.pricingImpv = EMPTY_FLOAT
    
        self.theoPrice = EMPTY_FLOAT    # 理论价
        self.theoDelta = EMPTY_FLOAT    # 合约的希腊值（乘以了合约大小）
        self.theoGamma = EMPTY_FLOAT
        self.theoTheta = EMPTY_FLOAT
        self.theoVega = EMPTY_FLOAT
    
        self.posValue = EMPTY_FLOAT     # 持仓市值
        self.posDelta = EMPTY_FLOAT     # 持仓的希腊值（乘以了持仓）
        self.posGamma = EMPTY_FLOAT
        self.posTheta = EMPTY_FLOAT
        self.posVega = EMPTY_FLOAT 
        
        # 期权链
        self.chain = None
        
    #----------------------------------------------------------------------
    def calculateImpv(self):
        """计算隐含波动率"""
        underlyingPrice = self.underlying.midPrice
        if not underlyingPrice:
            return        
        
        self.askImpv = self.calculateImpv(self.askPrice1, underlyingPrice, self.k,
                                          self.r, self.t, self.cp)
        self.bidImpv = self.calculateImpv(self.bidPrice1, underlyingPrice, self.k,
                                          self.r, self.t, self.cp)        
        self.midImpv = (self.askImpv + self.bidImpv) / 2
    
    #----------------------------------------------------------------------
    def calculateTheoGreeks(self):
        """计算理论希腊值"""
        underlyingPrice = self.underlying.midPrice
        if not underlyingPrice or not self.pricingImpv:
            return
        
        self.theoPrice, self.theoDelta, self.theoGamma, self.theoTheta, self.theoVega = self.calculateGreeks(underlyingPrice, 
                                                                                                             self.k, 
                                                                                                             self.r, 
                                                                                                             self.t, 
                                                                                                             self.pricingImpv, 
                                                                                                             self.cp)
        
    #----------------------------------------------------------------------
    def calculatePosGreeks(self):
        """计算持仓希腊值"""
        self.posValue = self.theoPrice * self.netPos
        self.posDelta = self.theoDelta * self.netPos
        self.posGamma = self.theoGamma * self.netPos
        self.posTheta = self.theoTheta * self.netPos
        self.posVega = self.theoVega * self.netPos                
    
    #----------------------------------------------------------------------
    def newTick(self, tick):
        """行情更新"""
        super(OmOption, self).newTick(tick)
        self.calculateImpv()
    
    #----------------------------------------------------------------------
    def newUnderlyingTick(self):
        """标的行情更新"""
        self.calculateImpv()
        self.calculateTheoGreeks()
        self.calculatePosGreeks()
        
    #----------------------------------------------------------------------
    def newTrade(self, trade):
        """成交更新"""
        super(OmOption, self).newTrade(trade)
        self.calculatePosGreeks()


########################################################################
class OmChain(object):
    """期权链"""

    #----------------------------------------------------------------------
    def __init__(self, symbol, callList, putList):
        """Constructor"""
        self.symbol = symbol
        
        # 原始容器
        self.callDict = OrderedDict()
        self.putDict = OrderedDict()
        self.optionDict = OrderedDict()
        
        for option in callList:
            option.chain = self
            self.callDict[option.symbol] = option
            self.optionDict[option.symbol] = option
        
        for option in putList:
            option.chain = self
            self.putDict[option.symbol] = option
            self.optionDict[option.symbol] = option
        
        # 持仓数据
        self.longPos = EMPTY_INT
        self.shortPos = EMPTY_INT
        self.netPos = EMPTY_INT
        
        self.posValue = EMPTY_FLOAT
        self.posDelta = EMPTY_FLOAT
        self.posGamma = EMPTY_FLOAT
        self.posTheta = EMPTY_FLOAT
        self.posVega = EMPTY_FLOAT
    
    #----------------------------------------------------------------------
    def calculatePosGreeks(self):
        """计算持仓希腊值"""
        # 清空数据
        self.longPos = 0
        self.shortPos = 0
        self.netPos = 0
        self.posDelta = 0
        self.posGamma = 0
        self.posTheta = 0
        self.posVega = 0
        
        # 遍历汇总
        for option in list(self.optionDict.values()):
            self.longPos += option.longPos
            self.shortPos += option.shortPos
            
            self.posValue += option.posValue
            self.posDelta += option.posDelta
            self.posGamma += option.posGamma
            self.posTheta += option.posTheta
            self.posVega += option.posVega
        
        self.netPos = self.longPos - self.shortPos    
    
    #----------------------------------------------------------------------
    def newTick(self, tick):
        """期权行情更新"""
        option = self.optionDict[tick.symbol]
        option.newTick(tick)
    
    #----------------------------------------------------------------------
    def newUnderlyingTick(self):
        """期货行情更新"""
        for option in list(self.optionDict.values()):
            option.newUnderlyingTick()
            
        self.calculatePosGreeks()
        
    #----------------------------------------------------------------------
    def newTrade(self, trade):
        """期权成交更新"""
        option = self.optionDict[trade.symbol]
        
        # 缓存旧数据
        oldLongPos = option.longPos
        oldShortPos = option.shortPos
        
        oldPosValue = option.posValue
        oldPosDelta = option.posDelta
        oldPosGamma = option.posGamma
        oldPosTheta = option.posTheta
        oldPosVega = option.posVega
        
        # 更新到期权s中
        option.newTrade(trade)
        
        # 计算持仓希腊值
        self.longPos = self.longPos - oldLongPos + option.longPos
        self.shortPos = self.shortPos - oldShortPos+ option.shortPos
        self.netPos = self.longPos - self.shortPos
        
        self.posValue = self.posValue - oldPosValue + option.posValue
        self.posDelta = self.posDelta - oldPosDelta + option.posDelta
        self.posGamma = self.posGamma - oldPosGamma + option.posGamma
        self.posTheta = self.posTheta - oldPosTheta + option.posTheta
        self.posVega = self.posVega - oldPosVega + option.posVega


########################################################################
class OmPortfolio(object):
    """持仓组合"""

    #----------------------------------------------------------------------
    def __init__(self, name, underlyingList, chainList):
        """Constructor"""
        self.name = name
        
        # 原始容器
        self.underlyingDict = OrderedDict()
        self.chainDict = OrderedDict()
        self.optionDict = {}
        
        for underlying in underlyingList:
            self.underlyingDict[underlying.symbol] = underlying
            
        for chain in chainList:
            self.chainDict[chain.symbol] = chain
            self.optionDict.update(chain.callDict)
            self.optionDict.update(chain.putDict)
        
        # 持仓数据
        self.longPos = EMPTY_INT
        self.shortPos = EMPTY_INT
        self.netPos = EMPTY_INT
        
        self.posValue = EMPTY_FLOAT
        self.posDelta = EMPTY_FLOAT
        self.posGamma = EMPTY_FLOAT
        self.posTheta = EMPTY_FLOAT
        self.posVega = EMPTY_FLOAT
    
    #----------------------------------------------------------------------
    def calculatePosGreeks(self):
        """计算持仓希腊值"""
        self.longPos = 0
        self.shortPos = 0
        self.netPos = 0
        
        self.posValue = 0
        self.posDelta = 0
        self.posGamma = 0
        self.posTheta = 0
        self.posVega = 0
        
        for underlying in self.underlyingList:
            self.posDelta += underlying.posDelta
        
        for chain in self.chainList:
            self.longPos += chain.longPos
            self.shortPos += chain.shortPos
            
            self.posValue += chain.posValue
            self.posDelta += chain.posDelta
            self.posGamma += chain.posGamma
            self.posTheta += chain.posTheta
            self.posVega += chain.posVega
        
        self.netPos = self.longPos - self.shortPos        
    
    #----------------------------------------------------------------------
    def newTick(self, tick):
        """行情推送"""
        symbol = tick.symbol
        
        if symbol in self.optionDict:
            chain = self.optionDict[symbol].chain
            chain.newTick(tick)
            self.calculatePosGreeks()
        elif symbol in self.underlyingDict:
            underlying = self.underlyingDict[symbol]
            underlying.newTick(tick)
            self.calculatePosGreeks()
    
    #----------------------------------------------------------------------
    def newTrade(self, trade):
        """成交推送"""
        symbol = trade.symbol
        
        if symbol in self.optionDict:
            chain = self.optionDict[symbol].chain
            chain.newTrade(trade)
            self.calculatePosGreeks()
        elif symbol in self.underlyingDict:
            underlying = self.underlyingDict[symbol]
            underlying.newTrade(trade)
            self.calculatePosGreeks()