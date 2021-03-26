from WindPy import w
import numpy as np
import pandas as pd
from datetime import date, timedelta
import math
from statsmodels.regression.linear_model import OLS
from tqdm import tqdm


class YieldCurveInterpolator:
    def __init__(self, startdate: str, enddate: str):
        """
        初始化一个国债收益率变化插值器的实例
        :param str startdate: 开始日，形式YYYY-MM-DD
        :param str enddate: 截止日，形式YYYY-MM-DD
        """
        self.startdate = startdate
        self.enddate = enddate

        self.tenor = np.array([0, 1/12, 2/12, 3/12, 6/12, 9/12, 1,
                               2, 3, 4, 5, 6, 7, 8,
                               9, 10, 15, 20, 30, 40, 50])  # 中债发布国债收益率的期限
        self.tenorcodes = "M1004136,M1004677,M1004829,S0059741,S0059742,S0059743,S0059744,\
                          S0059745,S0059746,M0057946,S0059747,M0057947,S0059748,M1000165,\
                          M1004678,S0059749,S0059750,S0059751,S0059752,M1004711,M1000170"
        # 对应上面期限的国债收益率的Wind代码

        self.scyield = np.array(w.edb(self.tenorcodes, self.startdate, self.startdate, "Fill=Previous").Data[0])
        # 开始日的国债收益率曲线
        self.ecyield = np.array(w.edb(self.tenorcodes, self.enddate, self.enddate, "Fill=Previous").Data[0])
        # 截止日的国债收益率曲线
        self.diff = [self.tenor, self.ecyield - self.scyield]
        # 国债收益率变化

    def interpolate(self, tenor: float):
        """
        对某期限的国债收益率进行插值
        :param float tenor: 期限
        :return float: 插值结果，对应的国债收益率
        """
        if 1 <= tenor <= 10:
            # 中债给出了1年到10年所有整数期限的国债收益率，因此如果希望插值的期限在1年到10年之间，
            # 可以直接用tenor的整数部分做运算后当下标定位被用来插值的左右两个国债收益率
            tenorint = int(tenor)
            return (tenor - tenorint) * (self.diff[1][tenorint + 6] - self.diff[1][tenorint + 5]) +\
                self.diff[1][tenorint + 5]

        elif 0 <= tenor < 1:
            # 中债给出小于1年的各期限不太规则，相邻两个期限的差不等，无法直接用tenor的整数部分定位，需要搜索tenor落在哪个区间里
            for i in range(6):
                if self.diff[0][i] <= tenor <= self.diff[0][i + 1]:
                    return (tenor - self.diff[0][i]) / (self.diff[0][i + 1] - self.diff[0][i]) *\
                        (self.diff[1][i + 1] - self.diff[1][i]) + self.diff[1][i]

        elif 10 < tenor <= 50:
            # 类似地，大于10年的的各期限不太规则（由于15年的存在），需要搜索tenor落在哪个区间里
            for i in range(15, 20):
                if self.diff[0][i] <= tenor <= self.diff[0][i + 1]:
                    return (tenor - self.diff[0][i]) / (self.diff[0][i + 1] - self.diff[0][i]) *\
                        (self.diff[1][i + 1] - self.diff[1][i]) + self.diff[1][i]

        else:
            # 大于50年的无法插值
            raise ValueError("invalid tenor")


class DebtFund:
    def __init__(self, fcode: str, startdate: str, enddate: str, indicator: str, benchmarksheet: pd.DataFrame, ratedata=None, creditdata=None, per=0.25):
        """
        初始化一个基金或基准组合的实例
        :param str fcode: 基金或基准组合的Wind代码
        :param str startdate: 开始日，形式YYYY-MM-DD
        :param str enddate: 截止日，形式YYYY-MM-DD
        :param str indicator: 表示该实例是基金（fund）还是基准（benchmark）的指示
        :param pd.DataFrame benchmarksheet: 指数数据表格
        :param pd.DataFrame ratedata: 利率债基金对应指数（中债-总财富系列）的日度收益率数据
        :param pd.DataFrame creditdata: 信用债基金对应指数（中债-信用债总财富系列）的日度收益率数据
        :param float per: 归因频率，以年为单位，默认季度归因即取0.25
        """

        self.fcode = fcode
        self.startdate = startdate
        self.enddate = enddate
        self.per = per

        if indicator not in ['fund', 'benchmark']:
            raise ValueError('invalid indicator')
        else:
            self.indicator = indicator

        self.mod = None    # modified duration
        self.crate = None   # coupon rate
        self.startdirtyprice = None
        self.pval = None    # principal value
        self.totalret = None    # 区间收益率
        self.benchmark = None   # 基准指数的代码
        self.ftype = None   # fund type, 利率债基金或信用债基金

        self.ratedata = ratedata    # 利率债基金对应指数（中债-总财富系列）的日度收益率数据
        self.creditdata = creditdata    # 信用债基金对应指数（中债-信用债总财富系列）的日度收益率数据

        self.top5 = None
        self.benchmarksheet = benchmarksheet

        if self.indicator == "fund":
            self.getfunddata()

        self.calmod()
        self.calcrate()
        self.caltotalret()
        self.calstartdirtyprice()
        self.calpval()
        self.selectbenchmark() if self.indicator == "fund" else None

    def caltotalret(self):
        """
        从Wind获取区间收益率，基金为净值增长和现金分红（但不考虑再投资）收益率，基准为指数百分比变化
        """
        startdate = date.fromisoformat(self.startdate) + timedelta(days=1)
        # self.startdate和self.enddate都是季度末，在用Wind获取区间收益率时需要将self.startdate加一天，以季度第一天为起始日

        self.totalret = w.wss(self.fcode, "NAV_div_return" if self.indicator == "fund" else "pct_chg_per",
                              "startDate=%s;endDate=%s" % (startdate.isoformat().replace("-", ""),
                                                           self.enddate.replace("-", ""))).Data[0][0]

    def calstartdirtyprice(self):
        """
        从Wind获取买入价值，基金为单位净值，基准为指数收盘价
        """
        self.startdirtyprice = w.wsd(self.fcode, "nav" if self.indicator == "fund" else "close",
                                     self.startdate, self.startdate, "Fill=Previous").Data[0][0]

        newdate = self.startdate if not isinstance(self.startdirtyprice, float) else None
        # 有可能因为self.startdate并不是交易日或者其他原因没有数据，需要往前逐日尝试

        trial = 0

        while trial <= 7 and (not isinstance(self.startdirtyprice, float)):
            # 如果往前一周还没有数据，就停止再往前
            newdate = newdate[:8] + str(int(newdate[-2:])-1)
            # self.startdate都是季末，无需用生成date对象进行运算，直接不断减日即可
            self.startdirtyprice = w.wsd(self.fcode, "nav" if self.indicator == "fund" else "close",
                                         newdate, newdate, "Fill=Previous").Data[0][0]
            trial += 1

    def getfunddata(self):
        """
        对于基金，从Wind拉取前五大重仓券的名称、代码、数量、期初全价、久期、票息率等信息，并且计算市值权重、面值权重，用字典打包存储在对象属性中；
        同时，根据利率债占净值比例是否大于50%进行利率债基金或信用债基金的分类
        """
        date = self.startdate.replace("-", "")    # YYYY-MM-DD形式调整为YYYYMMDD
        names, tickers, quantities = [], [], []    # 重仓券名称，重仓券Wind代码，重仓券持有数量

        for i in range(1, 6):   # 前五大重仓券，i从1到5
            ntqdata = w.wss(self.fcode, "prt_topbondname,prt_topbondwindcode,prt_topbondquantity", "rptDate=%s;order=%d"
                            % (date, i)).Data

            if ntqdata[0][0] is None:
                continue    # 有可能重仓券个数不足5个

            names.append(ntqdata[0][0])
            tickers.append(ntqdata[1][0])
            quantities.append(ntqdata[2][0])

        names, tickers, quantities = np.array(names), np.array(tickers), np.array(quantities)
        latestpars = np.array(w.wss(",".join(tickers), "latestpar", "tradeDate=%s" % date).Data[0])    # 重仓券面值
        dirtyprices = []    # 重仓券期初全价
        mods = []   # 重仓券期初久期

        for ticker in tickers:
            dpdurdata = w.wsd(ticker, "dirty_cnbd,modidura_cnbd", self.startdate, self.startdate,
                              "credibility=1;Fill=Previous").Data
            dp = dpdurdata[0][0] if len(dpdurdata) == 2 else None
            dur = dpdurdata[1][0] if len(dpdurdata) == 2 else None

            newdate = self.startdate if (not isinstance(dp, float)) or (not isinstance(dur, float)) else None
            # 同理，有可能因为self.startdate并不是交易日或者其他原因没有数据，需要往前逐日尝试，但不超过往前一周

            trial = 0

            while ((not isinstance(dp, float)) or (not isinstance(dur, float))) and (trial <= 7):
                newdate = newdate[:8] + str(int(newdate[-2:]) - 1)
                dpdurdata = w.wsd(ticker, "dirty_cnbd,modidura_cnbd", newdate, newdate,
                                  "credibility=1;Fill=Previous").Data
                dp = dpdurdata[0][0] if len(dpdurdata) == 2 else None
                dur = dpdurdata[1][0] if len(dpdurdata) == 2 else None
                trial += 1

            if not isinstance(dp, float) or not isinstance(dur, float):
                dirtyprices.append(0)
                mods.append(0)

            else:
                dirtyprices.append(dp)
                mods.append(dur)

        dirtyprices = np.array(dirtyprices)
        mods = np.array(mods)

        crates = np.array([w.wss(ticker, "couponrate").Data[0][0] for ticker in tickers])   # 重仓券票息率
        mktval = quantities * dirtyprices   # 重仓券持有市值
        mktvalweights = mktval / mktval.sum()   # 重仓券市值权重
        parval = quantities * latestpars    # 重仓券持有面值
        parweights = parval / parval.sum()  # 重仓券面值权重

        top5tobond = w.wss(self.fcode, "prt_top5tobond", "rptDate=%s;order=5" % date).Data[0][0]
        # Wind给出的前五大重仓券占债券投资组合市值比例

        self.top5 = {"names": names, "tickers": tickers, "quantities": quantities, "latestpars": latestpars,
                     "dirtyprices": dirtyprices, "mods": mods, "crates": crates, "mktvalweights": mktvalweights,
                     "parweights": parweights, "top5tobond": top5tobond}

        temp = w.wss(self.fcode,
                     "prt_governmentbond,prt_centralbankbill,prt_pfbvalue",
                     "rptDate=%s" % date).Data  # 政府债券、央行票据、政策性金融债市值

        sum = 0

        for each in temp:
            if each[0] is not None:
                if not math.isnan(each[0]):
                    sum += each[0]

        value = w.wss(self.fcode, "prt_bondvalue", "rptDate=%s" % date).Data[0][0]    # 债券投资市值

        weight = sum / value
        self.ftype = "rate" if weight > 0.5 else "credit"   # 若利率债占比高于50%，归类为利率债基金，否则为信用债基金

    def calmod(self):
        """
        计算期初久期
        对于基金，如果前五大重仓券占债券投资比例大于30%，则用市值加权法计算组合久期，否则将其回归至中债-信用债总财富系列指数或中债-总财富系列指数，
        用回归系数加权指数久期来估计，除非估计值小于零，此时仍然用持仓法估计；对于基准，指数期初久期无需计算，直接从指数数据表格中读取即可
        """
        startdate = date.fromisoformat(self.startdate) + timedelta(days=1)
        # self.startdate和self.enddate都是季度末，在用Wind获取日度收益率序列时需要将self.startdate加一天，以季度第一天为起始日

        if self.indicator == "fund":
            if self.top5["top5tobond"] > 30:    # 前五大重仓券占债券投资比例大于30%，市值法计算组合久期
                self.mod = (self.top5["mktvalweights"] * self.top5["mods"]).sum()
            else:
                try:
                    Xdata = self.ratedata if self.ftype == "rate" else self.creditdata
                    fundret = w.wsd(self.fcode, "NAV_adj_return1", startdate.isoformat(),
                                self.enddate, "")
                    # 日期序列没有考虑现金分红但不考虑无再投资的收益率，只能用复权单位净值增长率近似代替
                    ydata = pd.DataFrame(fundret.Data[0], index=pd.to_datetime(fundret.Times))
                    coefs = pd.Series(OLS(ydata, Xdata).fit().params)
                    regmod = 0

                    for each in (Xdata.columns):
                        if each == "1":  # 此时对应的系数为常数项，不需要考虑，但大多数时候该常数项绝对值也很小，影响不大
                            continue
                        regmod += self.benchmarksheet.loc[each, 'mod'] * coefs[each]

                    self.mod = regmod if regmod > 0 else (self.top5["mktvalweights"] * self.top5["mods"]).sum()

                except:    # 净值法出于某种原因无法进行，则仍然用市值法
                    self.mod = (self.top5["mktvalweights"] * self.top5["mods"]).sum()

        elif self.indicator == "benchmark":
            self.mod = self.benchmarksheet.loc[self.fcode, 'mod']

    def calcrate(self):
        """
        计算期初票息率
        对于基金用面值加权法计算票息率，对于基准指数通过读取Wind下载下来的指数数据表格获取
        """
        if self.indicator == "fund":
            self.crate = (self.top5["parweights"] * self.top5["crates"]).sum()

        elif self.indicator == "benchmark":
            self.crate = self.benchmarksheet['crate'][self.fcode]

    def calpval(self):
        """
        计算买入价格对应的面值
        对于基金用持仓数据计算期初全价所对应的组合面值，对于基准指数用债券定价公式、票息率、剩余期限、期初全价、到期收益率倒推，
        这些数据通过读取Wind下载下来的指数数据表格获取
        """
        if self.indicator == "fund":    # 基金组合计算单位净值对应的五大重仓券的面值，并求和即可
            pval = 0
            for i in range(len(self.top5["dirtyprices"])):
                if self.top5["dirtyprices"][i] == 0:
                    continue
                else:
                    pval += self.startdirtyprice * self.top5["mktvalweights"][i] *\
                            self.top5['latestpars'][i] / self.top5["dirtyprices"][i]
            self.pval = pval

        else:   # 基准指数根据债券定价公式倒推

            ytm = self.benchmarksheet['ytm'][self.fcode]
            ttm = self.benchmarksheet['ttm'][self.fcode]
            N = int(ttm)
            m = ttm - N
            self.pval = self.startdirtyprice * ((1 + ytm / 100) ** m) / (((1 - self.crate / ytm)
                        / ((1 + ytm / 100) ** N)) + self.crate / ytm + self.crate / 100)

    def selectbenchmark(self):
        """
        对于基金，根据其久期和类型选择对应财富指数作为业绩基准，比如某基金被归类为信用债，久期为3.45年，则选择中债-信用债财富3-5年指数作为其基准
        """
        if self.ftype == "rate":
            if self.mod < 3:
                self.benchmark = "CBA00321.CS"  # 中债-总财富1-3年
            elif 3 <= self.mod < 5:
                self.benchmark = "CBA00331.CS"  # 中债-总财富3-5年
            elif 5 <= self.mod < 7:
                self.benchmark = "CBA00341.CS"  # 中债-总财富5-7年
            elif 7 <= self.mod < 10:
                self.benchmark = "CBA00351.CS"  # 中债-总财富7-10年
            else:
                self.benchmark = "CBA00361.CS"  # 中债-总财富10年以上
        elif self.ftype == "credit":
            if self.mod < 1:
                self.benchmark = "CBA02711.CS"  # 中债-信用债总财富1年以下
            elif 1 <= self.mod < 3:
                self.benchmark = "CBA02721.CS"  # 中债-信用债总财富1-3年
            elif 3 <= self.mod < 5:
                self.benchmark = "CBA02731.CS"  # 中债-信用债总财富3-5年
            elif 5 <= self.mod < 7:
                self.benchmark = "CBA02741.CS"  # 中债-信用债总财富5-7年
            elif 7 <= self.mod < 10:
                self.benchmark = "CBA02751.CS"  # 中债-信用债总财富7-10年
            else:
                self.benchmark = "CBA02761.CS"  # 中债-信用债总财富10年以上


class ReturnContributor:
    def __init__(self, fund: DebtFund, yieldchange: float, spreadchange=None):
        """
        初始化一个总收益分解的实例
        :param DebtFund fund: 输入一个fund对象
        :param float yieldchange: 对应的国债收益率变化
        :param spreadchange: 如果输入的是基金，还需要输入利差变化
        """
        if not isinstance(fund, DebtFund):
            raise TypeError("input object must be an instance of class DebtFund")
        self.mod = fund.mod
        self.yieldchange = yieldchange
        self.crate = fund.crate
        self.pval = fund.pval
        self.per = fund.per
        self.price = fund.startdirtyprice
        self.totalret = fund.totalret
        self.indicator = fund.indicator

        self.income = None
        self.treasury = None
        self.spread = None
        self.spreadchange = spreadchange
        self.selection = None

        if self.indicator == "fund":
            if self.spreadchange is None:
                raise ValueError("must provide spread change if not benchmark")

        self.calincome()
        self.caltreasury()
        self.calspread()
        self.calselection()

    def calincome(self):
        """
        计算收入效应
        """
        self.income = self.pval * self.crate * self.per / self.price

    def caltreasury(self):
        """
        计算国债效应
        """
        self.treasury = - self.mod * self.yieldchange

    def calspread(self):
        """
        计算利差效应
        """
        if self.indicator == "benchmark":
            self.spread = self.totalret - self.income - self.treasury
            self.spreadchange = - self.spread / self.mod
        elif self.indicator == "fund":
            self.spread = - self.spreadchange * self.mod

    def calselection(self):
        """
        计算择券效应
        """
        if self.indicator == "fund":
            self.selection = self.totalret - self.income - self.treasury - self.spread
        elif self.indicator == "benchmark":
            self.selection = 0


class ReturnAttributor:
    def __init__(self, fundc, bmkc):
        """
        初始化一个总收益分解的实例
        :param ReturnContributor fundc: 基金组合的总收益分解实例
        :param ReturnContributor bmkc: 基准指数的总收益分解实例
        """
        if not isinstance(fundc, ReturnContributor) or not isinstance(bmkc, ReturnContributor):
            raise TypeError("both inputs need to be instances of class ReturnContributor")

        if fundc.indicator != "fund" or bmkc.indicator != "benchmark":
            raise ValueError("1st input should be a FUND return contributor, 2nd should be a BENCHMARK")

        self.fundc = np.array([fundc.income, fundc.treasury, fundc.spread, fundc.selection])
        self.bmkc = np.array([bmkc.income, bmkc.treasury, bmkc.spread, bmkc.selection])
        self.alphas = self.fundc - self.bmkc

    def getfundc(self):
        """
        返回基金的四个效应和总收益率
        :return list: 四个效应和总收益率的list，总共5个元素
        """
        return [*list(self.fundc), self.fundc.sum()]    # 四个效应和总收益率

    def getbmkc(self):
        """
        返回基准的四个效应和总收益率
        :return list: 四个效应和总收益率的list，总共5个元素
        """
        return [*list(self.bmkc), self.bmkc.sum()]    # 四个效应和总收益率

    def getalphas(self):
        """
        返回基金相对基准超额收益率的四个效应和总阿尔法
        :return list: 四个效应和总阿尔法的list，总共5个元素
        """
        return [*list(self.alphas), self.alphas.sum()]    # 四个阿尔法和总阿尔法


class Campisi:
    def __init__(self, fcode: str, startdate: str, enddate: str, benchmarksheet_path: str):
        """
        初始化一个对单只基金单个季度截面做Campisi归因分析的实例
        :param str fcode: 基金或基准组合的Wind代码
        :param str startdate: 开始日，形式YYYY-MM-DD
        :param str enddate: 截止日，形式YYYY-MM-DD
        :param str benchmarksheet_path: 基准指数数据Excel文件路径
        """
        self.fcode = fcode
        self.startdate = startdate
        self.enddate = enddate
        sheetname = self.startdate[:4] + "Q" + str(int(self.startdate[5:7]) // 3)   # self.startdate对应的季度
        self.benchmarksheet = pd.read_excel(benchmarksheet_path, sheetname, index_col='code').dropna(axis=0)
        self.yc = YieldCurveInterpolator(self.startdate, self.enddate)
        self.ratedata = None
        self.creditdata = None
        self.result = []
        self.getregdata()

    def getregdata(self):
        """
        获取净值法回归需要用的指数日度收益率数据
        """
        startdate1 = date.fromisoformat(self.startdate) + timedelta(days=1)
        rateindices = ["CBA00321.CS", "CBA00331.CS", "CBA00341.CS", "CBA00351.CS", "CBA00361.CS"]
        creditindices = ["CBA02711.CS", "CBA02721.CS", "CBA02731.CS", "CBA02741.CS", "CBA02751.CS", "CBA02761.CS"]
        rateret = w.wsd(",".join(rateindices), "pct_chg", startdate1.isoformat(), self.enddate, "")
        creditret = w.wsd(",".join(creditindices), "pct_chg", startdate1.isoformat(), self.enddate, "")
        self.ratedata = pd.DataFrame(index=pd.to_datetime(rateret.Times), columns=["1"] + rateindices)
        for idx, code in enumerate(rateindices):
            self.ratedata[code] = rateret.Data[idx]
        self.ratedata["1"] = 1
        self.creditdata = pd.DataFrame(index=pd.to_datetime(creditret.Times), columns=["1"] + creditindices)
        for idx, code in enumerate(creditindices):
            self.creditdata[code] = creditret.Data[idx]
        self.creditdata["1"] = 1

    def runmodel(self):
        """
        进行Campisi归因分析
        """
        targetfund = DebtFund(fcode=self.fcode, startdate=self.startdate, enddate=self.enddate, indicator="fund",
                              benchmarksheet=self.benchmarksheet, ratedata=self.ratedata, creditdata=self.creditdata)
        if abs(targetfund.totalret) > 10:
            print("excluded b/c total return is > 10% or < -10%.")
            return None
        benchmark = DebtFund(fcode=targetfund.benchmark, startdate=self.startdate, enddate=self.enddate,
                             benchmarksheet=self.benchmarksheet, indicator="benchmark")
        benchmarkc = ReturnContributor(benchmark, yieldchange=self.yc.interpolate(benchmark.mod))
        targetc = ReturnContributor(targetfund, spreadchange=benchmarkc.spreadchange,
                                    yieldchange=self.yc.interpolate(targetfund.mod))
        campisi = ReturnAttributor(targetc, benchmarkc)
        self.result.append([*campisi.getalphas(), "alpha"])
        self.result.append([*campisi.getfundc(), "contribution"])
        self.result.append([*campisi.getbmkc(), "benchmark"])
        self.result = pd.DataFrame(self.result, columns=['income', 'treasury', 'spread', 'selection', 'total', 'type'])


class CampisiXSection:
    def __init__(self, startdate: str, enddate: str, benchmarksheet_path: str):
        """
        初始化一个对多只基金单个季度截面做Campisi归因分析的实例，与Campisi类的实例并不冲突，也没有包含关系，二者平行，取决于分析时需要用哪个
        :param str startdate: 开始日，形式YYYY-MM-DD
        :param str enddate: 截止日，形式YYYY-MM-DD
        :param str benchmarksheet_path: 基准指数数据Excel文件路径
        """
        self.startdate = startdate
        self.enddate = enddate
        self.yc = YieldCurveInterpolator(self.startdate, self.enddate)
        # 统一准备该季度的国债收益率曲线插值器

        sheetname = self.startdate[:4] + "Q" + str(int(self.startdate[5:7]) // 3)   # self.startdate对应的季度
        self.benchmarksheet = pd.read_excel(benchmarksheet_path, sheetname, index_col='code').dropna(axis=0)
        # 统一读取startdate两个指数系列的在startdate的数据

        self.ratedata = None
        self.creditdata = None
        self.getregdata()
        self.result = []

    def getregdata(self):
        """
        获取净值法回归需要用的指数日度收益率数据
        """
        startdate = date.fromisoformat(self.startdate) + timedelta(days=1)
        rateindices = ["CBA00321.CS", "CBA00331.CS", "CBA00341.CS", "CBA00351.CS", "CBA00361.CS"]
        creditindices = ["CBA02711.CS", "CBA02721.CS", "CBA02731.CS", "CBA02741.CS", "CBA02751.CS", "CBA02761.CS"]
        rateret = w.wsd(",".join(rateindices), "pct_chg", startdate.isoformat(), self.enddate, "")
        creditret = w.wsd(",".join(creditindices), "pct_chg", startdate.isoformat(), self.enddate, "")
        self.ratedata = pd.DataFrame(index=pd.to_datetime(rateret.Times), columns=["1"] + rateindices)
        for idx, code in enumerate(rateindices):
            self.ratedata[code] = rateret.Data[idx]
        self.ratedata["1"] = 1
        self.creditdata = pd.DataFrame(index=pd.to_datetime(creditret.Times), columns=["1"] + creditindices)
        for idx, code in enumerate(creditindices):
            self.creditdata[code] = creditret.Data[idx]
        self.creditdata["1"] = 1
        # 统一获取两个指数系列的日度收益率

    def runmodel(self, samplesheet_path: str):
        """
        进行Campisi归因分析
        :param str samplesheet_path: 样本基金名单Excel文件路径
        """
        samplelist = pd.read_excel(samplesheet_path, sheet_name='Sheet1', index_col=0).dropna(axis=0)

        for fcode in tqdm(samplelist.index):
            startyear = int(self.startdate[:4])
            startquarter = math.ceil(int(self.startdate[5:7]) / 3)
            incdate = samplelist.loc[fcode, 'incdate']
            incyear = incdate.year
            incquarter = math.ceil(incdate.month / 3)

            if incyear > startyear:
                continue
            elif incyear == startyear and incquarter > startquarter:
                continue
            else:
                try:
                    targetfund = DebtFund(fcode=fcode, startdate=self.startdate, enddate=self.enddate, indicator="fund",
                                      benchmarksheet=self.benchmarksheet, ratedata=self.ratedata, creditdata=self.creditdata)
                    if abs(targetfund.totalret) > 10:
                        # 设定了10%的阈值，剔除总收益率大于10%或小于-10%的结果，避免基金转型或大额申购赎回事件干扰结果
                        continue
                    benchmark = DebtFund(fcode=targetfund.benchmark, startdate=self.startdate, enddate=self.enddate,
                                         benchmarksheet=self.benchmarksheet, indicator="benchmark")
                    benchmarkc = ReturnContributor(benchmark, yieldchange=self.yc.interpolate(benchmark.mod))
                    targetc = ReturnContributor(targetfund, spreadchange=benchmarkc.spreadchange,
                                                        yieldchange=self.yc.interpolate(targetfund.mod))
                    campisi = ReturnAttributor(targetc, benchmarkc)
                    self.result.append([fcode, *campisi.getalphas(), self.enddate, "alpha"])
                    self.result.append([fcode, *campisi.getfundc(), self.enddate, "contribution"])
                except:
                    pass
        self.result = pd.DataFrame(self.result, columns=['fcode', 'income', 'treasury', 'spread', 'selection', 'total',
                                                         'enddate', 'type'])
        try:
            self.result.to_excel('分季度结果/' + self.enddate + '.xlsx', index=False)    # 结果Excel文件存入【分季度结果】文件夹
        except:
            pass
            # 保存为Excel部分用try except的考虑是to_excel语句中的路径有可能被调整为保存在某个子文件夹中而不是与Campisi.py同级，
            # 但此时若忘记事先创建文件夹，就会导致错误，可能丢失结果；用try except可以保证无论Excel导出是否成功，结果DataFrame都保留了


class CampisiTimeSeries:
    def __init__(self, startdate: str, enddate: str, benchmarksheet_path: str):
        """
        初始化一个对多只基金多个季度时间序列上做Campisi归因分析的实例，其中会使用到CampisiXSection类的实例
        :param str startdate: 开始日，形式YYYY-MM-DD
        :param str enddate: 截止日，形式YYYY-MM-DD
        :param str benchmarksheet_path: 基准指数数据Excel文件路径
        """
        self.startdate = startdate
        self.enddate = enddate
        self.benchmarksheet_path = benchmarksheet_path
        self.results = []   # list中每个元素是一个CampisiXSection类的实例，表示一个季度的业绩归因结果

    def gendatestring(self, i, k):
        if k == 1:
            return str(i) + "-03-31"
        elif k == 2:
            return str(i) + "-06-30"
        elif k == 3:
            return str(i) + "-09-30"
        elif k == 4:
            return str(i) + "-12-31"

    def runmodel(self, samplesheet_path: str):
        """
        进行Campisi归因分析
        :param str samplesheet_path: 样本基金名单Excel文件路径
        """
        startyear = int(self.startdate[:4])
        startquarter = int(self.startdate[5:7]) // 3
        endyear = int(self.enddate[:4])
        endquarter = int(self.enddate[5:7]) // 3
        if endyear > startyear:
            for i in range(startyear, endyear + 1):
                if i == startyear:
                    for k in range(startquarter, 5):
                        startdate = self.gendatestring(i, k)
                        enddate = self.gendatestring(i, k + 1) if k < 4 else self.gendatestring(i + 1, 1)
                        campisi = CampisiXSection(startdate, enddate, self.benchmarksheet_path)
                        campisi.runmodel(samplesheet_path)
                        self.results.append(campisi)
                elif i == endyear:
                    for k in range(1, endquarter):
                        startdate = self.gendatestring(i, k)
                        enddate = self.gendatestring(i, k + 1) if k < 4 else self.gendatestring(i + 1, 1)
                        campisi = CampisiXSection(startdate, enddate, self.benchmarksheet_path)
                        campisi.runmodel(samplesheet_path)
                        self.results.append(campisi)
                else:
                    for k in range(1, 5):
                        startdate = self.gendatestring(i, k)
                        enddate = self.gendatestring(i, k + 1) if k < 4 else self.gendatestring(i + 1, 1)
                        campisi = CampisiXSection(startdate, enddate, self.benchmarksheet_path)
                        campisi.runmodel(samplesheet_path)
                        self.results.append(campisi)
        elif startyear == endyear and startquarter < endquarter:
            for k in range(startquarter, endquarter):
                startdate = self.gendatestring(startyear, k)
                enddate = self.gendatestring(startyear, k + 1) if k < 4 else self.gendatestring(startyear + 1, 1)
                campisi = CampisiXSection(startdate, enddate, self.benchmarksheet_path)
                campisi.runmodel(samplesheet_path)
                self.results.append(campisi)


if __name__ == '__main__':
    io = r'债券指数数据/债券指数数据.xlsx'   # 基准指数数据
    iosample = r'基金份额处理/基金名单.xlsx'   # 纯债基金名单

    w.start()
    print(w.isconnected())

    a = CampisiTimeSeries(startdate="2017-06-30", enddate="2020-12-31", benchmarksheet_path=io)

    a.runmodel(samplesheet_path=iosample)
