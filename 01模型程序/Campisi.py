from WindPy import w
import numpy as np
import pandas as pd
from datetime import date, timedelta
import math

io = "债券指数数据.xlsx"   # 基准指数数据
iosample = "样本总体.xlsx"   # 纯债基金名单

w.start()
print(w.isconnected())


class YieldCurveInterpolator:
    def __init__(self, startdate, enddate):     # 初始化输入YYYY-MM-DD格式起止日
        self.startdate = startdate
        self.enddate = enddate
        self.tenor = np.array([0, 1/12, 2/12, 3/12, 6/12, 9/12, 1,
                               2, 3, 4, 5, 6, 7, 8,
                               9, 10, 15, 20, 30, 40, 50])
        self.tenorcodes = "M1004136,M1004677,M1004829,S0059741,S0059742,S0059743,S0059744,\
                          S0059745,S0059746,M0057946,S0059747,M0057947,S0059748,M1000165,\
                          M1004678,S0059749,S0059750,S0059751,S0059752,M1004711,M1000170"
        self.scyield = np.array(w.edb(self.tenorcodes, self.startdate, self.startdate, "Fill=Previous").Data[0])
        self.ecyield = np.array(w.edb(self.tenorcodes, self.enddate, self.enddate, "Fill=Previous").Data[0])
        self.diff = [self.tenor, self.ecyield - self.scyield]

    def interpolate(self, tenor):   # interpolate方法输入以年为单位的期限，返回国债收益率变化值插值结果
        if 1 <= tenor <= 10:
            tenorint = int(tenor)
            return (tenor - tenorint) * (self.diff[1][tenorint + 6] - self.diff[1][tenorint + 5]) +\
                self.diff[1][tenorint + 5]
        elif 0 <= tenor < 1:
            for i in range(6):
                if self.diff[0][i] <= tenor <= self.diff[0][i + 1]:
                    return (tenor - self.diff[0][i]) / (self.diff[0][i + 1] - self.diff[0][i]) *\
                        (self.diff[1][i + 1] - self.diff[1][i]) + self.diff[1][i]
        elif 10 < tenor <= 50:
            for i in range(15, 20):
                if self.diff[0][i] <= tenor <= self.diff[0][i + 1]:
                    return (tenor - self.diff[0][i]) / (self.diff[0][i + 1] - self.diff[0][i]) *\
                        (self.diff[1][i + 1] - self.diff[1][i]) + self.diff[1][i]
        else:
            raise ValueError("invalid tenor")


class ReturnContributor:
    def __init__(self, fund, yieldchange, spreadchange=None):
        # 初始化输入一个fund对象，告诉是基准还是基金，如果是基金还需要输入利差变化
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
        self.income = self.pval * self.crate * self.per / self.price

    def caltreasury(self):
        self.treasury = - self.mod * self.yieldchange

    def calspread(self):
        if self.indicator == "benchmark":
            self.spread = self.totalret - self.income - self.treasury
            self.spreadchange = - self.spread / self.mod
        elif self.indicator == "fund":
            self.spread = - self.spreadchange * self.mod

    def calselection(self):
        if self.indicator == "fund":
            self.selection = self.totalret - self.income - self.treasury - self.spread
        elif self.indicator == "benchmark":
            self.selection = 0


class ReturnAttributor:
    def __init__(self, fundc, bmkc):
        if not isinstance(fundc, ReturnContributor) or not isinstance(bmkc, ReturnContributor):
            raise TypeError("both inputs need to be instances of class ReturnContributor")
        if fundc.indicator != "fund" or bmkc.indicator != "benchmark":
            raise ValueError("1st input should be a FUND return contributor, 2nd should be a BENCHMARK")
        self.fundc = np.array([fundc.income, fundc.treasury, fundc.spread, fundc.selection])
        self.bmkc = np.array([bmkc.income, bmkc.treasury, bmkc.spread, bmkc.selection])
        self.alphas = self.fundc - self.bmkc

    def getfundc(self):
        return [*list(self.fundc), self.fundc.sum()]

    def getbmkc(self):
        return [*list(self.bmkc), self.bmkc.sum()]

    def getalphas(self):
        return [*list(self.alphas), self.alphas.sum()]


class DebtFund:
    def __init__(self, fcode, startdate, enddate, indicator, benchmarksheet, ratedata=None, creditdata=None, per=0.25):
        self.fcode = fcode
        self.startdate = startdate
        self.enddate = enddate
        self.per = per
        self.indicator = indicator

        self.mod = None
        self.crate = None
        self.startdirtyprice = None
        self.pval = None
        self.totalret = None
        self.benchmark = None
        self.ftype = None

        self.ratedata = ratedata
        self.creditdata = creditdata

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
        startdate = date.fromisoformat(self.startdate) + timedelta(days=1)
        self.totalret = w.wss(self.fcode, "NAV_div_return" if self.indicator == "fund" else "pct_chg_per",
                              "startDate=%s;endDate=%s" % (startdate.isoformat().replace("-", ""),
                                                           self.enddate.replace("-", ""))).Data[0][0]

    def calstartdirtyprice(self):
        self.startdirtyprice = w.wsd(self.fcode, "nav" if self.indicator == "fund" else "close",
                                     self.startdate, self.startdate, "Fill=Previous").Data[0][0]
        newdate = self.startdate if not isinstance(self.startdirtyprice, float) else None
        trial = 0
        while trial <= 7 and (not isinstance(self.startdirtyprice, float)):
            newdate = newdate[:8] + str(int(newdate[-2:])-1)
            self.startdirtyprice = w.wsd(self.fcode, "nav" if self.indicator == "fund" else "close",
                                         newdate, newdate, "Fill=Previous").Data[0][0]
            trial += 1

    def getfunddata(self):
        date = self.startdate.replace("-", "")
        names, tickers, quantities = [], [], []
        for i in range(1, 6):
            ntqdata = w.wss(self.fcode, "prt_topbondname,prt_topbondwindcode,prt_topbondquantity", "rptDate=%s;order=%d"
                            % (date, i)).Data
            if ntqdata[0][0] is None:
                continue
            names.append(ntqdata[0][0])
            tickers.append(ntqdata[1][0])
            quantities.append(ntqdata[2][0])

        names, tickers, quantities = np.array(names), np.array(tickers), np.array(quantities)
        latestpars = np.array(w.wss(",".join(tickers), "latestpar", "tradeDate=%s" % date).Data[0])
        dirtyprices = []
        mods = []
        for ticker in tickers:
            dpdurdata = w.wsd(ticker, "dirty_cnbd,modidura_cnbd", self.startdate, self.startdate,
                              "credibility=1;Fill=Previous").Data
            dp = dpdurdata[0][0] if len(dpdurdata) == 2 else None
            dur = dpdurdata[1][0] if len(dpdurdata) == 2 else None
            newdate = self.startdate if (not isinstance(dp, float)) or (not isinstance(dur, float)) else None
            trial = 0
            while ((not isinstance(dp, float)) or (not isinstance(dur, float))) and (trial < 6):
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

        crates = np.array([w.wss(ticker, "couponrate").Data[0][0] for ticker in tickers])
        mktval = quantities * dirtyprices
        mktvalweights = mktval / mktval.sum()
        parval = quantities * latestpars
        parweights = parval / parval.sum()
        top5tobond = w.wss(self.fcode, "prt_top5tobond", "rptDate=%s;order=5" % date).Data[0][0]
        self.top5 = {"names": names, "tickers": tickers, "quantities": quantities, "latestpars": latestpars,
                     "dirtyprices": dirtyprices, "mods": mods, "crates": crates, "mktvalweights": mktvalweights,
                     "parweights": parweights, "top5tobond": top5tobond}
        temp = w.wss(self.fcode,
                     "prt_governmentbondtobond,prt_CDstonav,prt_centralbankbilltobond,prt_pfbtonav",
                     "rptDate=%s" % date).Data
        weight = 0
        for each in temp:
            if each[0] is not None:
                if not math.isnan(each[0]):
                    weight += each[0]
        self.ftype = "rate" if weight > 50 else "credit"

    def calmod(self):
        startdate = date.fromisoformat(self.startdate) + timedelta(days=1)
        if self.indicator == "fund":
            if self.top5["top5tobond"] > 30:
                self.mod = (self.top5["mktvalweights"] * self.top5["mods"]).sum()
            else:
                Xdata = self.ratedata if self.ftype == "rate" else self.creditdata
                fundret = w.wsd(self.fcode, "NAV_adj_return1", startdate.isoformat(),
                                self.enddate, "").Data[0]
                # 日期序列没有考虑现金分红但不考虑无再投资的收益率，只能用复权单位净值增长率近似代替
                ydata = pd.DataFrame(fundret)
                try:
                    XTX = np.dot(Xdata.T, Xdata)
                    XTy = np.dot(Xdata.T, ydata)
                    coefs = np.dot(np.linalg.inv(XTX), XTy)
                    if coefs[0][0] != coefs[0][0]:
                        print(self.fcode, "has", coefs[0][0], ", nav reg terminated.")
                        raise ValueError
                    regmod = 0
                    for idx, code in enumerate(list(Xdata)):
                        if code == "1":
                            continue
                        regmod += self.benchmarksheet['mod'][code] * coefs[idx][0]
                    self.mod = regmod if regmod > 0 else (self.top5["mktvalweights"] * self.top5["mods"]).sum()
                except:
                    self.mod = (self.top5["mktvalweights"] * self.top5["mods"]).sum()
        elif self.indicator == "benchmark":
            self.mod = self.benchmarksheet['mod'][self.fcode]

    def calcrate(self):
        if self.indicator == "fund":
            self.crate = (self.top5["parweights"] * self.top5["crates"]).sum()
        elif self.indicator == "benchmark":
            self.crate = self.benchmarksheet['crate'][self.fcode]

    def calpval(self):
        if self.indicator == "fund":
            pval = 0
            for i in range(len(self.top5["dirtyprices"])):
                if self.top5["dirtyprices"][i] == 0:
                    continue
                else:
                    pval += self.startdirtyprice * self.top5["mktvalweights"][i] *\
                            self.top5['latestpars'][i] / self.top5["dirtyprices"][i]
            self.pval = pval
        else:
            ytm = self.benchmarksheet['ytm'][self.fcode]
            ttm = self.benchmarksheet['ttm'][self.fcode]
            N = int(ttm)
            m = ttm - N
            self.pval = self.startdirtyprice * ((1 + ytm / 100) ** m) / (((1 - self.crate / ytm)
                        / ((1 + ytm / 100) ** N)) + self.crate / ytm + self.crate / 100)

    def selectbenchmark(self):
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


class Campisi:
    def __init__(self, fcode, startdate, enddate):
        self.fcode = fcode
        self.startdate = startdate
        self.enddate = enddate
        sheetname = self.startdate[:4] + "Q" + str(int(self.startdate[5:7]) // 3)
        self.benchmarksheet = pd.read_excel(io, sheetname, index_col='code').dropna(axis=0)
        self.yc = YieldCurveInterpolator(self.startdate, self.enddate)
        self.ratedata = None
        self.creditdata = None
        self.result = []
        self.getregdata()

    def getregdata(self):
        startdate1 = date.fromisoformat(self.startdate) + timedelta(days=1)
        rateindices = ["CBA00321.CS", "CBA00331.CS", "CBA00341.CS", "CBA00351.CS", "CBA00361.CS"]
        creditindices = ["CBA02711.CS", "CBA02721.CS", "CBA02731.CS", "CBA02741.CS", "CBA02751.CS", "CBA02761.CS"]
        rateret = w.wsd(",".join(rateindices), "pct_chg", startdate1.isoformat(), self.enddate, "").Data
        creditret = w.wsd(",".join(creditindices), "pct_chg", startdate1.isoformat(), self.enddate, "").Data
        self.ratedata = pd.DataFrame(columns=["1"] + rateindices)
        for idx, code in enumerate(rateindices):
            self.ratedata[code] = rateret[idx]
        self.ratedata["1"] = [1] * len(rateret[0])
        self.creditdata = pd.DataFrame(columns=["1"] + creditindices)
        for idx, code in enumerate(creditindices):
            self.creditdata[code] = creditret[idx]
        self.creditdata["1"] = [1] * len(creditret[0])

    def runmodel(self):
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
    def __init__(self, startdate, enddate):

        self.startdate = startdate
        self.enddate = enddate
        self.yc = YieldCurveInterpolator(self.startdate, self.enddate)
        # 统一准备该季度的国债收益率曲线插值器

        sheetname = self.startdate[:4] + "Q" + str(int(self.startdate[5:7]) // 3)
        self.benchmarksheet = pd.read_excel(io, sheetname, index_col='code').dropna(axis=0)
        # 统一读取startdate两个指数系列的在startdate的数据

        self.ratedata = None
        self.creditdata = None
        self.getregdata()
        self.result = []

    def getregdata(self):
        startdate = date.fromisoformat(self.startdate) + timedelta(days=1)
        rateindices = ["CBA00321.CS", "CBA00331.CS", "CBA00341.CS", "CBA00351.CS", "CBA00361.CS"]
        creditindices = ["CBA02711.CS", "CBA02721.CS", "CBA02731.CS", "CBA02741.CS", "CBA02751.CS", "CBA02761.CS"]
        rateret = w.wsd(",".join(rateindices), "pct_chg", startdate.isoformat(), self.enddate, "").Data
        creditret = w.wsd(",".join(creditindices), "pct_chg", startdate.isoformat(), self.enddate, "").Data
        self.ratedata = pd.DataFrame(columns=["1"] + rateindices)
        for idx, code in enumerate(rateindices):
            self.ratedata[code] = rateret[idx]
        self.ratedata["1"] = [1] * len(rateret[0])
        self.creditdata = pd.DataFrame(columns=["1"] + creditindices)
        for idx, code in enumerate(creditindices):
            self.creditdata[code] = creditret[idx]
        self.creditdata["1"] = [1] * len(creditret[0])
        # 统一获取两个指数系列的日度收益率

    def runmodel(self):
        samplelist = pd.read_excel(iosample, "Sheet2").dropna(axis=0)

        for idx, fcode in enumerate(samplelist['fcode']):
            startyear = int(self.startdate[:4])
            startquarter = math.ceil(int(self.startdate[5:7]) / 3)
            incdate = samplelist['incdate'][idx]
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
                    if targetfund.totalret > 10:
                        continue
                    benchmark = DebtFund(fcode=targetfund.benchmark, startdate=self.startdate, enddate=self.enddate,
                                         benchmarksheet=self.benchmarksheet, indicator="benchmark")
                    benchmarkc = ReturnContributor(benchmark, yieldchange=self.yc.interpolate(benchmark.mod))
                    targetc = ReturnContributor(targetfund, spreadchange=benchmarkc.spreadchange,
                                                        yieldchange=self.yc.interpolate(targetfund.mod))
                    campisi = ReturnAttributor(targetc, benchmarkc)
                    self.result.append([fcode, *campisi.getalphas(), self.enddate, "alpha"])
                    self.result.append([fcode, *campisi.getfundc(), self.enddate, "contribution"])
                    print("finished", idx)
                except:
                    print("looped", idx)
        self.result = pd.DataFrame(self.result, columns=['fcode', 'income', 'treasury', 'spread', 'selection', 'total',
                                                         'enddate', 'type'])
        self.result.to_csv(self.enddate+'.csv', index=False, encoding='utf-8')


class CampisiTimeSeries:
    def __init__(self, startdate, enddate):
        self.startdate = startdate
        self.enddate = enddate
        self.results = []

    def gendatestring(self, i, k):
        if k == 1:
            return str(i) + "-03-31"
        elif k == 2:
            return str(i) + "-06-30"
        elif k == 3:
            return str(i) + "-09-30"
        elif k == 4:
            return str(i) + "-12-31"

    def runmodel(self):
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
                        campisi = CampisiXSection(startdate, enddate)
                        campisi.runmodel()
                        self.results.append(campisi)
                elif i == endyear:
                    for k in range(1, endquarter):
                        startdate = self.gendatestring(i, k)
                        enddate = self.gendatestring(i, k + 1) if k < 4 else self.gendatestring(i + 1, 1)
                        campisi = CampisiXSection(startdate, enddate)
                        campisi.runmodel()
                        self.results.append(campisi)
                else:
                    for k in range(1, 5):
                        startdate = self.gendatestring(i, k)
                        enddate = self.gendatestring(i, k + 1) if k < 4 else self.gendatestring(i + 1, 1)
                        campisi = CampisiXSection(startdate, enddate)
                        campisi.runmodel()
                        self.results.append(campisi)
        elif startyear == endyear and startquarter < endquarter:
            for k in range(startquarter, endquarter):
                startdate = self.gendatestring(startyear, k)
                enddate = self.gendatestring(startyear, k + 1) if k < 4 else self.gendatestring(startyear + 1, 1)
                campisi = CampisiXSection(startdate, enddate)
                campisi.runmodel()
                self.results.append(campisi)


a = CampisiTimeSeries(startdate="2017-06-30", enddate="2020-09-30")
a.runmodel()
