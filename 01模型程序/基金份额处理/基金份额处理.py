import numpy as np
import pandas as pd

io = '样本总体.xlsx'

originallist = pd.read_excel(io, "Sheet1").dropna(axis=0)
rows = []

for i, each in enumerate(originallist['fname']):
    if each[-1] == "A":     # 如果基金为A份额
        name = each[:-1]    # name记录基金名称
        for k, eachtarget in enumerate(originallist['fname']):
            if eachtarget[:-1] == name and (eachtarget[-1] in ["B", "C", "D", "E", "H"]):
                # 同一基金的其他份额
                originallist.iloc[k, 1] = "drop"
                rows.append(k)
    if each[-2:] == "AB":   # 如果基金为AB份额
        name = each[:-2]    # name记录基金名称
        for k, eachtarget in enumerate(originallist['fname']):
            if eachtarget[:len(each)-2] == name and eachtarget[-1] in ["C", "D", "E", "H"]:      # 同一基金的C份额
                originallist.iloc[k, 1] = "drop"
                rows.append(k)

originallist.drop(labels=rows, axis=0, inplace=True)
originallist.to_csv('处理后样本总体.csv', index=False, encoding='utf_8_sig')
# to_csv之前仍然能够处理正常格式，但to_csv方法经常中文乱码，给定utf_8_sig编码一般可解决
