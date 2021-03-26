import pandas as pd
import numpy as np
from tqdm import tqdm


class CampisiResult:
    def __init__(self, file_path: str):
        """
        初始化一个Campisi模型结果对象
        :param file_path: Campisi模型多期分析结果Excel文件的路径
        """
        # --------------------------------------------------------------------------------------------------------------
        # 读取Excel文件及其中的各个工作表，每个工作表为一个季度的业绩归因分析结果
        self.file = pd.ExcelFile(file_path)
        self.sheets = dict()
        for sheet in self.file.sheet_names:
            self.sheets[sheet] = pd.read_excel(self.file, sheet_name=sheet)
            self.sheets[sheet] = self.sheets[sheet].loc[self.sheets[sheet]['type'] == 'contribution', :]
            # 先删去超额收益分解结果，只保留总收益分解结果
            self.sheets[sheet] = self.sheets[sheet][['fcode', 'selection', 'total']]
            # 只剩下代码、择券效应、总收益率
            self.sheets[sheet].set_index('fcode', inplace=True)
            # 将代码设置为行索引

        # --------------------------------------------------------------------------------------------------------------
        # 根据self.sheets中的各个表格，创建一个大表，用于统计排名
        index = set()
        for key, value in self.sheets.items():
            index = index | set(value.index.to_list())
        self.result = pd.DataFrame(index=index, columns=self.file.sheet_names + ['季度总数', '前1/3季度个数', '前1/3季度占比'])

    def rank(self, threshold: int):
        """
        先分季度计算截面上各基金的排名，再将结果整理到self.result中，并计算各基金排在前1/3的季度在其有数据的季度中的占比及排名
        :param int threshold: 对self.result中各个基金计算前1/3季度占比排名时要求至少要有多少个季度的数据
        """
        # --------------------------------------------------------------------------------------------------------------
        # 在self.sheets的各个表格中计算排名
        for key, value in self.sheets.items():
            value['季度排名'] = value['selection'].rank(method='max', ascending=False) / len(value['selection'])

        # --------------------------------------------------------------------------------------------------------------
        # 将各基金的季度排名结果输出到self.result中，并计算有数据的季度个数、其中排名在1/3的季度个数以及前1/3的季度占比
        for fund in tqdm(self.result.index):
            valid = 0
            total = 0
            for quarter in self.file.sheet_names:
                try:
                    self.result.loc[fund, quarter] = self.sheets[quarter].loc[fund, '季度排名']
                    total += 1
                    if self.result.loc[fund, quarter] < 1/3:    # 修改该参数即可调节排名的阈值，例如现在是统计基金排在前1/3的季度
                        valid += 1
                except:
                    self.result.loc[fund, quarter] = np.nan
            self.result.loc[fund, '季度总数'] = total
            self.result.loc[fund, '前1/3季度个数'] = valid
            if total >= threshold:
                self.result.loc[fund, '前1/3季度占比'] = valid / total

    def output(self, output_path: str):
        """
        将self.result输出成Excel文件
        :param str output_path: 输出结果Excel文件的路径
        """
        self.result.to_excel(output_path)


if __name__ == '__main__':
    a = CampisiResult(file_path='2020年.xlsx')   # file_path为整理好的单个Excel文件的路径
    a.rank(threshold=2)
    # threshold为至少需要有多少个季度的数据，即某基金有择券效应数据的季度数不足threshold，就不会对其计算排名在前1/3的季度所占其有数据季度的比例
    a.output(output_path='2020年结果.xlsx')    # output_path为希望输出结果Excel文件的路径

