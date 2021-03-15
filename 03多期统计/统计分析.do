cd "E:\College\Gap\Huatai\债券基金Campisi\03多期统计"
use 归因分析结果数据.dta, clear

keep if type == "alpha"
sort type id time
xtset id time
xtreg total selection, fe
est store fe
xtreg total selection, re
est store re
hausman fe re
reg total selection