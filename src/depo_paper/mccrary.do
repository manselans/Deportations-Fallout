clear all
cap log close

* INSERT YOUR PATH TO DATA:
global path "Y:/Data/Workdata/703566/noa/projects/deportation_orders"

* load data created in analysis.py
use "$path/temp/mccrary.dta", clear

local tmp "Xj Yj r0 f_hat se_hat"
DCdensity seniority, breakpoint(0) gen(`tmp') b(1)

scalar theta_s = r(theta)
scalar se_s = r(se)
scalar t_s = theta_s / se_s 

clear

set obs 1
gen theta = theta_s
gen se = se_s
gen t = t_s

export delimited using "$path/output/tables/mccrary_results.csv"
erase "$path/temp/mccrary.dta"

exit, clear
