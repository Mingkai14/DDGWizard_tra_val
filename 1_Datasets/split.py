import math
import random

import xlrd
import xlwt

rb=xlrd.open_workbook('./data/raw_dataset_arranged.xls')
rs=rb.sheet_by_index(0)
nrow=rs.nrows
who_set=[]
tra_val_set=[]
tes_set=[]
for i in range(1,nrow):
    row=[]
    pdb=rs.cell_value(i,0)
    variation=rs.cell_value(i,1)
    chain=rs.cell_value(i,2)
    ddg=rs.cell_value(i,3)
    pH=rs.cell_value(i,4)
    T=rs.cell_value(i,5)
    row.append(pdb)
    row.append(variation)
    row.append(chain)
    row.append(ddg)
    row.append(pH)
    row.append(T)
    who_set.append(row)


random.seed(42)
random.shuffle(who_set)
tra_val_set=who_set[:math.ceil(len(who_set)/10*9)]
tes_set=who_set[math.floor(len(who_set)/10*9)+1:]


wb=xlwt.Workbook()
ws=wb.add_sheet('sheet1')
header=['PDB','Variation','Chain','ddG','pH','T']
for i in range(len(header)):
    ws.write(0,i,header[i])
for i in range(len(tra_val_set)):
    for j in range(len(tra_val_set[i])):
        ws.write(i+1,j,tra_val_set[i][j])
wb.save('./data/S7089.xls')

wb=xlwt.Workbook()
ws=wb.add_sheet('sheet1')
header=['PDB','Variation','Chain','ddG','pH','T']
for i in range(len(header)):
    ws.write(0,i,header[i])
for i in range(len(tes_set)):
    for j in range(len(tes_set[i])):
        ws.write(i+1,j,tes_set[i][j])
wb.save('./data/S787.xls')



