# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 18:07:32 2021

@author: patzo
"""
%reset
import pandas as pd
import numpy as np
from pulp import *

data=pd.read_excel("M:\OMSA\ISYE6501\HW11\diet.xls", header=0)
foods=data[:64]

#List of foods data
foods_data=foods.values.tolist()
#print(foods_data)

#list of food names
food_names=[x[0] for x in foods_data]
#print(food_names)

#cost of each food
cost=dict([(x[0],float(x[1]))for x in foods_data])

#Dictionary for rest of the nutrition types
nutrients=[]
for i in range (0,11):
    nutrients.append(dict([(x[0],float(x[i+3])) for x in foods_data]))


#min and max value dictionary
minvalue=data[65:66].values.tolist()
maxvalue=data[66:67].values.tolist()

#Create the optimization problem
prob=LpProblem('Diet_Problem',LpMinimize)

#Create variables
food_vars=LpVariable.dicts("Foods",food_names,0)
food_vars_selected=LpVariable.dicts("Foods_Selected",food_names,0,1,LpBinary)

#Objective Function
prob += lpSum([cost[x]*food_vars[x] for x in food_names])

#constraints
for i in range (0,11):
    prob += lpSum([nutrients[i][x]*food_vars[x] for x in food_names]) >= minvalue[0][i+3]
    prob += lpSum([nutrients[i][x]*food_vars[x] for x in food_names]) <= maxvalue[0][i+3]
    
#Solve Problem
soln=prob.solve()

print("Solution: ")
for var in prob.variables():
    if var.varValue > 0 and "food_select" not in var.name: 
        print(str(var.varValue)+" units of "+str(var).replace('Foods_','') )

print(f'{chr(10)}Total cost of food = ${round(value(prob.objective), 4)} ')
