#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


# Lab 3: målt fart:

# # Mot radaren:
# v1 = [-0.39, -0.39, -0.40]
# v2 = [-1.11, -1.12, -1.11]
# v3 = [-2.05, -2.04, -2.04]

# # Fra radaren:
# v4 = [0.25, 0.25, 0.26]
# v5 = [0.85, 0.85, 0.85]
# v6 = [1.66, 1.68, 1.67]


# Lab 2: målte vinkler: 

xm120 = [-125.03, -125.03, -125.03, -125.03, -125.03]
xm90 = [-96.18, -91.07, -92.07, -92.07, -98.21]
xm65 = [-65.4, -65.21, -65.21, -65.21, -65.21]
xm35 = [-31.95, -37.74, -31.95, -37.74, -39.64]
x0 = [0, -3.54, -1.68, -1.68, -1.68]
x10 = [11.3, 11.3, 9.64, -11.93, -11.93]
x75 = [87, 81.05, 87, 79.11, 87]
x90 = [99.64, 91.95, 97.74, 95.82, 99.64]
x170 = [120, 120, 177.46, 173.19, 174.79]
x180 = [-178.37+360, 177.58, -161.58+360, 180, -178.43+360]

def Var(x):
    n = len(x)
    mu = 0
    var = 0
    for i in range(n):
        mu += x[i]/n
    for i in range(n):
        var += 1/(n-1)*(x[i]-mu)**2
    return var

def Mu(x):
    n = len(x)
    mu = 0
    var = 0
    for i in range(n):
        mu += x[i]/n
    return mu


# Lab 2:
print(f"Standardavvik: {np.sqrt(Var(xm120))}")
print(f"Standardavvik: {np.sqrt(Var(xm90))}")
print(f"Standardavvik: {np.sqrt(Var(xm65))}")
print(f"Standardavvik: {np.sqrt(Var(xm35))}")
print(f"Standardavvik: {np.sqrt(Var(x0))}")
print(f"Standardavvik: {np.sqrt(Var(x10))}")
print(f"Standardavvik: {np.sqrt(Var(x75))}")
print(f"Standardavvik: {np.sqrt(Var(x90))}")
print(f"Standardavvik: {np.sqrt(Var(x170))}")
print(f"Standardavvik: {np.sqrt(Var(x180))}\n")

print(Mu(xm120))
print(Mu(xm90))
print(Mu(xm65))
print(Mu(xm35))
print(Mu(x0))
print(Mu(x10))
print(Mu(x75))
print(Mu(x90))
print(Mu(x170))
print(Mu(x180), '\n')

print(abs(Mu(xm120)+120))
print(abs(Mu(xm90)+90))
print(abs(Mu(xm65)+65))
print(abs(Mu(xm35)+35))
print(abs(Mu(x0)-0))
print(abs(Mu(x10)-10))
print(abs(Mu(x75)-75))
print(abs(Mu(x90)-90))
print(abs(Mu(x170)-170))
print(abs(Mu(x180)-180))

Ref = [-120, -90, -65, -35, 0, 10, 75, 90, 170, 180]
m = [Mu(xm120), Mu(xm90), Mu(xm65), Mu(xm35), Mu(x0), Mu(x10), Mu(x75), Mu(x90), Mu(x170), Mu(x180)]

plt.scatter(m, Ref)
plt.plot(m, m, color='red')
plt.xlabel("Målinger målt med systemet [$^{\circ}$]", fontsize = 16)
plt.ylabel("Referansemålinger [$^{\circ}$]", fontsize = 16)
plt.legend(['Ideell linje', 'Målinger'])

# Lab 3:
# print(f"Standardavvik v1 m/s mot: {np.sqrt(Var(v1))}")
# print(f"Standardavvik v2 m/s mot: {np.sqrt(Var(v2))}")
# print(f"Standardavvik v3 m/s mot: {np.sqrt(Var(v3))}")
# print(f"Standardavvik v4 m/s fra: {np.sqrt(Var(v4))}")
# print(f"Standardavvik v5 m/s fra: {np.sqrt(Var(v5))}")
# print(f"Standardavvik v6 m/s fra: {np.sqrt(Var(v6))}")