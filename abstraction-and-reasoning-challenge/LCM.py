#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 22:09:24 2020

@author: aktasos
"""

def find_lcm(num1, num2): 
    if(num1>num2): 
        num = num1 
        den = num2 
    else: 
        num = num2 
        den = num1 
    rem = num % den 
    while(rem != 0): 
        num = den 
        den = rem 
        rem = num % den 
    gcd = den 
    lcm = int(int(num1 * num2)/int(gcd)) 
    return lcm 
      
l = [9, 9, 9, 9, 9, 6, 10, 10, 10, 20, 3, 3, 3, 9, 9, 21, 21, 21, 3, 3, 3, 7, 7, 7, 9, 10, 10, 20, 23, 23, 9, 9, 11, 11, 11, 11, 12, 12, 25, 23, 9, 8, 10, 9, 6, 9, 9, 9, 3, 3, 3, 3, 21, 21, 21, 18, 15, 14, 8, 8, 6, 8, 10, 10, 10, 4, 2, 5, 3, 3, 3, 11, 10, 9, 9, 8, 11, 19, 14, 16, 3, 3, 3, 3, 3, 10, 10, 10, 10, 10, 8, 3, 10, 10, 10, 10, 4, 3, 5, 4, 6, 5, 17, 17, 17, 9, 9, 9, 9, 10, 10, 10, 3, 3, 10, 10, 10, 5, 5, 5, 3, 3, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 7, 9, 9, 8, 9, 9, 1, 1, 1, 1, 1, 1, 3, 3, 2, 4, 2, 13, 11, 11, 6, 3, 6, 6, 11, 15, 12, 12, 3, 3, 3, 3, 3, 3, 3, 3, 30, 30, 30, 19, 14, 1, 1, 1, 1, 1, 1, 1, 6, 6, 6, 6, 8, 15, 13, 10, 11, 11, 11, 11, 11, 18, 18, 18, 18, 10, 10, 10, 10, 12, 12, 10, 12, 12, 12, 18, 2, 3, 5, 20, 10, 15, 3, 4, 2, 10, 10, 10, 10, 10, 17, 17, 17, 16, 16, 5, 5, 5, 5, 5, 5, 5, 30, 30, 30, 30, 13, 13, 13, 13, 13, 13, 18, 16, 14, 10, 10, 10, 3, 3, 3, 24, 27, 27, 7, 7, 10, 7, 8, 8, 8, 15, 3, 7, 30, 20, 10, 10, 10, 3, 3, 3, 3, 3, 5, 4, 8, 13, 13, 13, 13, 30, 20, 20, 20, 5, 7, 5, 20, 10, 14, 14, 14, 15, 15, 9, 9, 7, 7, 11, 11, 18, 19, 9, 16, 7, 12, 10, 10, 10, 2, 2, 2, 12, 12, 12, 12, 12, 12, 12, 1, 1, 1, 1, 1, 1, 9, 9, 13, 13, 13, 4, 6, 6, 10, 15, 20, 20, 20, 20, 10, 8, 6, 29, 29, 29, 3, 3, 3, 30, 10, 14, 3, 5, 4, 5, 5, 3, 1, 1, 4, 4, 4, 4, 14, 14, 12, 22, 20, 19, 12, 12, 12, 12, 15, 11, 13, 3, 3, 3, 13, 7, 7, 10, 10, 10, 10, 10, 10, 15, 15, 8, 5, 7, 11, 11, 11, 11, 15, 15, 15, 3, 3, 3, 3, 3, 16, 5, 5, 10, 8, 10, 11, 12, 18, 18, 18, 3, 3, 3, 3, 3, 3, 3, 10, 10, 10, 23, 23, 23, 12, 8, 9, 9, 9, 3, 3, 15, 15, 7, 6, 6, 6, 10, 10, 10, 4, 4, 4, 4, 13, 13, 16, 16, 3, 3, 3, 3, 3, 6, 4, 6, 8, 10, 10, 3, 3, 3, 3, 4, 7, 6, 4, 8, 6, 6, 6, 6, 6, 3, 3, 3, 15, 15, 15, 5, 5, 7, 10, 10, 15, 15, 19, 21, 22, 8, 5, 11, 10, 10, 22, 20, 17, 20, 20, 20, 11, 11, 11, 11, 6, 6, 6, 6, 20, 20, 20, 10, 8, 8, 3, 3, 3, 3, 3, 10, 10, 10, 10, 10, 3, 3, 4, 3, 3, 4, 6, 3, 3, 3, 3, 22, 12, 19, 4, 2, 4, 21, 21, 21, 10, 15, 18, 8, 5, 6, 1, 3, 4, 1, 1, 3, 3, 3, 3, 4, 4, 4, 4, 4, 9, 9, 9, 20, 20, 20, 20, 4, 2, 4, 2, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 23, 25, 21, 4, 3, 2, 6, 6, 6, 10, 10, 10, 23, 23, 23, 23, 14, 16, 13, 17, 7, 6, 6, 6, 9, 9, 9, 15, 15, 9, 8, 8, 8, 8, 24, 29, 19, 3, 5, 9, 10, 10, 10, 8, 8, 6, 6, 19, 13, 15, 15, 12, 6, 8, 10, 10, 10, 20, 11, 15, 6, 9, 8, 9, 7, 8, 2, 2, 2, 21, 21, 21, 14, 7, 11, 3, 3, 3, 4, 4, 4, 10, 10, 6, 3, 4, 11, 11, 11, 19, 12, 6, 9, 17, 9, 9, 9, 3, 2, 2, 10, 10, 10, 12, 6, 16, 6, 12, 15, 9, 18, 16, 16, 16, 9, 9, 13, 13, 14, 6, 6, 6, 10, 10, 10, 4, 4, 4, 4, 10, 10, 10, 3, 3, 3, 3, 10, 10, 15, 12, 14, 16, 8, 12, 8, 9, 8, 9, 14, 18, 19, 3, 3, 3, 3, 4, 4, 4, 4, 6, 3, 6, 5, 6, 5, 6, 3, 4, 4, 3, 19, 19, 19, 3, 4, 5, 3, 3, 3, 13, 15, 16, 4, 4, 3, 7, 9, 10, 8, 15, 16, 14, 3, 2, 1, 1, 2, 3, 2, 3, 4, 6, 6, 8, 10, 10, 8, 8, 12, 3, 8, 6, 4, 4, 9, 9, 30, 30, 30, 7, 9, 9, 4, 4, 4, 4, 4, 4, 5, 10, 10, 3, 3, 2, 10, 10, 10, 5, 3, 5, 3, 3, 3, 3, 3, 3, 3, 3, 9, 9, 18, 18, 18, 5, 5, 5, 5, 7, 7, 7, 9, 6, 6, 9, 15, 15, 15, 15, 3, 3, 3, 3, 3, 4, 4, 3, 10, 10, 10, 3, 3, 3, 3, 3, 3, 9, 9, 16, 4, 4, 6, 10, 10, 18, 16, 11, 11, 13, 15, 10, 20, 20, 20, 11, 11, 12, 9, 9, 10, 10, 10, 13, 18, 30, 20, 14, 24, 14, 16, 16, 16, 16, 3, 5, 5, 7, 6, 6, 9, 9, 12, 4, 3, 5, 1, 1, 1, 10, 11, 11, 13, 14, 13, 9, 7, 6, 10, 10, 6, 8, 10, 6, 6, 3, 3, 3, 3, 3, 3, 4, 2, 6, 6, 8, 6, 6, 3, 2, 3, 3, 7, 4, 3, 12, 12, 12, 19, 14, 15, 9, 9, 9, 16, 16, 16, 19, 9, 6, 4, 8, 5, 3, 5, 6, 4, 5, 7, 7, 7, 6, 6, 6, 12, 12, 11, 8, 6, 8, 8, 8, 9, 9, 9, 3, 3, 3, 9, 9, 4, 4, 4, 4, 5, 3, 3, 5, 9, 9, 9, 4, 4, 4, 4, 4, 3, 3, 3, 13, 13, 12, 14, 15, 4, 3, 2, 2, 2, 2, 6, 6, 6, 10, 12, 13, 7, 3, 5, 5, 10, 10, 10, 10, 10, 10, 12, 13, 14, 10, 10, 10, 3, 3, 3, 3, 3, 3, 3, 12, 11, 11, 10, 10, 5, 3, 3, 10, 10, 15, 2, 3, 1, 4, 15, 12, 11, 10, 10, 10, 10, 10, 10, 15, 15, 15, 3, 6, 7, 10, 10, 1, 1, 1, 1, 3, 3, 3, 3, 3, 7, 8, 10, 15, 20, 20, 9, 11, 12, 9, 5, 5, 5, 5, 8, 5, 3, 3, 5, 7, 10, 11, 3, 10, 10, 10, 1, 1, 1, 1, 10, 10, 10, 2, 3, 4, 11, 14, 17, 14, 14, 4, 4, 4, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 21, 11, 12, 4, 3, 4, 17, 10, 8, 19, 16, 17, 10, 10, 10, 10, 10, 14, 15, 16, 10, 10, 11, 11, 11, 6, 6, 10, 10, 10, 10, 3, 5, 7, 17, 17, 5, 3, 7, 6, 8, 9, 12, 13, 18, 12, 3, 3, 3, 3, 10, 10, 10, 12, 10, 12, 13, 17, 18, 8, 6, 6, 4, 4, 3, 3, 3, 3, 3, 14, 14, 16, 16, 6, 12, 4, 3, 5, 5, 15, 15, 15, 1, 1, 1, 1, 10, 10, 10, 1, 1, 1, 2, 1, 2, 3, 3, 3, 3, 3, 7, 7, 7, 10, 10, 10, 10, 5, 15, 15, 5, 3, 3, 3, 3, 3, 3, 3, 3, 5, 5, 5]



num1 = l[0] 
num2 = l[1] 
lcm = find_lcm(num1, num2) 
  
for i in range(2, len(l)): 
    lcm = find_lcm(lcm, l[i]) 
      
print(lcm) 