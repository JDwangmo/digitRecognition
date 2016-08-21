# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-08-21'
    Email:   '383287471@qq.com'
    Describe:
"""
from __future__ import print_function


#encoding=utf8

step = 3

with open('M3_log.txt','r') as fout:
    for line in fout:
        line = line.strip()
        if step == 1:
            if line.startswith('num_filter1,num_filter2, hidden1,filter1,filter2,filter3 is '):
                print(line.replace('num_filter1,num_filter2, hidden1,filter1,filter2,filter3 is ','').replace('.',''))
        if step == 2:
            if line.startswith('测试结果汇总：') :
                print(line.replace('测试结果汇总：[','').replace(']',''))
        if step == 3:
            if line.startswith('验证中训练数据结果：'):
                print(line.replace('验证中训练数据结果：[','').replace(']',''))
