#coding:utf-8
'''
 author:dabo
'''
import numpy as np
import pandas as pd
import tensorflow as tf
import os,random


def checkFile(input_dir):
    print("begin to check")
    category_dir = os.path.join(input_dir, "category")
    intensity_dir = os.path.join(input_dir, "intensity")
    pts_dir = os.path.join(input_dir, "pts")
    c1 = os.listdir(category_dir)
    i1 = os.listdir(intensity_dir)
    p1 = os.listdir(pts_dir)
    print(category_dir,intensity_dir,pts_dir)
    print(len(c1), len(i1), len(p1))
    #assert len(c1) == len(i1)
    #assert len(c1) == len(p1)
    count = 0
    for c in c1:
        if count % 1000 == 0:
            print(count)
        count += 1
        #print(count)
        if c in i1 and c in p1:
            pass
        else:
            print("error ", c)

        d_c1 = pd.read_csv(os.path.join(category_dir, c), header=None)
        d_i1 = pd.read_csv(os.path.join(intensity_dir, c), header=None)
        d_p1 = pd.read_csv(os.path.join(pts_dir, c), header=None)

        if len(d_c1) == len(d_i1) and len(d_c1) == len(d_p1):
            pass
        else:
            print("hang ", c)


mymodels = __import__("code_02_models")

pn = mymodels.PointNetSeg()

path = "../data/training"
# checkFile(path)
pn.train(path)
# pn.predict("../testdata/", "../answer/pointnet_10030111/")
