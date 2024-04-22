# -*- coding: utf-8 -*-
"""
Created on Sat May 23 19:54:33 2020

@author: CQQ
"""
import numpy as np
from sklearn import preprocessing
import random
from scipy.spatial.distance import pdist,squareform
def Load_Data(filename):
    Datasets = np.loadtxt(filename,delimiter=',')
    [InstanceNum,AttributeNum] = Datasets.shape
    DataMatrix = Datasets[:,:AttributeNum-1]
    DataLabel = (Datasets[:,AttributeNum-1]).astype(int)
    return DataMatrix,DataLabel

def PreprocessAtribute(DataMatrix):
    """
    Parameters
    ----------
    DataMatrix:Data attribute matrix 

    Returns
    -------
    nor_mat:Using normalize method deal with attribute matrix
    map_mat:Using cca method deal with attribute matrix
    """ 
    #Normalization 0-1
    max_abs_scaler = preprocessing.MinMaxScaler()
    Normalization_Datamatrix = max_abs_scaler.fit_transform(DataMatrix)
    #Raise Attribute make norm in same hypersphere
    Diag_Matrix = np.diag(np.dot(DataMatrix,np.transpose(DataMatrix)))
    Norm_Max = max(Diag_Matrix)
    MapMatrix = np.c_[DataMatrix,np.sqrt(Norm_Max-Diag_Matrix)]
    return Normalization_Datamatrix,MapMatrix

def RecordIndexOfClass (DataLabel):
    NumOfClass = len(np.unique(DataLabel))
    Record_Matrix = []
    for i in range(1,NumOfClass+1):
        ClassIndex_i = np.where(DataLabel==i)
        ClassIndex_i = ClassIndex_i[0]
        Record_Matrix.append(ClassIndex_i)
    return Record_Matrix


def calculate(InputMatrix):
    X = pdist(InputMatrix,'euclidean')
    DistanceMatrix = squareform(X,force='no',checks=True)
    return DistanceMatrix

def AddNoise_N(Trainlabel,nr,Record_Matrix):
    IndexNoise,nc= set(),len(Record_Matrix)
    #list(Trainlabel)[:] can obtain a new list (deep copy)
    NoisedTrainlabel = np.array(list(Trainlabel)[:])
    for i in range(nc):
        nci = list(range(1,nc+1))
        ns = int(np.ceil(len(Record_Matrix[i])*nr))
        index_s = random.sample(list(Record_Matrix[i]),ns) 
        IndexNoise = IndexNoise.union(set(index_s))
        del nci[i]
        for j in range(ns):
            NoisedTrainlabel[index_s[j]] = random.sample(nci,1)[0]
    return  list(IndexNoise),NoisedTrainlabel  
def AddNoise(TrainMatrix,Trainlabel,nr,kdt,Record_Matrix):
    """
    Parameters
    ----------
    nr:Add noise ratio in original data
    kdt:KD-Tree bulid on original data matrix
    Record_Matrix:index of each class in train data

    Returns
    -------
    in:index of noise in original data matrix
    nl:add noise index of original data label
    """
    IndexNoise ,NK,nc= {},7,len(Record_Matrix)
    #list(Trainlabel)[:] can obtain a new list (deep copy)
    NoisedTrainlabel = np.array(list(Trainlabel)[:])
    ClassKind = range(1,nc+1) 
    for i in ClassKind:
        NumofAddNoise = np.ceil(len(Record_Matrix[int(i-1)])*nr)
        NumOfKCentre = int(np.ceil(NumofAddNoise/NK))
        n,k=0,0
        RemainIndex = np.delete(np.array(ClassKind),i-1)
        while n!=NumOfKCentre:
           ChoosedIndex = random.sample(list(Record_Matrix[i-1]),1) 
           ListOfKnn = kdt.query(TrainMatrix[ChoosedIndex[0]].reshape(1,-1),k=NK,return_distance=False)
           #ensure knn of a sample's label is same
           if len(np.unique(Trainlabel[ListOfKnn]))==1:
               ChangeClass= random.sample(list(RemainIndex),1)
               NoisedTrainlabel[ListOfKnn]=ChangeClass        
               IndexNoise = set(list(ListOfKnn)[0])|set(IndexNoise) 
               n = n + 1     
           else:
               k = k + 1
           if k>np.ceil(len(Trainlabel)/3):
               FinalIndexNoise=[]
               break
    FinalIndexNoise = list(IndexNoise)      
    return FinalIndexNoise,NoisedTrainlabel
def CompareNoiseLabel(sim,nl):
    """
    Parameters
    ----------
    sim:Distance of train data with sort index
    nl:Contain noise train label

    Returns
    -------
    CompareMatrix:第一列存放样本i同类样本个数(包括自己)，第二列之后存放样本i的下标，后面存档同类近邻样本
    """
    NumOfInstance = len(nl)
    CompareMatrix = np.zeros((NumOfInstance,NumOfInstance+1))
    for i in range(NumOfInstance):
        label_i = nl[i]
        tempk=0
        for j in range(0,NumOfInstance):
            if label_i == nl[sim[i][j]]:
                CompareMatrix[i][j+1]=sim[i][j]
                tempk = tempk + 1
            else:
                CompareMatrix[i][0] = tempk
                break
    CompareMatrix = CompareMatrix.astype(int)
    return  CompareMatrix 

def Indicator(IN_True,IN_Filter,Data_Label,Data,neigh,tree,svm,Data_Test,Data_TeL):
     presion = len(list(set(IN_True)&set(IN_Filter)))*1.0/len(IN_True)
     remove_non_noise = len(list(set(IN_Filter)-set(IN_True)))*1.0/len(list(set(range(len(Data_Label)))-set(IN_True)))
    
     FTr_I = list(set(range(len(Data_Label)))-set(IN_Filter))
     RemainMatrix,RemainLabel= Data[FTr_I],Data_Label[FTr_I]
     if len(set(RemainLabel))!=len(set(Data_Label)):
         ACC_Nei,ACC_Tree,ACC_SVM = 0,0,0
     else:
         IN_Class = RecordIndexOfClass(RemainLabel)
         min_num=IN_Class[0].size
         for i in range(1,len(IN_Class)):
             cl_num = IN_Class[i].size
             if min_num>cl_num:
                 min_num = cl_num
         if min_num<3:
             ACC_Nei,ACC_Tree,ACC_SVM = 0,0,0
         else:
             neigh.fit(RemainMatrix,RemainLabel)
             tree.fit(RemainMatrix,RemainLabel)
             svm.fit(RemainMatrix,RemainLabel)
             ACC_Nei,ACC_SVM,ACC_Tree = neigh.score(Data_Test,Data_TeL),svm.score(Data_Test,Data_TeL),tree.score(Data_Test,Data_TeL)
     return presion,remove_non_noise,ACC_Nei,ACC_Tree,ACC_SVM
 
   
def RCCA(CompareMatrix):
     """
    Parameters
    ----------
    CompareMatrix:same as above
         
    Returns
    -------
    RCCA:index of reduced
     """
     NumTrain = len(CompareMatrix)
     SparseMatrix = np.zeros((NumTrain,NumTrain)).astype(int)
     for i in range(NumTrain):
          SparseMatrix[i,CompareMatrix[i,1:CompareMatrix[i,0]+1]]=1
     SortCompareMatrix = CompareMatrix[np.argsort(CompareMatrix[:,0]),:][::-1]
     RCCA=[SortCompareMatrix[0][1]]
     for i in range(1,NumTrain):
          num_i = SortCompareMatrix[i][0]
          index = SortCompareMatrix[i,1:num_i+1]
          lenRCCA = len(RCCA)
          k=0
          for j in range(lenRCCA):
              Slist = SparseMatrix[RCCA[j]] 
              if sum(Slist[index])!=len(index):
                   k = k + 1
                   if k==lenRCCA:
                        RCCA.append(SortCompareMatrix[i][1])  
              else:
                   break
     return RCCA
#================================== SCCA ======================================      
#SCCA表示对选出来的噪声点进行集群查找
def SCCA(FiltedSet,CompareMatrix):
     NoiseLen = len(FiltedSet)  
     FinalNoiseSet = np.array([])
     for i in range(0,NoiseLen):
         TT = CompareMatrix[FiltedSet[i],1:CompareMatrix[FiltedSet[i]][0]+1]
         FinalNoiseSet = np.append(FinalNoiseSet,TT)
     NoiseSet = {}.fromkeys(list(FinalNoiseSet)).keys()
     return   NoiseSet