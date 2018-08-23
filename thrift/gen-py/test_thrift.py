# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 17:00:21 2018

@author: ceedi
"""

from thrift.transport import TSocket  
from thrift.transport import TTransport  
from thrift.protocol import TBinaryProtocol  
from thrift.server import TServer  
import numpy as np
import json 
import pandas as pd
import pulp as pl
from skmultilearn.adapt import MLkNN
#根据实际的包结构去引入  
from test8 import TestService  

class tools:
    def jiageku_sql(jiageku):
        jiageku2=pd.DataFrame(columns=jiageku.columns)
        cc=pd.unique(jiageku['weizhi'])
        for piece in cc:
            d=jiageku[jiageku['weizhi']==piece]
            max_price=max(d['jiage'])
            min_price=min(d['jiage'])
            if len(pd.unique(d['jiage']))==1:
                for dd in d.index:
                    d.set_value(dd, 'xianxiaodu', 1)
            else: 
                for dd in d.index:
                    temp=(d.loc[dd,'jiage']-min_price)*0.3/(max_price-min_price)+0.7
                    d.set_value(dd, 'xianxiaodu', temp)            
            jiageku2=jiageku2.append(d)        
        return jiageku2

    def TrainProcess(train_data,train_target):
        cla= MLkNN(k=4, s=0.01)
        cla.fit(train_data, train_target)
        return cla
    
    def ChangeItemProcess(zhuanjia_new,laoren1):
        laoren1=laoren1.reshape((1,21))
        d=zhuanjia_new.columns
        Mask=pd.DataFrame(np.zeros((len(zhuanjia_new),zhuanjia_new.columns.size)),columns=d)
        SumofXWeight=pd.DataFrame(np.zeros((1,zhuanjia_new.columns.size)),columns=d)
        Y=pd.DataFrame(np.zeros((1,zhuanjia_new.columns.size)),columns=d)
        TopCapacityN = 3
        Mask[zhuanjia_new>=TopCapacityN]=1
        Mask[zhuanjia_new>=7]=7 
             
        #输入项大于等于5，认为该项能力弱  
        ImportantByInput = 3
        
        YY = pd.DataFrame(np.dot(laoren1,np.array(zhuanjia_new)),columns=d)  
        IndexYY = 0
        for row in range(0,len(zhuanjia_new)):    
            for clumn in range(0,zhuanjia_new.columns.size): 
                if(laoren1[0][row]>=ImportantByInput and Mask.iloc[row,clumn]>=1):
                    SumofXWeight.iloc[0,clumn] = SumofXWeight.iloc[0,clumn]+1
                    if(Mask.iloc[row,clumn]>=7):                     
                        Y.iloc[0,clumn]=Y.iloc[0,clumn]+1  #这里有改动，没有给汪
                        IndexYY =IndexYY+1
        sort_YY=YY.T.sort_values(by=0,ascending=False).T
        SumofXWeight1=SumofXWeight[SumofXWeight>0].dropna(axis=1,how='any') 
        sort_X=SumofXWeight1.T.sort_values(by=0,ascending=False).T
        sort_YY_index=list(sort_YY.columns)
    #    print(sort_YY_index)
        sort_X_index=list(sort_X.columns)             
        Y1=Y[Y>0].dropna(axis=1,how='any')
        Y_index=list(Y1.T.sort_values(by=0,ascending=False).T.columns) 
        sort_YY_X = [val for val in sort_YY_index[:6] if val in sort_X_index]
        ret_list = [item for item in Y_index if item not in sort_YY_X]  
        FinalChangeItem=sort_YY_X+ret_list
        return FinalChangeItem

    def ahp_qz(jiageku,kitchen,list1):
        if kitchen==1:
            sort=pd.Series(np.array([1,1,2,2,3,3,3,4,4,5]),index=['W01','K01','G01','E01','B01','B02','B03','L01','D01','T01'])
        else:
            sort=pd.Series(np.array([1,2,2,3,3,3,4,4,8,9]),index=['W01','G01','E01','B01','B02','B03','L01','D01','K01','T01'])
            
        step=0.7
        sort_c=pd.Series(np.linspace(1,len(list1),len(list1),dtype='int'),index=list1)
        
        jiageku2=pd.DataFrame(columns=jiageku.columns)
        for piece in list1:
            b=jiageku[jiageku['daxiang']==piece]
            jiageku2=jiageku2.append(b) 
            
        jiageku2=jiageku2.loc[:,['daxiang','kongjian','weizhi']]    
        result = jiageku2.drop_duplicates()
        result2=result.reset_index(drop = True)          
        c=list(pd.unique(result2['weizhi']))
        
        def ahp(a):
            b=np.array(a)
            [c,d]=np.linalg.eig(b)
            w=np.zeros((len(b),1))
            eigvalmag=np.imag(c)
            realeigval=c[abs(eigvalmag)<0.000001]
            maxeigval=max(realeigval)
            index=np.argwhere(c==maxeigval)
            vecinit=d[:,index[0][0]]  
            for i in range(0,len(b)):
                w[i][0]=np.real(vecinit[i])/np.sum(np.real(vecinit[:]))
            return w
            
        daxiang_ahp=pd.DataFrame(np.ones((len(c),len(c))),columns=c,index=c)
        kongjian_ahp=pd.DataFrame(np.ones((len(c),len(c))),columns=c,index=c)
        for pos1 in c:
            for pos2 in c:
                d1=list(result2[result2['weizhi']==pos1]['daxiang'])            
                d2=list(result2[result2['weizhi']==pos2]['daxiang'])
                e1=list(result2[result2['weizhi']==pos1]['kongjian'])
                e2=list(result2[result2['weizhi']==pos2]['kongjian'])
                temp1=step**(sort_c[d1[0]]-sort_c[d2[0]])
    #            print(temp1)
    #            daxiang_ahp.loc[pos1,pos2]=temp1
                daxiang_ahp.set_value(pos1,pos2,temp1)
                temp2=step**(sort[e1[0]]-sort[e2[0]])
                kongjian_ahp.set_value(pos1,pos2,temp2)
        daxiang_kongjian=pd.DataFrame(np.array([[1,3],[1/3,1]]))
        daxiang_kongjian_w=ahp(daxiang_kongjian)       
        daxiang_w=ahp(daxiang_ahp)
        kongjian_w=ahp(kongjian_ahp)
        hebing=np.append(daxiang_w,kongjian_w,axis=1)
        quanzhong=np.dot(hebing,daxiang_kongjian_w)    
        result2.insert(3,'quanzhong',quanzhong)
        result2.index = result2['weizhi'].tolist()
        jiageku3=pd.DataFrame(columns=jiageku.columns)
        for val in result2['weizhi']:
            d=jiageku[jiageku['weizhi']==val].copy()
            temp=result2.loc[val,'quanzhong']        
            d.set_value(d.index,'quanzhong',temp)
            jiageku3=jiageku3.append(d)
        jiageku3=jiageku3.reset_index(drop = True)
        return jiageku3
    
    def PriceCon(index_num,price,up):
        a=index_num
        j=0
        e=[0]
        ee=0
        for piece in a:
            b=price[price['daxiang']==piece]
            c=pd.unique(b['weizhi'])
            for jj in c:
                ee=ee+list(b['weizhi']).count(jj)
                e.append(ee)
            j=j+1
        
        result2=price
        cc1=result2.shape[0]
        a=[1]*(cc1)
        str1='x'
        for i in range(0,cc1):
            if(i<10):
                str2=str1+"0"+str(i)
                a[i]=str2
            else:
                str2=str1+str(i)
                a[i]=str2 
                
        prob = pl.LpProblem("The Whiskas Problem", pl.LpMaximize)
        x = pl.LpVariable.dicts("",a,0,1,pl.LpInteger)
        
        def get(s):
            return str(s)+'low'  
        name1 = list(map(get,a))
        name1 = dict(zip(a,name1))
        
        
        prob += pl.lpSum([result2.loc[i,'quanzhong']*result2.loc[i,'xianxiaodu']*x[a[i]] for i in result2.index]), "SumAbility"
        prob += pl.lpSum([result2.loc[i,'jiage']*x[a[i]] for i in result2.index]) <= up, "SumPrice"
        
        for i in range(0,len(e)-1):
            prob += pl.lpSum([x[a[j]] for j in range(e[i],e[i+1])])<=1, name1[a[i]]
        prob.solve() 
        
        kaiguan=[]
        for v in prob.variables():
            kaiguan.append(v.varValue==1)
        jieguo=result2[kaiguan]
        jieguo2=jieguo.loc[:,['daxiang','kongjian','weizhi','chanpin']]
        return jieguo2
    
    def tjsx(ChangePro,chanpinshuxing1,shengao):
        ChangePro['shuxing']=0
    
        ChangePro.set_index(["chanpin"], inplace=True)
#        print(chanpinshuxing)
        chanpinshuxing = chanpinshuxing1.copy()
        chanpinshuxing.set_index(["chanpin"], inplace=True)
    
        for piece in ChangePro.index:
            if piece in chanpinshuxing.index:
                cc=chanpinshuxing.loc[piece,shengao]
                ChangePro.loc[piece,'shuxing']=cc
    
        change_item1=ChangePro.reset_index()
        mid=change_item1["chanpin"]
        change_item1.drop(labels=["chanpin"],axis=1,inplace=True)    
        change_item1.insert(3,"chanpin",mid)
        return change_item1


#test8.thrift的具体实现  
class TestServiceHandler:  
    def __init__(self): 
        self.log = {}
        self.js = '{$$shengao$$:$$166-170cm$$,$$targetdata$$:[[0,1,1,1,1,1,0,1,0,1,0,1,1,1,1],[0,0,1,0,1,0,0,1,0,0,0,1,1,1,1],[0,1,1,0,1,0,0,1,0,0,1,0,1,1,0],[1,1,1,0,1,0,0,1,0,0,0,0,0,1,0],[1,1,1,0,1,0,0,0,0,0,0,0,1,0,0],[0,1,1,0,0,1,1,1,0,0,1,1,0,0,1],[1,1,1,0,1,1,1,1,1,0,0,1,1,0,0],[0,0,1,0,1,0,0,1,0,0,0,0,0,1,1],[1,1,1,0,1,0,0,1,0,0,0,0,0,1,1],[0,0,1,0,1,0,0,0,0,0,0,0,1,0,1],[0,0,1,0,1,1,0,1,0,1,1,1,0,0,1],[0,1,1,0,1,0,0,1,0,0,0,0,1,0,0],[0,1,1,0,1,0,0,1,0,0,0,0,1,1,1],[0,1,1,1,1,0,0,1,0,0,0,1,0,1,1],[0,1,1,0,0,0,0,1,1,1,0,0,0,1,1],[0,1,0,0,0,0,0,0,0,0,1,1,0,0,0],[1,1,1,0,0,0,0,1,1,1,1,1,0,0,1],[1,0,1,0,1,1,0,1,0,1,1,1,1,1,1],[0,1,1,0,0,0,0,1,0,1,0,0,0,0,1],[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,1,1,1,1,0,1,1,0,0,0,0,1,0],[1,0,1,0,1,0,0,1,0,0,1,1,0,0,1],[1,0,1,0,1,0,0,0,0,0,1,1,0,0,0],[1,1,1,1,0,0,0,1,0,1,1,1,1,1,1],[0,0,1,0,1,0,0,1,0,1,0,0,1,0,1],[0,1,1,0,1,1,1,1,1,0,1,1,0,0,1],[0,0,1,0,1,1,0,1,0,0,1,1,0,1,1],[1,0,1,0,1,0,0,1,0,0,0,0,0,0,1],[1,1,1,1,1,0,0,1,0,0,0,0,0,1,0],[1,1,1,0,1,1,0,0,0,0,0,0,1,0,0],[0,0,1,0,0,0,0,1,0,1,0,0,1,0,1],[0,0,1,0,1,0,0,1,1,1,1,1,1,0,1],[0,1,1,0,1,0,0,1,0,1,0,0,0,0,1],[0,1,1,1,1,1,0,1,0,1,0,0,0,1,1],[0,0,0,0,1,1,0,0,0,0,1,1,0,0,0],[0,0,0,0,1,1,0,0,0,0,1,1,0,0,0],[0,1,1,0,1,1,1,1,1,1,0,0,0,0,1],[1,1,1,1,0,0,0,1,0,1,1,0,0,1,1],[0,0,1,0,1,1,0,1,0,0,1,1,1,1,1],[1,0,1,0,1,1,0,1,0,1,1,1,1,0,1],[0,0,1,0,1,0,0,0,0,1,0,0,0,1,1],[0,0,1,0,1,1,0,1,0,0,0,1,0,0,1],[0,1,1,0,1,1,0,1,0,1,0,1,1,0,1],[0,0,1,0,0,0,0,1,0,0,0,0,0,1,0],[0,0,1,0,0,1,1,1,1,1,0,0,0,0,1],[0,1,1,0,1,1,0,1,0,1,0,1,0,0,1],[0,0,1,0,0,0,0,1,0,1,0,0,0,1,0],[0,1,1,0,0,1,0,1,0,1,0,1,0,0,0],[0,0,1,0,0,0,0,1,0,0,0,1,0,0,0],[0,1,1,0,1,1,0,1,0,0,0,1,1,1,1],[0,1,1,0,0,0,0,1,0,1,0,1,0,0,1],[0,0,1,0,0,0,0,1,0,0,0,0,0,0,1],[0,1,1,1,1,0,0,1,1,1,0,1,0,0,0],[0,0,1,0,1,0,0,1,0,1,1,1,0,0,1],[1,1,1,0,0,0,0,1,0,1,1,1,0,1,1],[0,0,1,0,1,1,0,1,0,1,0,1,0,1,1],[0,0,1,1,1,1,1,1,1,1,0,0,1,1,1],[0,0,1,0,0,0,0,1,0,1,0,1,0,1,1],[1,0,1,0,1,1,0,1,1,0,0,0,0,0,1],[0,0,1,0,0,0,0,1,0,1,0,1,0,1,1],[0,0,1,0,0,0,0,1,0,1,0,0,0,1,1],[0,0,1,0,0,0,0,0,0,0,0,0,0,1,1],[0,0,1,0,1,1,1,1,1,1,0,0,0,1,1],[0,0,1,0,1,1,1,1,1,1,0,0,0,1,1],[0,0,1,0,1,1,0,1,0,0,1,1,0,1,0],[1,1,1,0,1,1,0,1,0,0,0,0,0,1,1],[0,0,1,0,0,0,0,1,0,1,0,1,0,1,1],[0,0,1,0,1,1,1,1,1,1,0,1,1,1,1],[0,0,1,0,1,1,1,1,0,0,0,1,1,1,1],[0,0,1,0,0,0,0,1,1,1,0,1,0,1,1]],$$traindata$$:[[0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0],[0,1,1,0,1,0,0.6,0,1,0,0,0.6,1,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,1,0,0,0,0.6,0,0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,1,1],[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0],[1,0,0,0,1,1,1,1,0,0,0,0,0,0,1,0,0,0,0,1,1],[0,0,1,1,0,0,0,0.6,0,0,0,0,0,0,0,0,1,0,0,0,0],[0,0,1,1,0,0,0,0.6,0,0,1,0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0.6,0,0,0,0,1,0,0,0,0,0,0,0,1],[0,0,0.5,0,0.5,0,0,0.5,0,1,0,0.3,0.5,0,0,0,0,0,0.5,0.5,0.5],[0,0,0,0,1,0,0,0,0,0,0,0.6,1,0,0,0,1,0,0,0,1],[0,0,1,0,0,0,0,0.6,0,0,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0.3,0,1,0,0,0.5,0,0,0,0,0,0,0,0.5],[0,1,1,1,1,1,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],[0,0,0.5,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0],[0,0,0,0.5,0,0,0,0.3,0,0,0,0,1,0,0,0,0,0,0,0,1],[0,0,0.5,0.5,1,0,0,0.5,0,0,0,0.3,1,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,1],[0,0,0,0,0,0,0,0.3,0,0,0,0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0.3,0,0,0,0,0,0,0,0,0,0,0,1,0],[0,0,0.5,0.5,1,0.5,0.3,0.3,0,1,0,0,1,0,1,0,0,0,0,0,1],[0,0,0,0,0,0,0,0.3,0,0,0,0.3,1,0,0,0,0,0,0,0,0],[0,0,0.5,0,0,0,0,0,0,0,0,0.3,0,0,0,0,0,0,0,1,1],[0,0,0.5,0.5,0,0,0,0.3,0,0,0,0,1,0,0,0,0,0,0,0,0],[0.5,0.5,0.5,0.5,1,0.5,0,0.7,0,0,0,0,0,0,1,0,0,0,0,1,1],[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],[0.5,0,0,0,0,0,0,0,0,0,0.5,0,0,1,0,0,0,0,0,0,1],[0,0,0,0,0.5,0,0,0.3,0,0,0,0,1,0,1,0,1,0,0,0,0],[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0.3,0,0,0,0,1,0,0,0,1,1,0,1,0],[0,0,0,0.5,0,0,0,0.3,0,0,0,0,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0.3,0,0,0,0,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0],[0,0,0.5,0.5,0.5,0.5,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1],[0.5,0.5,0.5,0.5,1,0.5,0.3,0,0,1,0.5,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0.5,0,0,0,0,0,0,0,0,0,1,0,0,1],[0,0,0,0,0,0,0,0.7,0,0,0,0,1,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0.7,0,0,0,0,1,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0.7,0,0,0,0,0,0,0,0,0,0,0,1,0],[0,0,1,0,1,0,0,0.3,0,0,0,0,0,0,0,0,1,0,0,0,0],[0,0,0.5,0,0.5,0.5,0.3,0,0,1,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0.5,0,0,0,0.3,0,0,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0.5,0,0,0,0,1,0,0,0.3,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0.3,0.3,0,0,0,0.3,1,0,0,0,0,0,0,0,0],[0,0,0.5,0.5,0,0.5,0.3,0.5,0,0,0,0.3,1,0,0,0,0,0,0,0,0],[0,0,0.5,0.5,0.5,0.5,0.3,0.5,0,0,0,0.3,1,0,0,0,1,0,0,1,0],[0,0,0,0,0,0,0,0.5,0,0,0,0.3,1,0,0,0,0,0,0,0,0],[0,0,0.5,0,1,0.5,0.5,0,0,1,0,0,1,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0.5,0,0,0,0,1,0,0,0,1,0,0,1,0],[0,0,0.5,0.5,0.5,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0.5,0,0,0,0,1,0,0,0,0,0,0,0,0],[0.5,0.5,0.5,0.5,0.5,0.5,0.3,0,0,0,0,0.5,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0.3,0,0,0,0,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],[0,0,0.5,0.5,0,0,0,0.7,0,0,0,0,1,0,0,0,1,0,0,1,0],[0,0,0,0,0,0,0,0.5,0,0,0,0,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0.3,0,0,0,0,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],[0,0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],[0,0,0,0.5,0,0,0,0.3,0,0,0,0,1,0,0,0,0,0,0,0,0],[0,0,0,0.5,0,0,0.3,0.3,0,0,0,0,1,0,0,0,1,0,0,0,1],[0,0,0.5,0.5,0.5,0.5,0.3,0.5,0,0,0,0,1,0,0,0,0,0,0,0,0],[0,0,0.5,0.5,0.5,0.5,0.3,0.3,0,0,0,0,0,0,0,0,1,0,0,0,1]],$$jiageku$$:[[$$O01$$,$$K01$$,$$P05$$,$$W020$$,349,0,0,0,0,0],[$$O01$$,$$K01$$,$$P05$$,$$W021$$,309,0,0,0,0,0],[$$O01$$,$$B01$$,$$P11$$,$$W005$$,538,0,0,0,0,0],[$$O01$$,$$B01$$,$$P11$$,$$W006$$,608,0,0,0,0,0],[$$O01$$,$$B01$$,$$P11$$,$$W007$$,469,0,0,0,0,0],[$$O01$$,$$B01$$,$$P11$$,$$W008$$,798,0,0,0,0,0],[$$O01$$,$$B01$$,$$P11$$,$$W009$$,699,0,0,0,0,0],[$$O01$$,$$B01$$,$$P12$$,$$W001$$,1661,0,0,0,0,0],[$$O01$$,$$B01$$,$$P12$$,$$W002$$,1340,0,0,0,0,0],[$$O01$$,$$B01$$,$$P12$$,$$W003$$,2599,0,0,0,0,0],[$$O01$$,$$B01$$,$$P12$$,$$W004$$,1898,0,0,0,0,0],[$$O01$$,$$B01$$,$$P13$$,$$W010$$,59,0,0,0,0,0],[$$O01$$,$$B01$$,$$P13$$,$$W011$$,79,0,0,0,0,0],[$$O01$$,$$B01$$,$$P13$$,$$W012$$,45,0,0,0,0,0],[$$O01$$,$$B01$$,$$P14$$,$$W013$$,350,0,0,0,0,0],[$$O01$$,$$B01$$,$$P14$$,$$W014$$,198,0,0,0,0,0],[$$O01$$,$$W01$$,$$P21$$,$$W022$$,219,0,0,0,0,0],[$$O01$$,$$W01$$,$$P21$$,$$W023$$,169,0,0,0,0,0],[$$O01$$,$$W01$$,$$P22$$,$$W010$$,59,0,0,0,0,0],[$$O01$$,$$W01$$,$$P22$$,$$W011$$,79,0,0,0,0,0],[$$O01$$,$$W01$$,$$P22$$,$$W012$$,45,0,0,0,0,0],[$$O02$$,$$G01$$,$$P01$$,$$W015$$,675,0,0,0,0,0],[$$O02$$,$$G01$$,$$P01$$,$$W016$$,950,0,0,0,0,0],[$$O02$$,$$G01$$,$$P01$$,$$W018$$,480,0,0,0,0,0],[$$O02$$,$$G01$$,$$P01$$,$$W019$$,380,0,0,0,0,0],[$$O03$$,$$K01$$,$$P06$$,$$W066$$,12.96,0,0,0,0,0],[$$O03$$,$$K01$$,$$P06$$,$$W067$$,10.32,0,0,0,0,0],[$$O03$$,$$K01$$,$$P06$$,$$W068$$,258.84,0,0,0,0,0],[$$O03$$,$$K01$$,$$P06$$,$$W069$$,258.84,0,0,0,0,0],[$$O03$$,$$K01$$,$$P06$$,$$W070$$,110.64,0,0,0,0,0],[$$O03$$,$$K01$$,$$P06$$,$$W071$$,379.44,0,0,0,0,0],[$$O03$$,$$W01$$,$$P23$$,$$W066$$,22.68,0,0,0,0,0],[$$O03$$,$$W01$$,$$P23$$,$$W067$$,18.06,0,0,0,0,0],[$$O03$$,$$W01$$,$$P23$$,$$W068$$,452.97,0,0,0,0,0],[$$O03$$,$$W01$$,$$P23$$,$$W069$$,452.97,0,0,0,0,0],[$$O03$$,$$W01$$,$$P23$$,$$W070$$,193.62,0,0,0,0,0],[$$O03$$,$$W01$$,$$P23$$,$$W071$$,664.02,0,0,0,0,0],[$$O04$$,$$E01$$,$$P02$$,$$W101$$,100,0,0,0,0,0],[$$O04$$,$$E01$$,$$P02$$,$$W102$$,240,0,0,0,0,0],[$$O04$$,$$K01$$,$$P07$$,$$W101$$,100,0,0,0,0,0],[$$O04$$,$$K01$$,$$P07$$,$$W102$$,240,0,0,0,0,0],[$$O04$$,$$B01$$,$$P15$$,$$W101$$,100,0,0,0,0,0],[$$O04$$,$$B01$$,$$P15$$,$$W102$$,240,0,0,0,0,0],[$$O04$$,$$B01$$,$$P16$$,$$W101$$,100,0,0,0,0,0],[$$O04$$,$$B01$$,$$P16$$,$$W102$$,240,0,0,0,0,0],[$$O04$$,$$W01$$,$$P24$$,$$W101$$,100,0,0,0,0,0],[$$O04$$,$$W01$$,$$P24$$,$$W102$$,240,0,0,0,0,0],[$$O05$$,$$K01$$,$$P08$$,$$W028$$,1899,0,0,0,0,0],[$$O05$$,$$K01$$,$$P08$$,$$W029$$,999,0,0,0,0,0],[$$O05$$,$$K01$$,$$P08$$,$$W030$$,999,0,0,0,0,0],[$$O05$$,$$K01$$,$$P09$$,$$W031$$,199,0,0,0,0,0],[$$O05$$,$$K01$$,$$P09$$,$$W032$$,299,0,0,0,0,0],[$$O05$$,$$K01$$,$$P09$$,$$W033$$,599,0,0,0,0,0],[$$O06$$,$$K01$$,$$P10$$,$$W024$$,478,0,0,0,0,0],[$$O06$$,$$K01$$,$$P10$$,$$W025$$,298,0,0,0,0,0],[$$O06$$,$$W01$$,$$P25$$,$$W026$$,999,0,0,0,0,0],[$$O06$$,$$W01$$,$$P25$$,$$W027$$,3691,0,0,0,0,0],[$$O08$$,$$B01$$,$$P19$$,$$W098$$,750,0,0,0,0,0],[$$O08$$,$$B01$$,$$P19$$,$$W099$$,580,0,0,0,0,0],[$$O08$$,$$B01$$,$$P20$$,$$W100$$,198,0,0,0,0,0],[$$O08$$,$$B01$$,$$P20$$,$$W103$$,218,0,0,0,0,0],[$$O08$$,$$W01$$,$$P29$$,$$W061$$,268,0,0,0,0,0],[$$O08$$,$$W01$$,$$P29$$,$$W062$$,129,0,0,0,0,0],[$$O10$$,$$W01$$,$$P26$$,$$W089$$,398,0,0,0,0,0],[$$O10$$,$$W01$$,$$P26$$,$$W090$$,135,0,0,0,0,0],[$$O10$$,$$W01$$,$$P26$$,$$W091$$,390,0,0,0,0,0],[$$O10$$,$$W01$$,$$P26$$,$$W092$$,158,0,0,0,0,0],[$$O11$$,$$E01$$,$$P03$$,$$W048$$,158,0,0,0,0,0],[$$O11$$,$$E01$$,$$P03$$,$$W049$$,198,0,0,0,0,0],[$$O11$$,$$E01$$,$$P03$$,$$W050$$,228,0,0,0,0,0],[$$O11$$,$$E01$$,$$P03$$,$$W051$$,258,0,0,0,0,0],[$$O11$$,$$E01$$,$$P03$$,$$W052$$,148,0,0,0,0,0],[$$O11$$,$$E01$$,$$P03$$,$$W053$$,229,0,0,0,0,0],[$$O11$$,$$E01$$,$$P03$$,$$W054$$,259,0,0,0,0,0],[$$O11$$,$$E01$$,$$P03$$,$$W055$$,344,0,0,0,0,0],[$$O11$$,$$E01$$,$$P03$$,$$W056$$,98,0,0,0,0,0],[$$O11$$,$$E01$$,$$P03$$,$$W057$$,759,0,0,0,0,0],[$$O11$$,$$E01$$,$$P03$$,$$W058$$,338,0,0,0,0,0],[$$O12$$,$$E01$$,$$P04$$,$$W042$$,501,0,0,0,0,0],[$$O12$$,$$E01$$,$$P04$$,$$W043$$,561,0,0,0,0,0],[$$O12$$,$$E01$$,$$P04$$,$$W044$$,1176,0,0,0,0,0],[$$O12$$,$$E01$$,$$P04$$,$$W045$$,410,0,0,0,0,0],[$$O12$$,$$E01$$,$$P04$$,$$W046$$,598,0,0,0,0,0],[$$O12$$,$$E01$$,$$P04$$,$$W047$$,548,0,0,0,0,0],[$$O13$$,$$B01$$,$$P18$$,$$W034$$,300.2,0,0,0,0,0],[$$O13$$,$$B01$$,$$P18$$,$$W035$$,69,0,0,0,0,0],[$$O13$$,$$B01$$,$$P18$$,$$W036$$,102,0,0,0,0,0],[$$O13$$,$$B01$$,$$P18$$,$$W037$$,372,0,0,0,0,0],[$$O14$$,$$W01$$,$$P28$$,$$W063$$,1277.38,0,0,0,0,0],[$$O14$$,$$W01$$,$$P28$$,$$W064$$,268,0,0,0,0,0],[$$O14$$,$$W01$$,$$P28$$,$$W065$$,528,0,0,0,0,0],[$$O14$$,$$W01$$,$$P28$$,$$W093$$,175,0,0,0,0,0],[$$O15$$,$$B01$$,$$P17$$,$$W040$$,69,0,0,0,0,0],[$$O15$$,$$B01$$,$$P17$$,$$W041$$,40,0,0,0,0,0],[$$O15$$,$$W01$$,$$P27$$,$$W040$$,69,0,0,0,0,0],[$$O15$$,$$W01$$,$$P27$$,$$W041$$,40,0,0,0,0,0]],$$chanpinshuxing$$:[],$$laorenyuzhi$$:[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],$$zhuanjia$$:[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,4],[0,3,3,2,0,4,4,4,6,6,0,0,0,0,0],[3,4,5,3,0,0,0,5,0,5,0,2,0,5,4],[0,2,2,0,0,0,0,7,0,3,0,2,0,7,4],[3,5,6,4,0,0,2,5,3,3,0,0,3,3,4],[4,4,7,4,0,6,2,6,2,7,2,0,2,2,7],[4,6,6,3,0,0,3,6,3,3,3,2,3,3,7],[0,0,0,2,6,0,0,0,0,0,0,0,4,0,3],[2,2,3,0,5,0,0,2,0,2,0,0,0,0,0],[6,7,7,3,3,2,3,5,3,5,4,5,0,3,5],[2,0,2,0,0,0,0,2,0,0,0,0,0,0,5],[3,4,4,3,5,3,2,4,3,3,3,3,3,3,2],[0,0,5,2,5,2,0,0,0,2,0,2,0,3,2],[3,5,5,2,3,2,0,3,2,3,2,3,0,3,3],[3,5,5,2,5,4,0,3,0,2,3,3,4,2,0],[3,2,3,0,5,0,0,0,0,2,2,2,0,0,0],[2,0,2,0,0,0,0,0,0,0,2,2,0,0,3],[2,5,4,0,0,6,3,3,3,3,5,5,0,3,2],[2,5,5,2,0,2,2,4,2,2,3,4,0,4,3],[7,5,4,2,2,0,0,0,0,2,2,3,4,0,6],[2,2,2,0,2,0,0,0,0,2,2,2,7,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]}'
        json_str = self.js.replace('$$','"')
        self.params = json.loads(json_str)
        self.jiageku=pd.DataFrame(self.params['jiageku'],columns=['daxiang','kongjian','weizhi','chanpin',
                         'jiage','xianxiaodu','quanzhong','quanzhong2','quanzhong3','quanzhong4'])
        self.train_data=np.array(self.params['traindata'])
        self.target_data=np.array(self.params['targetdata'])
        self.zhuanjia2=pd.DataFrame(self.params['zhuanjia'],columns=['O01','O02','O03','O04',
                           'O05','O06','O07','O08','O09','O10',
                           'O11','O12','O13','O14','O15'])
        self.zhuanjia2.drop(self.zhuanjia2.index[21],inplace=True)
#        self.jiageku2=pd.DataFrame(columns=self.jiageku.columns)
        self.shengao=self.params['shengao']
        self.jiageku2 = tools.jiageku_sql(self.jiageku)
        self.cla = tools.TrainProcess(self.train_data, self.target_data)
        self.chanpinshuxing=pd.DataFrame(self.params['chanpinshuxing'],columns=['chanpin','<150cm','151-155cm','156-160cm','161-165cm','166-170cm','171-175cm','176-180cm','>180cm'])
    
            
    def test(self,laoren_js):
        json_str = laoren_js.replace('$$','"')
        js_params = json.loads(json_str)
        laoren1 = np.array(js_params['laoren']) 
        up = laoren1[22]
        kich = laoren1[23]
        laoren=np.copy(laoren1[0:21])
        c=pd.unique(self.jiageku2['daxiang'])
        d=np.setdiff1d(c,['O16','O17'])
        zhuanjia_new=self.zhuanjia2[d]
        if max(laoren)==0:
            laoren_MinMax=laoren
        else:
            laoren_MinMax=laoren/10
        Final=tools.ChangeItemProcess(zhuanjia_new,laoren)
        sss=self.cla
        laoren_MinMax1=laoren_MinMax.reshape((1,21))
        pre=sss.predict(laoren_MinMax1)
        jieguo=pre.todense()
        jieguo1=np.argwhere(jieguo == 1)
        jieguo2=jieguo1[:,1]
        changeitem=['O01','O02','O03','O04',
                'O05','O06','O07','O08','O09','O10',
                'O11','O12','O13','O14','O15']
        jieguo3=[changeitem[i] for i in jieguo2]
        FinalChange2=[val for val in Final if val in jieguo3]
        FinalChange1=[val for val in Final if val not in jieguo3]
        FinalChangeItem=FinalChange2+FinalChange1
        if 'O16' in c:
            FinalChangeItem=list(['O16'])+FinalChangeItem
    #    print(kitchen)
        if 'O17' in c and kich==1:
            FinalChangeItem=FinalChangeItem+list(['O17'])
    #    print(FinalChangeItem)
        if len(FinalChangeItem)==0:
            return 0
        else:
            price2 = tools.ahp_qz(self.jiageku2,kich,FinalChangeItem)
            ChangePro = tools.PriceCon(FinalChangeItem,price2,up)
       
            if 'O11' in list(ChangePro['daxiang']) and 'O12' in list(ChangePro['daxiang']) :
                p=0
                for val in ['W054','W055','W056']:
                    if val in list(ChangePro['chanpin']):
                        p=1
                if p == 0 :
                    B=[val for val in ['W054','W055','W056'] if val in list(price2['chanpin'])]
                    if len(B)==0:
                        ChangePro=ChangePro[ChangePro['daxiang']!='O12']
                    else:
                        C=price2[price2['chanpin']==B[0]]
                        D=ChangePro[ChangePro['daxiang']=='O11']
                        if max(D['price'])<min(C['price']):
                            ChangePro=ChangePro[ChangePro['daxiang']!='O12']
                        else:
                            ChangePro=ChangePro[ChangePro['daxiang']!='O11']
                            E=C.loc[C['price']==min(C['price']),C.columns]
                            ChangePro=ChangePro.append(E)
            
            for val in ['W094','W095']:
                if val in list(ChangePro['chanpin']):
                    for val2 in ['W076','W073','W072']:                                                                                                
                        if val2 in list(ChangePro['chanpin']):
                            ChangePro=ChangePro[ChangePro['chanpin']!=val]
#        print(self.chanpinshuxing)
        change_item=tools.tjsx(ChangePro,self.chanpinshuxing,self.shengao) 
#        print(self.chanpinshuxing)
        df=change_item.to_json(orient='index',force_ascii=False)
        return df
        
#    def test(self,num,name): 
#        a=np.zeros(3)
#        return name + str(num)+str(a[0])  
  
if __name__ == '__main__':  
    handler = TestServiceHandler()
#    print(handler.test("{$$laoren$$:[10,10,10,10,10,10,3,5,0,10,0,0,10,0,0,0,10,0,0,10,10,0,15000,1]}"))
#    print(handler.test("{$$laoren$$:[0,0,0,0,10,0,0,0,0,10,0,0,10,0,0,0,10,0,0,0,0,0,15000,1]}"))
    processor = TestService.Processor(handler)  
    transport = TSocket.TServerSocket(host='127.0.0.1',port=9090)  
    tfactory = TTransport.TBufferedTransportFactory()  
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()  
  
#    server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)  
    server = TServer.TThreadPoolServer(processor, transport, tfactory, pfactory)
    server.setNumThreads(5)
    print('python server:ready to start')  
    server.serve()  