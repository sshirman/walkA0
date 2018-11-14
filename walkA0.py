#!/home/sasha/anaconda3/bin/python
# coding: utf-8

# In[9]:

#This code is a Strategic Monte Carlo Code developed to integrate with and interact with minAone.py
#Code is written by Sasha Shirman
#June 2018

###################################################################################################

##This code is built to run a monte carlo search on the probability space of the dynamical system
##given the measured data. The input of this code is the output of minAone, and the supporting files
##that are listed below. The output is a file for each variable with each *row* being an accepted time
##series in the search and a file for all the parameters with each *column* being the appropriate
##parameter estimate and each *row* being the specific walker step of the search.
    #This output then needs to be analyzed (plotted as a histogram type function) to get means and 
    #standard deviations

#as inputs you must have a file called "ParallelEqns.py". This is a python file with the correct
#imported functions that defines your equations. The input for ParallelEqns should be a vector
#of variable values (think of it as at one time, but it can handle at all measurement times as well)
#The output or return of ParallelEqns should be a vector of derivatives in the order that the
#variables are assigned in minAone and here. The function in ParallelEqns.py that you are defining
#should be called "ParDeriv". For example you should have def ParDeriv(x): d1 = derivative for variable1
#d2 = derivative for variable2, etc, return np.array([d1,d2,d3, etc]). Here di are written as functions
#of the x[j]'s in the appropriate manner

#Another input you have is the measurement function. This should be named "measurementfunction.py". This
#file is only needed if you are using a measurement function that is not just the variable (for example
#you do not need a measurement function file if you are measuring voltage and voltage is one of your
#dynamical variables). This file should be set up in a similar way as the paralleleqns file in that it
#has all the imports needed and it defines a function of the variables.

#If there is a measurement function that is not just equal to a subset of the variables, then numData
#is a dummy variable that is only needed to make the code work and should just be an empty set ([]).
#If there is no measurement file, then numData should be a set of all the variable indices of the
#recorded variables. For example if the first and third variables are recorded, then numData = [0, 2]
# -- remember that python starts counting from 0!!!

#There also is an input called "NumInfo.txt" which is the numerical information file. This file is where
#you put all the information the code needs, like the number of states, number of parameters, number of
#walkers, best and second best paths, and alpha and beta values used (and some other terms). 

import sys
import numpy as np
import scipy as sp
import sympy as sym
import matplotlib.pyplot as plt
import os
from scipy.integrate import quad
import copy
from scipy.special import erfinv
from statsmodels.stats.weightstats import DescrStatsW as ws

cwd = os.getcwd()
sys.path.append(cwd)

inputs = np.loadtxt('NumInfo.txt', delimiter = ',', usecols = (1,))
nS = int(inputs[0])
nP = int(inputs[1])
nM = int(inputs[2])
nT = int(inputs[3])
deltaT = inputs[4]
Nwalk = int(inputs[5])
Nburn = int(inputs[6])
alpha = inputs[7]
beta = int(inputs[8])
betabad = int(inputs[9])
nSteps = Nburn*500
nSave = int(Nburn/25)
Rf = np.reshape(inputs[10:10+nS],(nS,1))
numData = inputs[10+nS:-4].astype(int)
best = int(inputs[-4])
second = int(inputs[-3])
Attempt = int(inputs[-2])
Inputs = int(inputs[-1])

#This is to copy the data that has been recorded into the right format for this code to read and work with. This copies
#files that start with 'datai.txt' if you recorded data

y = np.zeros((nM,nT))
if len(numData)>0:
    for index in range(nM):
        y[index,:] = sp.loadtxt('Analysis_Attempt{1}/data{0}.txt'.format(numData[index],Attempt))[:nT]
elif len(numData)==0:
    for index in range(nM):
        y[index,:] = sp.loadtxt('Analysis_Attempt{1}/measurement{0}.txt'.format(numData,Attempt))[:nT]

################################################################################################
##This is to grab the input of data from minAone outputs!
bestP = sp.loadtxt('Analysis_Attempt{3}/D{0}_M{1}_IC{2}.dat'.format(nS,nM, best,Attempt))
secondP = sp.loadtxt('Analysis_Attempt{3}/D{0}_M{1}_IC{2}.dat'.format(nS,nM,second,Attempt))

RmInv = np.divide(bestP[beta,2],nM)

bestpath = np.array(bestP[beta,3:])
secondpath = np.array(secondP[betabad,3:]) #I might need to change this based on the new version of minAone and what the
#output is. I just would need to look at what that matrix looks like and also how it names the outputs.
####################################################################################################

#This rearranges the data from the shape that is present in minAone to the shape that I want to use,a matrix with
#each row representing a time series of the corresponding variable.
#I also have all variables in these matrices while all parameters are in their own matrices

BestPath = np.reshape(bestpath[:-nP], (nT, nS)).T
SecondPath = np.reshape(secondpath[:-nP], (nT, nS)).T

ParameterBest = bestpath[-nP:]
ParameterSecond = secondpath[-nP:]

#These matrices use the best and second best matrices to create values for the standard deviation
#and the step size

#SigmaData = np.absolute(np.multiply(np.subtract(BestPath, SecondPath),4))
SigmaParam = np.absolute(np.multiply(np.subtract(ParameterBest, ParameterSecond), 4))
#DeltaData = np.absolute(np.divide(np.subtract(BestPath, SecondPath),20.0))
DeltaParam = np.absolute(np.divide(np.subtract(ParameterBest, ParameterSecond), 20.0))


#choosing same search standard deviation for every time point of any given variable (this is set to
#the largest search range). The purpose of this is to make sure that you don't have regions where two
#very similar paths cross and give a search range of ~zero.
SigmaData = np.multiply(np.max(np.absolute(np.multiply(np.subtract(BestPath, SecondPath),4)),axis=1),np.ones(np.shape(BestPath)).T).T
DeltaData = np.multiply(np.max(np.absolute(np.divide(np.subtract(BestPath, SecondPath),20.0)),axis = 1),np.ones(np.shape(BestPath)).T).T

SigmaWalkerParam = np.sqrt(np.divide(1,np.add(np.divide(1,np.square(SigmaParam)),np.divide(1,np.square(DeltaParam)))))
SigmaWalkerData = np.sqrt(np.divide(1,np.add(np.divide(1,np.square(SigmaData)),np.divide(1,np.square(DeltaData)))))
QParam1 = np.multiply(ParameterBest,np.square(DeltaParam))
QParam2 = np.add(np.square(DeltaParam),np.square(SigmaParam))
QData1 = np.multiply(BestPath,np.square(DeltaData))
QData2 = np.add(np.square(DeltaData),np.square(SigmaData))

if Inputs == 1:
    I = np.reshape(sp.loadtxt('Analysis_Attempt{0}/Input.txt'.format(Attempt))[:nT], (nT,1))

import ParallelEqns

ParDeriv = ParallelEqns.ParDeriv
ParallelEqns.nS = nS

#This gives spits out a matrix such that each row is a time series of the derivative
#of the corresponding variable

#Following is the gaussian centered at the "best" guess of minAone and with standard deviation given
#by the "second best" guess from minAone

"""def sampling(current, stepsize, std, mean):
    F = np.multiply(np.divide(np.multiply(np.multiply(std, stepsize),np.sqrt(2)), np.sqrt(np.square(std)+np.square(stepsize))),erfinv(1-np.multiply(2,np.random.rand(*current.shape)))) + np.divide(np.multiply(mean, np.square(stepsize))+np.multiply(current, np.square(std)),np.square(stepsize)+np.square(std))
    return F"""
    
def sampling(current,q1,q2,standarddev,standardbias):
    F = np.random.normal(np.divide(np.add(q1,np.multiply(current,np.square(standardbias))),q2),standarddev)
    return F


def Derivatives(current, parameter):
    Derivatives = np.zeros(np.shape(current))
    for i in range(Nwalk):
        Derivatives[i] = ParDeriv(current[i], parameter[i])
    return Derivatives

def measurement(x,number_recordings):
    if os.path.isfile('measurementfunction.py'):
        import measurementfunction
        return measurementfunction.h(x)
    else:
        return x[:,number_recordings,:]

#Need to adjust so that we have an option of the objective function has a measurement function in it
#instead of just(x-y) -- so it can have (h(x) - y). That will come later

#Also need to figure out what the value of Rm is used in minAone so that the action is the same as
#through the minimization procedure
def ActionM(current,data, variablesrecorded):
    A = np.sum(np.square(np.around(np.subtract(measurement(current,variablesrecorded),data),7)), axis = (1,2))
    return A
    

#def ActionF(current,parameter):
#    k1 = Derivatives(current[:,:,:-1], parameter)
#    k2 = Derivatives(current[:,:,:-1] + (deltaT/2.0)*k1,parameter)
#    k3 = Derivatives(current[:,:,:-1] + (deltaT/2.0)*k2,parameter)
#    k4 = Derivatives(current[:,:,:-1] + (deltaT)*k3,parameter)
#    A = np.sum(np.multiply((Rf*alpha**beta), np.square(np.subtract(current[:,:,1:], np.add(current[:,:,:-1], (deltaT/6.0)*(k1+2.0*k2+2.0*k3+k4))))),axis = (1,2))
#    return A


#below is an attempt to make the action calculate for all walkers simultaneously
def ActionF(current,parameter):
    current2 = np.transpose(current,(1,2,0))
    parameter2 = np.transpose(parameter,(1,0))
    if Inputs == 0:
        k1 = ParDeriv(current2[:,:-1,:], parameter2)
        k2 = ParDeriv(current2[:,:-1,:] + (deltaT/2.0)*k1,parameter2)
        k3 = ParDeriv(current2[:,:-1,:] + (deltaT/2.0)*k2,parameter2)
        k4 = ParDeriv(current2[:,:-1,:] + (deltaT)*k3,parameter2)
        ks = ((k1+2.0*k2+2.0*k3+k4))
        A = np.sum(np.multiply(np.reshape(Rf,(nS,1,1)),np.square(np.around(current2[:,1:,:]- current2[:,:-1,:] - deltaT*ks/6.0,7))),axis = (0,1))*(alpha**beta)
    if Inputs == 1:
        k1 = ParDeriv(current2[:,:-1,:], parameter2, I[:-1])
        k2 = ParDeriv(current2[:,:-1,:] + (deltaT/2.0)*k1,parameter2, np.divide(I[:-1]+I[1:],2))
        k3 = ParDeriv(current2[:,:-1,:] + (deltaT/2.0)*k2,parameter2, np.divide(I[:-1]+I[1:],2))
        k4 = ParDeriv(current2[:,:-1,:] + (deltaT)*k3,parameter2, I[1:])
        ks = ((k1+2.0*k2+2.0*k3+k4))
        A = np.sum(np.multiply(np.reshape(Rf,(nS,1,1)),np.square(np.around(current2[:,1:,:]- current2[:,:-1,:] - deltaT*ks/6.0,7))),axis = (0,1))*(alpha**beta)
    
    return A

#0.5*deltaT*np.add(derivative[:,:,1:],derivative[:,:,:-1])

def Action(current,parameter, data, variablesrecorded):
    Action = ActionM(current, data, variablesrecorded) + ActionF(current, parameter)
    return np.divide(Action,RmInv)

from scipy.stats import norm as bias
from scipy.stats.mstats import gmean as geommean


location = copy.copy(BestPath)
parameterspot = copy.copy(ParameterBest)

#in many/all cases the product of nS*nT+nP of gaussian distributions when calculating the weight ends up being far too small and gets rounded to zero. When the code here calculates the weights when removing bias, it only looks at the relative sizes of the different weights (meaning weights [2,2] gives the same bias as [1,1] and [0.001,0.001]). So what the following function does is it calculates the average exponential in the gaussian bias function at each time step. This will be done once in the code at the beginning of the main analysis section. The output here is a power scaling factor. We will multiply each gaussian when calculating the bias by 10^{scaling factor} before taking the product over all the time steps. This will allow each walker and each iteration to have the same overall scaling factor in the weights while also making sure that these weights do not end up being zero from being too small for computational accuracy.
#def weightscaling(current,parameter,bestv,stdv):
#    weight = np.ones((Nwalk,nS,nT))
#    for i in range(nS):
#        for t in range(nT):
#            weight[:,i,t] = bias.pdf(current[:,i,t], mean = bestv[i,t], cov = stdv[i,t])
#    return np.max(np.log10(np.prod(weight,axis = 1)))

def weightcalc(current, parameter, bestv, bestp, stdv, stdp):
    weightpre = np.ones((Nwalk,nS,nT))
    weightpre = bias.pdf(current, bestv, stdv)
    weightpre =np.reshape(geommean(np.prod(weightpre, axis = 1),axis = 1),(Nwalk,1))
    if nP ==1:
        weightpre = np.multiply(weightpre,np.power(bias.pdf(parameter, bestp, stdp),np.divide(1,np.float(nT))))
    elif nP ==0:
        weightpre = weightpre
    else:
        weightpre = np.multiply(weightpre,np.power(np.reshape(np.prod(bias.pdf(parameter, bestp, stdp),axis = 1),(Nwalk,1)),np.divide(1,np.float(nT))))
    #double check if above works in the >1 parameter case
    return weightpre

testerposition = location+0.5*SigmaData*np.ones((Nwalk,nS,nT))*(0.5-np.random.rand(Nwalk,nS,nT))  
testerparam = parameterspot + 0.5*SigmaParam*np.ones((Nwalk,nP))*(0.5-np.random.rand(Nwalk,nP))  


try:
    os.makedirs('Analysis_Attempt{0}/Results'.format(Attempt))
except OSError:
    if not os.path.isdir('Analysis_Attempt{0}/Results'.format(Attempt)):
        raise

with open('Analysis_Attempt{0}/Results/InitialParamEstimate.txt'.format(Attempt),'ab') as initialP:
    np.savetxt(initialP,testerparam)
    

with open('Analysis_Attempt{0}/Results/MeanSearchParameter.txt'.format(Attempt), 'ab') as bestpar:
    np.savetxt(bestpar,ParameterBest)

    
    
step = 0
##############################################################################################################
#This part will keep running through options until it picks a complete path with the gaussian distribution.
#Once a proposed path is decided on, you can then test how this compares to the last one

#You start by setting new* to tester*. Then you cycle through until new* is chosen along the appropriate
#distribution. After which you will be able to compare the new new* value to tester*. If you decide to
#accept the new new* value, you will set tester* = new*. Then start over again by setting new* = tester*
#(you now have two arrays with the new values instead of two with the old ones). If you do not decide to
#accept the new* values, then you set tester* = tester* and don't change that value. Then again new* is set
#to tester* and you try to find a different set of values for new* that will be accepted
print("Starting Burn in Period \n")
savenumber = 0
while step <nSteps+Nburn:
    newpath = copy.copy(testerposition)
    newparam = copy.copy(testerparam)
    goodaction = []
    '''newpath = sampling(testerposition,DeltaData, SigmaData, BestPath)
    newparam = sampling(testerparam, DeltaParam, SigmaParam, ParameterBest)'''
    newpath = sampling(testerposition,QData1,QData2,SigmaWalkerData,SigmaData)
    newparam = sampling(testerparam,QParam1,QParam2,SigmaWalkerParam,SigmaParam)
    testaction = np.random.rand(Nwalk)
    actionnew = Action(newpath, newparam, y, numData)
    actionold = Action(testerposition,testerparam,y,numData)
    goodaction = np.invert(np.logical_or((-actionnew+actionold)>2.0*np.log(testaction), actionnew<actionold)) #added the factor of 1/2 into the action by multiplying np.log(testaction) by two. I think this factor of two needs to be there because the way the cost function is defined in minAone is missing that factor of 2, which means that the action it spits out at the flat portion is equal to 1/T sum((x-y)^2) = sigma^2*nM. We want action to have a 1/(2*sigma^2) term which means that we should scale the action by nM/(minAone_out *2)
    goodpath =  np.transpose(np.repeat(np.array([np.repeat(np.array([goodaction]),nS,axis = 0)]),nT,axis = 0),(2,1,0))
    goodparam = np.repeat(np.array([goodaction]),nP,axis = 0).T
    
    '''goodpath = np.zeros([Nwalk, nS, nT])
    goodparam = np.zeros([Nwalk, nP])
    
    for i in range(Nwalk):
        goodpath[i,:,:] = goodaction[i]
        goodparam[i,:] = goodaction[i]'''

    maskedtestposition = np.ma.array(testerposition, mask = goodpath)
    maskednewposition = np.ma.array(newpath, mask = goodpath)
    maskedtestparam = np.ma.array(testerparam, mask = goodparam)
    maskednewparam = np.ma.array(newparam, mask = goodparam)
    maskedtestposition[~maskedtestposition.mask] = maskednewposition[~maskednewposition.mask]
    testerposition = maskedtestposition.data
    maskedtestparam[~maskedtestparam.mask] = maskednewparam[~maskednewparam.mask]
    testerparam =maskedtestparam.data
    newpath = []
    newparam = []
    if step%int(nSave) ==0 and step < Nburn+1:
        sys.stdout.write('\r')
        sys.stdout.write('[%-20s] %.1f%%' % ('='*int(20*(step+1)/Nburn), (100.0*(step+1)/Nburn)))
        sys.stdout.flush()
    elif step == Nburn+1:
        print("\nStarting Main Analysis")
    if step>Nburn+1 and (step-Nburn+1)%(int(nSteps/200)) == 0:
        sys.stdout.write('\r')
        sys.stdout.write('[%-50s] %.1f%%' % ('='*int(50*(step-Nburn)/(nSteps-1)), float(100.0*(float(step) - float(Nburn))/float(nSteps-1))))
        sys.stdout.flush()
        
    if step>Nburn-1 and (step-Nburn)%(50*nSave) == 0:
        for i in range(nS):
            with open('Analysis_Attempt{1}/Results/variable_{0}.txt'.format(i,Attempt),'ab') as f:
                np.savetxt(f,testerposition[:,i,:])
        with open('Analysis_Attempt{0}/Results/variableweights.txt'.format(Attempt),'ab') as vw:
            np.savetxt(vw,weightcalc(testerposition, testerparam, BestPath, ParameterBest, SigmaData, SigmaParam))

    if  step>Nburn-1 and (step-Nburn)%nSave == 0:
        with open('Analysis_Attempt{0}/Results/parameters_estimate.txt'.format(Attempt), 'ab') as p:
            np.savetxt(p, testerparam)
        with open('Analysis_Attempt{0}/Results/parameterweights.txt'.format(Attempt),'ab') as pw:
            np.savetxt(pw,weightcalc(testerposition, testerparam, BestPath, ParameterBest, SigmaData, SigmaParam))
    step+=1

print('\n')
with open('Analysis_Attempt{0}/Results/SigmaSearchParameter.txt'.format(Attempt), 'ab') as s:
    np.savetxt(s,SigmaParam)
    
for i in range(nS):
    with open('Analysis_Attempt{1}/Results/SigmaSearchVariable_{0}.txt'.format(i,Attempt),'ab') as g:
        np.savetxt(g, SigmaData[i,:])

#The following deletes the histogram like data of the variables and just saves the average path as well as the standard deviation of the path for each
#the variables. This also saves the "best path" and best parameter from minAone

varweights = np.loadtxt('Analysis_Attempt{0}/Results/variableweights.txt'.format(Attempt))
varweights = np.power(np.divide(varweights,np.power(10,np.mean(np.log10(varweights)))),nT)

for i in range(nS):
    var = np.loadtxt('Analysis_Attempt{1}/Results/variable_{0}.txt'.format(i,Attempt))
    meanvar = np.mean(var,axis=0)
    standard = np.std(var, axis=0)
    #weightedmean = np.zeros(nT)
    #weightedstandard = np.zeros(nT)
    #weightmatrix = gaussian(var,BestPath[i,:],SigmaData[i,:])
    stats = ws(var, weights = np.divide(1,varweights),ddof=0)
    weightedmean = (stats.mean)
    weightedstandard = stats.std
    
    '''for point in range(nT):
        stats = ws(var[:,point], weights = weightmatrix[:,point],ddof=0)
        weightedmean[point] = stats.mean
        weightedstandard[point] = stats.std'''

    with open('Analysis_Attempt{1}/Results/mean_variable_{0}.txt'.format(i,Attempt), 'ab') as mvar:
        np.savetxt(mvar,meanvar)

    with open('Analysis_Attempt{1}/Results/std_variable_{0}.txt'.format(i,Attempt),'ab') as stdvar:
        np.savetxt(stdvar, standard)

    with open('Analysis_Attempt{1}/Results/unweighted_mean_variable_{0}.txt'.format(i,Attempt), 'ab') as mvarweight:
        np.savetxt(mvarweight,weightedmean)

    with open('Analysis_Attempt{1}/Results/unweighted_std_variable_{0}.txt'.format(i,Attempt), 'ab') as stdvarweight:
        np.savetxt(stdvarweight, weightedstandard)

    with open('Analysis_Attempt{1}/Results/last_time_step_variable_{0}.txt'.format(i,Attempt),'ab') as lastvar:
        np.savetxt(lastvar, var[:,-1])

    os.remove('Analysis_Attempt{1}/Results/variable_{0}.txt'.format(i,Attempt))

    with open('Analysis_Attempt{1}/Results/MeanSearchVariable_{0}.txt'.format(i,Attempt), 'ab') as bestvar:
        np.savetxt(bestvar,BestPath[i,:])

    var=[]
    meanvar = []
    standard = []
