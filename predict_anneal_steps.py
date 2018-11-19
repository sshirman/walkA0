import os

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

'''

Usage: python /path/to/predict_minAone.py [predict-time-steps] [PATH file num]



For example, to predict after the end of D7_M3_PATH1.dat, run this in the same directory:

python ~/pythonscripts/predict_minAone.py 10000 1



OUTPUT: Will save two files: data.dat with contains D+2 columns: [times, x0, x1, ..., xD, inj]. It will also save the parameter values in params.dat



_________________________________________________________



Prediction script for minAone. Should read in the numbers of states, parameters, time points from  specs.txt; read the equations from equations.txt; and load the parameter values from one of the path files, and do an integration.



Current file must have sufficient data AFTER the estimation to use for prediction. You can change the first line of specs.txt after the estimtaion, and the prediction will start from the appropriate point part-way through the estimation.



For external functions (ex. ghk(V) in myfunctions.cpp) you will have to rewrite the function (derivatives not necessary) in python file called "myfunctions.py"



myfunctions.py should start with "from sympy.functions import *" if it uses any mathematical functions (i.e. use sympy version of exp(x), not numpy or other package). Any "if" statements you want to include myfunctions.py need to be written using the Piecewise sympy function instead. 



Currently can only handle 1 input currenet. I think need to change eqn_wrapper() with multiple currents.

__________________________________________________________________________________

Addition by Sasha Shirman on May 31,2016:

if loops were added that allowed for nI (or number of injected currents) to be zero.

This allows for systems with no input (L96)

'''

import matplotlib

import numpy as np

matplotlib.use('Agg')

import sys

import scipy as sp

from scipy.integrate import odeint

import sympy as sym

import matplotlib.pyplot as plt

import os



try:

    import myfunctions

except:

    print "Alert: no myfunctions.py file in directory"



if len(sys.argv) < 2:

    raise ValueError("Num time steps not specified")

elif len(sys.argv)<3:

    raise ValueError("Path not specified")    



predict_steps = int(sys.argv[1])

pathnum = sys.argv[2]



### Load numbers of states, measurments and equations:



with open('equations.txt') as eqfile:

    count = 0

    eqns = []

    eqnstrings = []

    states = []

    params = []

    inj = []

    funcs = []

    funcstrings = []

    for i,line in enumerate(eqfile):

        if line[0] == '#': continue

        count += 1

        # skip first line, problem name

        if count == 1:  continue

        if count == 2: 

            nS, nP, nU, nI,nF,nM = [int(x) for x in line.strip().split(',')]

            

        #equations start at line 3

        elif count <= 2 + nS:

           # print "eqns lines = ", line

            eqnstrings.append(line)

        # Variables names at line 4+nS

        elif 3+ nS< count < 4 + 2*nS:

            # print "states lines = ", line

            states.append(sym.symbols(line))

        # Parameters start at line 4+2*nS

        elif 3+2*nS < count < 4+2*nS+nP:

            # print "param lines = ", line

            params.append(sym.symbols(line))

        # Injected current starts at 5+2*nS+nP+nU+nM

        elif nI>0 and 4+2*nS+nP+nU+nM<= count < 4+2*nS+nP+nU+nM+nI:

            #print "Iinj lines = ", line

            inj.append(sym.symbols(line))

        elif nF>0 and 4+2*nS+nP+nU+nM+nI <= count < 4+2*nS+nP+nU+nM+nI+nF:

            #print "Fcn lines = ", line

            fcnname, varnum = line.strip().split(',')



            funcstrings.append(fcnname)

            try:

                fcntmp = eval('myfunctions.'+fcnname)

            except: 

                print fcnname

                ValueError("Is function defined in myfunctions.py?")



                

            funcs.append(fcntmp)



data_files = []

with open('specs.txt') as specsfile:

    count = 0

    for i,line in enumerate(specsfile):

        if line[0] == '#': continue

        count += 1

        # skip first line, problem name

        if count == 1:

            nT = 2*int(line.strip())+1

        elif count == 2:

            skip = int(line.strip())

        elif count == 3:            

            dt = float(line.strip())/2.0

        elif 4 < count <= 4 + nM:

            data_files.append(line.strip())

        elif count == 5 + nM:

            Ifile = line.strip()

    



# Load in current file, ignore first 'skip' number of time steps

if nI>0:
    I = sp.loadtxt(Ifile)[skip:]

if nI>0 and len(I) < nT + predict_steps:

    raise ValueError("Current file too short. Only {0} steps available for prediction".format(len(I)-nT-1))


# Use data.dat file if that exists (this script as been run before). Else, read in param values from a path file, and save last path as data.dat

#try:

#    path = sp.loadtxt('data.dat', usecols=sp.arange(1,nS+1))

#    param_values = sp.loadtxt('params.dat', usecols=[1],dtype=float)

#    import ipdb; ipdb.set_trace()

#    print "Using existing data.dat file and params.dat file"

#except:

# Above was written to save some time loading path file, but I don't think it takes long neough to invest the effort to make it work.



print "loading Path file, saving to data.dat and params.dat"

# Take the last nP values, and the last row (w/ largest beta)

pathfile = 'D{0}_M{1}_IC{2}.dat'.format(nS,nM,pathnum)

last_path_data = sp.loadtxt(pathfile)
all_path_data = last_path_data


for anneal_step in range(len(all_path_data[:,0])):

    last_path_data = all_path_data[anneal_step, :]


    param_values = last_path_data[-nP:]

    path = last_path_data[3:3+nT*nS]

    sp.ndarray.reshape

    path = path.reshape((nT,nS))



    data_times = sp.arange(dt,(nT)*dt+dt,dt)

    if nI>0:

        sp.savetxt('data.dat', sp.column_stack((data_times,path,I[:nT])))

    else:
        sp.savetxt('data.dat',sp.column_stack((data_times,path)))


    param_array = sp.array(zip([x.name for x in params],param_values))

    sp.savetxt('params.dat', param_array, fmt='%s',delimiter=' \t')

 

    print "Path saved"



    funcNameSpace = dict(zip(funcstrings,funcs))

    paramNameSpace = dict(zip(params,param_values))

    

    x = sym.symbols('x:'+str(nS+nI))
    eqns = []

    for eq in eqnstrings:

        eqfunc = sym.sympify(eq,locals=funcNameSpace)

        eqfunc = eqfunc.subs(paramNameSpace)

        eqlamb = sym.lambdify(x, eqfunc.subs(zip(states+inj,x)))        

        eqns.append(eqlamb)


    if nI>0:

        def eqns_wrapper(x,t,I,dt):

            deriv = []

            Iinj = I[sp.floor(t/dt)]    

            input = list(x)+[Iinj]

            for eq in eqns:

                deriv.append( eq(*input) )        

            return deriv
    else:

        def eqns_wrapper(x,t,dt):

            deriv = []

            input = list(x)

            for eq in eqns:

                deriv.append( eq(*input) )

            return deriv        



    init = path[-1,:]



    print "Integrating..."

    time = sp.linspace(nT*dt,(nT+predict_steps)*dt, predict_steps)


    if nI>0:

        predict = odeint(eqns_wrapper, init, time, (I,dt))

    else:

        predict = odeint(eqns_wrapper,init,time,(dt,))



    data0 = sp.loadtxt(data_files[0])

    data0 = data0[skip:]



    if len(data0) < nT+predict_steps:

        print "Warning: data file does not have enough data points for entire prediction range"

        data_length = len(data0)

    else:

        data_length = nT+predict_steps



    plt.plot(time, predict[:,0],'b',label="Prediction")

    plt.plot(time[:data_length-nT], data0[nT:data_length], 'r', label="Data")

    plt.title("Soma Voltage")

    plt.legend()
    ensure_dir('path{0}/anneal_step{1}.png'.format(pathnum, anneal_step))

    plt.savefig('path{0}/predict{1}.png'.format(pathnum, anneal_step))

#plt.show()

    plt.plot(data_times,)

    plt.close()

    plt.figure(figsize=(20,10))
    
    if nI>0:

        plt.subplot(2,1,1)

    else:
        plt.subplot(1,1,1)

    plt.plot(data_times,path[:,0], color = 'blue', label = "Estimate")

    plt.plot(np.hstack((data_times,time)), data0[0:data_length], color = 'black', label="Data", alpha = 0.7)

    plt.plot(time,predict[:,0],color = 'red', label = 'Prediction')

    plt.ylabel("voltage (mV)", fontsize = 20)

    plt.legend()



    if nI>0:

        plt.subplot(2,1,2)

        plt.plot(np.hstack((data_times,time)),I[0:data_length], color = 'purple', label = "Injected Current")

        plt.xlabel("time (ms)", fontsize = 20)

        plt.ylabel("current (pA)", fontsize = 20)

        plt.legend()

    plt.savefig('path{0}/anneal_step{1}.png'.format(pathnum, anneal_step))

    plt.close()


    if nI>0:

        predict = sp.column_stack((time,predict, I[nT:nT+predict_steps]))

    else:

        predict = sp.column_stack((time,predict))

    sp.savetxt('path{0}/predict{1}.dat'.format(pathnum, anneal_step), predict)







        

    
