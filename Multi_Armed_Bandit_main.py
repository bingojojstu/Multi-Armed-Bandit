from IPython import get_ipython
get_ipython().magic('reset -sf')
import numpy as np
import random

from matplotlib.pyplot import subplot, legend, xlabel, ylabel, semilogx, axis, show,plot, text
from scipy.stats import beta
from beta_bandit import *
import random

theta = (0.074,0.076,0.071,0.081) #testing trails
N = 60001 #total volume
batch=5000 #volume per age

def is_conversion(title):
    #if random.random() < theta[title]: 
    if random.random() < random.normalvariate(theta[title],theta[title]*0.1):  # add 10% standard deviation
        return True
    else:
        return False

conversions = [0]*len(theta)
trials = [0]*len(theta)
num_options=len(theta)

trials = zeros(shape=(N,len(theta)))
successes = zeros(shape=(N,len(theta)))

choices = zeros(shape=(N,))
bb = BetaBandit()


for i in range(N):
    if i%batch==0:
        trials1=trials[i]
        successes1=successes[i]
    choice = bb.get_recommendation(trials1,successes1)
    choices[i] = choice
    conv = is_conversion(choice)
    bb.add_result(choice, conv)
    trials[i] = bb.trials
    successes[i] = bb.successes
    if i==0:   #cleansing 
        trials[i]=[i]*len(theta)
        successes[i]=[i]*len(theta)
        
n = arange(N)+1

ll=shape(trials)[1]
x=range(ll)
y=trials[-1].tolist()
width=1/1.5
import matplotlib.pyplot as plt
subplot(111)
plt.bar(x,y,width,color='blue')
xlabel("Models")
ylabel("Number of trials/Ad")
show()

#%matplotlib qt 
subplot(111)
n = arange(N)+1
plot(n, trials[:,0], label="model 0")
plot(n, trials[:,1], label="model 1")
plot(n, trials[:,2], label="model 2")
plot(n, trials[:,3], label="model 3")
#plot(n, trials[:,4], label="model 4")
#plot(n, arange(N)+1, label="diagonal")
legend()
xlabel("Number of trials")
ylabel("Number of trials/Ad")
legend(loc=2)
show()


test1=zeros(shape=(N/batch+1,len(theta)))
j=1
test1[0]=[batch/len(theta)]*len(theta)
for i in range(N):
    if (i%batch==0) & (i !=0):
        test1[j]=trials[i]-trials[i-batch]
        j=j+1

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
fontP = FontProperties()
fontP.set_size('small')
subplot(111)
n = arange(N/batch+1)
plt.stackplot(n,[test1[:,0],test1[:,1],test1[:,2],test1[:,3]])#,test1[:,4]])
#plt.stackplot(n,[trials[:,0],trials[:,1],trials[:,2],trials[:,3],trials[:,4]])
plt.legend(['model 0','model 1','model 2','model 3'],bbox_to_anchor=(1.3, 0.8))
plt.xlabel('age')
plt.ylabel('volume')
plt.show()


subplot(111)
n = arange(N/batch+1)
dashed_u=[0.05]*(N/batch+1)
dashed_l=[0.95]*(N/batch+1)
plot(n, test1[:,0]/batch, label="model 0")
plot(n, test1[:,1]/batch, label="model 1")
plot(n, test1[:,2]/batch, label="model 2")
plot(n, test1[:,3]/batch, label="model 3")
#plot(n, test1[:,4]/batch, label="model 4")
plot(n, dashed_l, 'b--',label="upper confidence")
plot(n, dashed_u, 'b--',label="lower confidence")
plt.xlabel('age')
plt.ylabel('Probability of being optimal')
plt.legend(bbox_to_anchor=(1.47, 0.8))
plt.show()