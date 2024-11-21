# import packages
import random
from scipy.stats import poisson, uniform
import numpy as np
import itertools as it
import math
import gurobipy as gp
from gurobipy import GRB
import copy
import timeit
import time
rng = np.random.default_rng()

## input values
C = 3 #number of disciplines
U = 3 # number of urgency levels

T_MDO_o = 2 # number of weeks between MDO 
T_start_o = 7 # predefined number of initial treatment weeks
max_cont_o = 5 #maximum number of continuation
T_MDO_i = 2 # number of weeks between MDM
T_start_i = 3   # predefined number of initial treatment weeks
max_cont_i = 5 #maximum number of contituation 

max_nr_i = 25
# probability of contituation
ranges_o = [0.33, 0.6, 0.8, 0.9, 1]
probs_o = [0.66, 0.4, 0.5, 0.5, 0]
ranges_i = [0.25, 0.55, 0.8, 0.95, 1]
probs_i = [0.75, 0.6, 4/9, 0.25, 0]


#weeks until discharge set
L_o = [i for i in range(T_start_o+1)]
L_i = [i for i in range(T_start_i+1)]

#arrival rates
lam_o = 3 #mean number of arrivals to the waiting list
lam_i = 3 #mean number of inpatient arrivals

#capacity setting
f =[120, 145, 110]


# list of treatment plan of outpatients
P = [[1, 1, 1], [1,1,2], [2, 2, 0]]

P_nr = [i for i in range(1,len(P)+1)]
P_dist = [0.6, 0.2, 0.2] #distribution of treatment plan


# list of treatment plan of inpatients
Q = [[3, 4, 4],[2, 3, 1]]
Q_dist = [2/4, 2/4] #distribution of treatment plan


# waiting list properties
U_dist = [0.5, 0.1, 0.4]
urgency_nr = [u for u in range(1, U+1)] #urgency array 

# properties for the cost function
U_cost  = np.array([0.55, 1.05, 2.05])
C_cost = np.array([1.5, 1.5, 1.5])

#end of system setting
##############BELOW HERE ONLY FUNCTIONS##############

#description of the outpatient treatment plan in the next T_MDO weeks
## 
Pw_int = list(it.product(P_nr, repeat=T_MDO_o+1))
Pw= []
for i in Pw_int:
    if  i[T_MDO_o]!=0:
        count = 0
        count_0 = 0
        for j in range(1,len(i)):
            if i[j]!= i[j-1]: 
                count+=1
            if i[j-1]==0:
                count_0+=1
        
        if count <= 1 and count_0<=1: 
            if i[0]==0:
                Pw.append(i)

            else:
                Pw.append(i)
Pw_nr = [i for i in range(len(Pw))]


#matrix that tells to which Pw number the patient continues the next week if the next week treatment plan p is plannend
Pw_index = np.zeros((len(Pw), len(P)), dtype=np.int64)
count_pw = 0
for pw in Pw:
    count_p = 0

    for p in range(1,len(P)+1):
        new = pw[1:]
        new= new + (p,)
        getIndex = list([x for x, y in enumerate(Pw) if y==new])
        if len(getIndex)>0:
            Pw_index[count_pw][count_p] = getIndex[0]
        else:
            Pw_index[count_pw][count_p] = -1
        count_p+=1
    count_pw+=1
start_index = []
for ix in range(1, len(P)+1):
    start_index.append([i for i in range(len(Pw)) if all(j ==ix for j in Pw[i])][0])


def find_indices(search_list, search_item):
    indices = []
    for (index, item) in enumerate(search_list):
        if item == search_item:
            indices.append(index)
    return indices


## possible arrivals inpatient probability distrubtion given current number of inpatients
pos_arrivals_nrpat = []
pos_arrivals_cdf_nrpat = []
pos_arrivals_prob_nrpat =[]
prob_nr_cdf_tot = []
nr_tot = []

for nr_pat in range(max_nr_i+1):  #determine of each given number of inpatient currently
    pos_arrivals_c = []
    pos_arrivals_cdf_c = []
    pos_arrivals_prob_c = []
    prob_nrr = []
    nr = []
    for c in range(C):
        prob_tot = []
        prob_tot_x = [[]]
        for i in range(max(0, min(10, max_nr_i-nr_pat))+1): #get all possible arrivals
            if i==  max(0, max_nr_i-nr_pat):
                prob_nr = poisson.sf(i-1, lam_i)
            else:
                prob_nr = poisson.pmf(i, lam_i)
            prob_nrr.append(prob_nr)
            nr.append(i)
            if i == 0:
                prob_tot_x =   [np.sum(list(it.product([[q[c]] for q in Q], repeat=int(i))), axis = 1)]
                prob_tot = np.prod(list(it.product(Q_dist, repeat=int(i))), axis = 1)*prob_nr
                
            else: 
                prob_tot_x =  np.concatenate((prob_tot_x, np.sum(list(it.product([[q[c]] for q in Q],  repeat=int(i))), axis = 1)), axis = 0)
                prob_tot = np.concatenate((prob_tot, np.prod(list(it.product(Q_dist, repeat=int(i))), axis = 1)*prob_nr), axis = 0)
                
            
        total_list_ar_prob = prob_tot
        total_list_ar = prob_tot_x
        pos_arrivals = []
        pos_arrivals_prob = []
        for m in total_list_ar:
            if len(find_indices(pos_arrivals, m)) ==0:
                ind = find_indices(total_list_ar, m)
                ss = 0  
                for ii in ind:
                    ss += total_list_ar_prob[ii]
                pos_arrivals_prob.append(np.array([ss]))
                pos_arrivals.append(m)

        combined_lists = list(zip(pos_arrivals, pos_arrivals_prob))
        sorted_lists = sorted(combined_lists, key=lambda x: x[0])

        pos_arrivals_c.append([item[0] for item in sorted_lists])
        pos_arrivals_prob_c.append([item[1] for item in sorted_lists])

        pos_arrivals_cdf_c.append([np.sum(pos_arrivals_prob_c[c][:i+1]) for i in range(len(pos_arrivals_prob_c[c]))])
    pos_arrivals_nrpat.append(pos_arrivals_c)
    pos_arrivals_cdf_nrpat.append(pos_arrivals_cdf_c)
    pos_arrivals_prob_nrpat.append(pos_arrivals_prob_c)
    nr_tot.append(nr)
    prob_nr_cdf_tot.append([np.sum(prob_nrr[:i+1]) for i in range(len(prob_nrr))])
    # exp_pos_arrival= np.sum([pos_arrivals_prob[i]*np.sum(pos_arrivals[i]) for i in range(len(pos_arrivals_prob))])


def getPosArrivals(inpatient_list, timeinterval):
    """
        Returns probability distribution of demand of new inpatient arrivals
    """
    nr_pat = np.zeros((timeinterval), dtype = np.int16)
    for pat in inpatient_list:
        for t in range(timeinterval):
            if pat.L  > t:
                nr_pat[t]+=1
    pos_ar_t = []
    pos_ar_prob_t = []
    pos_ar_cdf_t = []

    for t in range(timeinterval):
        pos_ar_t.append(pos_arrivals_nrpat[nr_pat[t]])
        pos_ar_cdf_t.append(pos_arrivals_cdf_nrpat[nr_pat[t]])
    
    return pos_ar_t, pos_ar_cdf_t

def getPosOptions_O(outpatient_list, timeinterval):
    """
        Returns probaility distribution of demenad of decision moment outpatinet
    """
    pos_options_totaal = [[[np.array([0.])] for c in range(C)] for i in range(T_MDO_o+1)]
    pos_options_prob_totaal = [[[np.array([1.])] for c in range(C)]for i in range(T_MDO_o+1)]
    pos_options_cdf_totaal = [[[np.array([1.])] for c in range(C)] for i in range(T_MDO_o+1)]

    for t in range(timeinterval-T_MDO_o-1):
        pat_opt = []
        pat_opt_prob = []

        for pat in outpatient_list:
            if pat.L == T_MDO_o+t:
                pat_opt.append([P[pat.fullTreatmentPlan[pat.currentMDO]-1], np.zeros((C))])
                pp = probs_o[pat.currentMDO]
                pat_opt_prob.append([pp, 1-pp])

        if len(pat_opt)==0:
            pos_options_totaal.append([[np.array([0.])] for c in range(C)]   )
            pos_options_prob_totaal.append([[np.array([1.])] for c in range(C)]  )
            pos_options_cdf_totaal.append([[np.array([1.])] for c in range(C)]  )
        else:
            xx= np.sum(list(it.product(*pat_opt)), axis = 1)
            yy = np.prod(list(it.product(*pat_opt_prob)), axis = 1)
                    
            opt_c =[]
            cdf_c = []
            prob_c=[]
            for c in range(C): 

                pos_options = []
                pos_options_prob = []

                for m in [xx[i][c] for i in range(len(xx))]:
                    if len(find_indices(pos_options, m)) ==0:
                        ind = find_indices([xx[i][c] for i in range(len(xx))], m)
                        ss = np.zeros((1))
                        for ii in ind:
                            ss+=  yy[ii]

                        pos_options.append(np.array([m]))
                        pos_options_prob.append(ss)

                combined_lists = list(zip(pos_options, pos_options_prob))
                sorted_lists = sorted(combined_lists, key=lambda x: x[0])

                opt_c.append([item[0] for item in sorted_lists])
                pos_options_prob = [item[1] for item in sorted_lists]
                prob_c.append(pos_options_prob)
                cdf_c.append([np.sum(pos_options_prob[:i+1]) for i in range(len(pos_options_prob))])

            pos_options_totaal.append(opt_c)
            pos_options_cdf_totaal.append(cdf_c)
            pos_options_prob_totaal.append(prob_c)
    return pos_options_totaal, pos_options_prob_totaal,  pos_options_cdf_totaal
        
def getPosOptions_I(inpatient_list, timeinterval):
    """
        Returns probaility distribution of demenad of decision moment inpatient
    """
    pos_options_totaal = [[[np.array([0.])] for c in range(C)] for i in range(T_MDO_i+1)]
    pos_options_prob_totaal = [[[np.array([1.])] for c in range(C)]for i in range(T_MDO_i+1)]
    pos_options_cdf_totaal = [[[np.array([1.])] for c in range(C)] for i in range(T_MDO_i+1)]

    for t in range(timeinterval-T_MDO_i-1):
        pat_opt = []
        pat_opt_prob = []

        for pat in inpatient_list:
            if pat.L == T_MDO_i+t:
                pat_opt.append([Q[pat.fullTreatmentPlan[pat.currentMDO]-1], np.zeros((C))])
                pp = probs_i[pat.currentMDO]
                pat_opt_prob.append([pp, 1-pp])

        if len(pat_opt)==0:
            pos_options_totaal.append([[np.array([0.])] for c in range(C)]   )
            pos_options_prob_totaal.append([[np.array([1.])] for c in range(C)]  )
            pos_options_cdf_totaal.append([[np.array([1.])] for c in range(C)]  )
        else:
            xx= np.sum(list(it.product(*pat_opt)), axis = 1)
            yy = np.prod(list(it.product(*pat_opt_prob)), axis = 1)

            opt_c =[]
            cdf_c = []
            prob_c =[]
            for c in range(C): 

                pos_options = []
                pos_options_prob = []
                for m in [xx[i][c] for i in range(len(xx))]:
                    if len(find_indices(pos_options, m)) ==0:
                        ind = find_indices([xx[i][c] for i in range(len(xx))], m)
                        ss = np.zeros((1))
                        for ii in ind:
                            ss+=  yy[ii]

                        pos_options.append(np.array([m]))
                        pos_options_prob.append(ss)

                combined_lists = list(zip(pos_options, pos_options_prob))
                sorted_lists = sorted(combined_lists, key=lambda x: x[0])

                opt_c.append([item[0] for item in sorted_lists])
                pos_options_prob = [item[1] for item in sorted_lists]
                prob_c.append(pos_options_prob)
                cdf_c.append([np.sum(pos_options_prob[:i+1]) for i in range(len(pos_options_prob))])

            pos_options_totaal.append(opt_c)
            pos_options_prob_totaal.append(prob_c)
            pos_options_cdf_totaal.append(cdf_c)
    return pos_options_totaal, pos_options_prob_totaal, pos_options_cdf_totaal

def getProbCombination(list1, list2, timeinterval):
    """
        Returns joint probability distribution of two diistrbutions
    """
    results = []
    for t in range(timeinterval): # ASSUMES T_MDO_i = T_MDO_o
        xx = np.sum(list(it.product(list1[t][0], list2[t][0])), 1)
        yy = np.prod(list(it.product(list1[t][1], list2[t][1])), axis = 1)
        if len(xx)==1:
            pos_options_comb =np.array(xx)
            pos_options_comb_prob = np.array(yy)
            
        else: 
            pos_options_comb = []
            pos_options_comb_prob =[]
            for m in xx:
                if len(find_indices(pos_options_comb, m)) ==0:
                    ind = find_indices(xx, m)
                    ss = 0
                    for ii in ind:
                        ss+= yy[ii]
                    pos_options_comb.append(m)
                    pos_options_comb_prob.append(ss)
        combined_lists = list(zip(pos_options_comb, pos_options_comb_prob))
        sorted_lists = sorted(combined_lists, key=lambda x: x[0])

        pos_options_comb = [item[0] for item in sorted_lists]
        pos_options_comb_prob = [item[1] for item in sorted_lists]
        
        results.append([pos_options_comb,pos_options_comb_prob ])
    return results

def MarkovChainDecision(ranges):
    xx =uniform.rvs(0,1)
    for rr in range(len(ranges)):
        if xx< ranges[rr]:
            return rr

class Outpatient():
    """ Defines a outpatient, with random assigned properties"""
    def __init__(self, name):
        self.name = name
        self.nrMDO = MarkovChainDecision(ranges_o)
        
        if self.nrMDO > max_cont_o:
            self.nrMDO = max_cont_o
        self.LoS = T_start_o+(T_start_o-T_MDO_o+1)*(self.nrMDO)
        self.fullTreatmentPlan = [rng.choice(range(1, len(P)+1), size = 1, p = P_dist)[0]]*(self.nrMDO+1)
        
        self.urgency = np.random.choice(U,size = 1, p= U_dist)[0]
        self.urgencyInitial = copy.copy(self.urgency)
        self.waitingList = True
        self.stay = 0
        self.timeWaitingList = 0
    
    def notAdmitted(self):
        """Increase urgency level if not admitted"""
        if self.urgency!= U-1:
            self.urgency+= 1
        self.timeWaitingList += 1
        
    def getAdmitted(self):
        """Admit patient and set properties accordingly"""
        if self.waitingList:
            self.waitingList =False
            self.currentPlan = self.fullTreatmentPlan[0]
            self.currentPw = start_index[self.currentPlan-1]
            self.currentMDO = 0
            self.L = T_start_o
        else: 
            #provide error if something went wrong
            print('patient already admitted: '+str(self.name))

    def nextWeek(self):
        """Update patient properties if system tranfers to next week."""
        if not self.waitingList:
            if self.LoS!= self.stay:
                if self.L == T_MDO_o: # decision moment!
                    if self.currentMDO!= self.nrMDO:
                        self.stay+= 1
                        self.currentMDO += 1
                        self.currentPlan = self.fullTreatmentPlan[self.currentMDO]
                        self.currentPw = Pw_index[self.currentPw][self.currentPlan-1]
                        self.L= T_start_o
                    else: 
                        self.stay +=1
                        self.currentPw = Pw_index[self.currentPw][self.currentPlan-1]
                        self.L -= 1
                else:
                    self.stay += 1
                    self.L-= 1
                    if self.L<0:
                        print('false')
                        print(self.nextMDO, self.stay)
                        raise ValueError()
                    self.currentPw = Pw_index[self.currentPw][self.currentPlan-1]
                    
    def returnDemand(self, time):
        """ Returns demand that patient will have for the remaining of its stay"""
        demand = []
        current_demand = P[self.currentPlan-1]     
        for t in range(min(time, self.LoS-self.stay+1)):      
            if self.L == T_MDO_o+t and self.currentMDO!= self.nrMDO-1:
                current_demand = P[self.fullTreatmentPlan[self.currentMDO]-1]
            demand.append(current_demand)
        return demand
    
    def returnDemandCURRENT(self):
        """Returns the current demand of the patient"""
        return  P[self.currentPlan-1] 
    
    def returnDemandKNOWN(self):
        """Returns the demand that is known to the planner for given patient"""
        demand = []
        current_demand = P[self.currentPlan-1]     
        for t in range(min(self.LoS-self.stay+1, self.L+1)):      
            if self.L == T_MDO_o+t and self.currentMDO!= self.nrMDO-1:
                current_demand = P[self.fullTreatmentPlan[self.currentMDO]-1]
            demand.append(current_demand)
        return demand
    
    def totalDemand(self):
        ll = len(self.fullTreatmentPlan)
        demand =[np.zeros((C))]
        for pp in range(ll):
            if pp == ll-1:
                demand =np.concatenate((demand, [P[self.fullTreatmentPlan[pp]-1]]*(T_start_o+1)), axis = 0)
            else:
                demand= np.concatenate((demand, [P[self.fullTreatmentPlan[pp]-1]]*(T_start_o-T_MDO_o+1)), axis = 0)
        
        return demand[1:]
    
def makeOutPatients(T, name):
    """Returns a random list of outpatient for given timeinterval @T"""
    arrivals = []
    for t in range(T):
        arrivals.append([Outpatient(str(name)+'-'+str(t)+'-'+str(x)) for x in range(np.random.poisson(lam= lam_o))])
    return arrivals

class Inpatient():
    """ Defines an inpatient, with random assigned properties"""
    def __init__(self, name):
        self.name = name
        self.nrMDO = MarkovChainDecision(ranges_i)
        if self.nrMDO > max_cont_i:
            self.nrMDO =max_cont_i
        self.currentMDO = 0
        self.LoS = T_start_i+(T_start_i-T_MDO_i+1)*(self.nrMDO)
        self.fullTreatmentPlan = [rng.choice(range(1, len(Q)+1), size = 1, p = Q_dist)[0]]*(self.nrMDO+1)
        self.stay =0
        self.L = T_start_i
        self.currentPlan = self.fullTreatmentPlan[0]
        
    def nextWeek(self):
        """Update patient properties if system tranfers to next week."""
        if self.LoS != self.stay:
            if self.L == T_MDO_i:
                if self.currentMDO != self.nrMDO:
                    self.stay += 1
                    self.currentMDO += 1
                    self.currentPlan = self.fullTreatmentPlan[self.currentMDO]
                    self.L = T_start_i
                else: 
                    self.stay += 1
                    self.L -=1
            else: 
                self.stay+=1
                self.L -=1
                if self.L <0:
                    print('false')
                    print(self.stay, self.LoS)
                    raise ValueError()
        else:
            if self.L !=0: raise ValueError()
                    
    def returnDemand(self, time):
        """return demand for remaining stay of patient"""
        demand = []
        current_demand = Q[self.currentPlan-1]
        #HOUD NIET REKENING MET VERANDERINGEN BEHANDELPLAN!
        for t in range(min(time, self.LoS-self.stay+1)):
            demand.append(current_demand)
        return demand
    
    def returnDemandCURRENT(self):
        """return the demand in the current week regarding the patient"""
        return  Q[self.currentPlan-1]

    def returnDemandKNOWN(self):
        """return the demand that is know to the planner of this patients"""
        demand = []
        current_demand = Q[self.currentPlan-1]     
        for t in range(min(self.LoS-self.stay+1, self.L+1)):      
            if self.L == T_MDO_i+t and self.currentMDO!= self.nrMDO-1:
                current_demand = Q[self.fullTreatmentPlan[self.currentMDO]-1]
            demand.append(current_demand)
        return demand
    
def makeInPatients(T, name):
    """
    Returns a random list of outpatient for given timeinterval
    :param int T: timeinterval to be considered
    """
    
    arrivals = []
    for t in range(T):
        arrivals.append([Inpatient(str(name)+'-'+str(t)+'-'+str(x)) for x in range(np.random.poisson(lam= lam_i))])
    return arrivals
        
def getPossibleActions(patient_waiting_list):
    """
    Return array with all possible actions given the waitting list
    """
    options = []
    for r in range(len(patient_waiting_list)+1):
        for x in list(it.combinations(patient_waiting_list, r)):
            options.append(x)
    return options

def updatePatientsLists(patient_waiting_list, outpatient_list, inpatient_list, trj_outp_t, trj_inp_t, action):
    """
    Update waitinglist, outpatient list and inpatient list given the new arrivals and action to be taken
    """
    toBeRemoved = []

    for pat in outpatient_list:
        if pat.LoS != pat.stay:
            pat.nextWeek()
        else:
            if pat.L >0:
                raise ValueError()
            toBeRemoved.append(pat)
    for pat in toBeRemoved:       
        outpatient_list.remove(pat)
        del pat
    toBeRemoved = []
    for pat in inpatient_list:
        if pat.LoS!= pat.stay:
            pat.nextWeek()
        else: 
            if pat.L >0:
                raise ValueError()
            toBeRemoved.append(pat)
    for pat in toBeRemoved:       
        inpatient_list.remove(pat)
        del pat       

    for x in action: 
        x.getAdmitted()
        patient_waiting_list.remove(x)
        outpatient_list.append(x)
    
    for pat in patient_waiting_list:
        pat.notAdmitted()
    
    for x in trj_outp_t:
        patient_waiting_list.append(x)

    for x in trj_inp_t:
        if len(inpatient_list)< max_nr_i:
            inpatient_list.append(x)

    return patient_waiting_list, outpatient_list, inpatient_list

def chooseRandomAction(patient_waiting_list):
    """
    return a random action out of the possible action that are possible given the waiting list
    """
    possible_actions = getPossibleActions(patient_waiting_list)
    return random.choice(possible_actions)

def makeInitialization_random(T =100):
    """
    Run a simulation for given timeinterval where each action is taken randomly, output the final waitinglist, inpatient list and outpatient list
    """
    patient_waiting_list = []
    outpatient_list = []
    inpatient_list = []

    init_T = T

    trj_inp = makeInPatients(init_T, 'init')
    trj_outp = makeOutPatients(init_T, 'init')
    for t in range(init_T):
        action =  chooseRandomAction(patient_waiting_list)
        patient_waiting_list, outpatient_list, inpatient_list = updatePatientsLists(patient_waiting_list,outpatient_list, inpatient_list, trj_outp[t], trj_inp[t], action)

    return patient_waiting_list, outpatient_list, inpatient_list

def makeInitialization_zeros():
    """Defines a empty waiting list , outpatient list and inpatient list"""
    return [],[],[]

def getDemandNow(outpatient_list, inpatient_list):
    """Return the current demand given the current admitted patients, array of size C"""
    demand = np.zeros((C))
    for pat in outpatient_list:
        dd = pat.returnDemandCURRENT()
        for c in range(C):
            demand[c] += dd[c]
    for pat in inpatient_list:
        dd = pat.returnDemandCURRENT()
        for c in range(C):
            demand[c] += dd[c]
    return demand

def getDemandList_O(outpatient_list, timeinterval):
    """Returns the demand of outpatients that is known to the planner for each week in the provided timeinterval"""
    demand = np.zeros((timeinterval, C))
    for pat in outpatient_list:
        dd = pat.returnDemandKNOWN()
        for i in range(min(len(dd), timeinterval)):
            demand[i] =np.add(demand[i], dd[i])
            # for c in range(C):
            #     demand[i][c] += dd[i][c]
    return demand

def getDemandList_I(inpatient_list, timeinterval):
    """Returns the demand of inpatients that is known to the planner for each week in the provided timeinterval"""
    demand = np.zeros((timeinterval, C))
    for pat in inpatient_list:
        dd = pat.returnDemandKNOWN()
        for i in range(min(len(dd), timeinterval)):
            for c in range(C):
                demand[i][c] += dd[i][c]
    return demand

def getRealDemandInpatient(current_inpatient_list, trj_inp, timeinterval):
    """Returns the demand of inpatients in reality for each week in the provided timeinterval"""
    demand_pats = np.zeros((timeinterval, C))
    nr_pats = np.zeros((timeinterval))
    for pat in current_inpatient_list:
        xx = pat.returnDemand(timeinterval)
        for t_d in range(min(len(xx), timeinterval)):
            demand_pats[t_d] = np.add(demand_pats[t_d], xx[t_d])
            nr_pats[t_d] = np.add(nr_pats[t_d], 1)
    for t in range(timeinterval-1):
        for pat in trj_inp[t]:
            if nr_pats[t+1] < max_nr_i:
                xx = pat.returnDemand(timeinterval)
                for t_d in range(t+1, min(len(xx)+t+1, timeinterval)):
                    demand_pats[t_d] = np.add(demand_pats[t_d], xx[t_d-t-1])
                    nr_pats[t_d] = np.add(nr_pats[t_d], 1)                   
    return demand_pats

def getRealDemandOutpatient(outpatient_list, timeinterval):
    """Returns the demand of outpatients in reality for each week in the provided timeinterval"""
    demand_pats = np.zeros((timeinterval,C))
    for pat in outpatient_list:
        xx = pat.returnDemand(timeinterval)  
        for t_d in range(min(len(xx),timeinterval)):
            demand_pats[t_d] = np.add(demand_pats[t_d], xx[t_d])
    return demand_pats

def getDemandNew_Inpatient(inpatient_list, trj_inp, timeinterval):
    """return the demand of inpatients that are not yet admitted for each week in the provided timeinterval"""
    demand_pats = np.zeros((timeinterval, C))
    nr_pats = np.zeros((timeinterval))
    for pat in inpatient_list:
        xx = pat.returnDemand(timeinterval)
        for t_d in range(min(len(xx), timeinterval)):
            nr_pats[t_d] = np.add(nr_pats[t_d], 1)
    for t in range(timeinterval-1):
        for pat in trj_inp[t]:
            if nr_pats[t+1] < max_nr_i:
                xx = pat.returnDemand(timeinterval)
                for t_d in range(t+1, min(t+T_start_i+1, timeinterval)):
                    demand_pats[t_d] = np.add(demand_pats[t_d], xx[t_d-t-1])
                    nr_pats[t_d] = np.add(nr_pats[t_d], 1)                   
                    
    return demand_pats

def getDemandInpatient_withoutNew(current_inpatient_list, timeinterval):
    """return the demand of inpatients in reality without the demand resulting from new patients admission in the provided timeinterval"""
    demand_pats = np.zeros((timeinterval, C))
    for pat in current_inpatient_list:
        xx = pat.returnDemand(timeinterval)

        for t_d in range(min(len(xx), timeinterval)):
            demand_pats[t_d] = np.add(demand_pats[t_d], xx[t_d])
    return demand_pats

def getDemandDecision_O(outpatient_list):
    """"return the demand of outpatients that have a decision moment provided per week"""
    demand = np.zeros((len(L_o),C))
    for t in range(len(L_o)):
        for pat in outpatient_list:
            if pat.L-t == T_MDO_o and pat.currentMDO != max_cont_o-1:
                dd = pat.returnDemandCURRENT()
                for c in range(C):
                    demand[t][c] += dd[c]
    return demand

def getDemandDecision_I(inpatient_list):
    """"return the demand of inpatients that have a decision moment provided per week"""
    
    demand = np.zeros((len(L_i),C))
    for t in range(len(L_i)):
        for pat in inpatient_list:
            if pat.L-t == T_MDO_i and pat.currentMDO != max_cont_i-1:
                dd = pat.returnDemandCURRENT()
                for c in range(C):
                    demand[t][c] += dd[c]
    return demand

def expDiff(demand, capacity):
    """Return a exponetinal difference between the provided demand and capacity"""
    capCost =0
    for c in range(C):
        capCost+= C_cost[c]*(np.exp(0.5*max(0,demand[c] - capacity[c]))-1)
    return capCost

def absDiff(demand, capacity):
    """Return a absolute difference between the provided demand and capacity"""
    
    capCost =0
    for c in range(C):
        capCost+= C_cost[c]*max(0,demand[c] - capacity[c])
    return capCost

def absDiff2(demand, capacity):
    """Return a quadratic difference between the provided demand and capacity"""
    capCost =0
    for c in range(C):
        capCost+= C_cost[c]*max(0,demand[c] - capacity[c])**2
    return capCost

def factorDiff(demand, capacity):
    """Return a linearly estimated quatric difference between the provided demand and capacity"""
    capCost = 0
    for c in range(C):
        diff = demand[c] - capacity[c]
        if diff > 0: capCost += diff
        if diff > 5: capCost += 2*(diff-5)
        if diff > 10: capCost += 3*(diff-10)
        # else: capCost  += 15
    return capCost

def getCost(patient_waiting_list, outpatient_list, inpatient_list, chosen_action):
    """Return the cost for the current week, given the action chosen"""
    post_decision_waiting_list = []
    for pat in patient_waiting_list:
        if not any([pat.name ==pat2.name for pat2 in chosen_action]):
            post_decision_waiting_list.append(pat)
    waitingCost = sum([U_cost[pat.urgency] for pat in post_decision_waiting_list])
    lijst = getDemandNow(outpatient_list, inpatient_list)
    capCost = absDiff(lijst, f)
    return waitingCost+ capCost
    
def PARA_demandDecision_O(outpatient_list, timeinterval, theta):
    """Return the extra demand resulting from outpatient decision moments given the parameter value"""
    demand = np.zeros((timeinterval, C))
    xx = getDemandDecision_O(outpatient_list)
    for tt in range(len(xx)):
        for t in range(T_MDO_o+1, min(timeinterval-tt,T_MDO_o+len(L_o))):
            for c in range(C):
                demand[tt+t][c] += theta*xx[tt][c] 
    return demand

def PARA_demandDecision_I(inpatient_list, timeinterval, theta):
    """Return the extra demand resulting from inpatient decision moments given the parameter value"""
    
    demand = np.zeros((timeinterval, C))
    xx = getDemandDecision_I(inpatient_list)
    for tt in range(len(xx)):
        for t in range(T_MDO_i+1, min(timeinterval-tt,T_MDO_i+len(L_i))):
            for c in range(C):
                demand[tt+t][c] += theta*xx[tt][c] 
    return demand

def PARA_demandDecision_I_seperate(inpatient_list, timeinterval, theta):
    """Return the extra demand resulting from outpatient decision moments given the parameter value, seperate for each discipline"""
    
    demand = np.zeros((timeinterval, C))
    xx = getDemandDecision_I(inpatient_list)
    for tt in range(len(xx)):
        for t in range(T_MDO_i+1, min(timeinterval-tt,T_MDO_i+len(L_i))):
            for c in range(C):
                demand[tt+t][c] += theta[c]*xx[tt][c] 
    return demand


def PARA_decoutp_NEWS(patient_waiting_list, inpatient_list, outpatient_list, inp_trj, outp_trj, timeinterval, alpha):
    """
    Return the extra demand resulting from outpatient decision moments determined by the newsvendor model,
    seperate for each discipline
    """
    
    thetas =[]
    rr_tot = []
    pos_options_totaal,_, pos_options_cdf_totaal = getPosOptions_O(outpatient_list, timeinterval)
    current_demands = getRealDemandInpatient(inpatient_list, inp_trj, timeinterval)+ getDemandList_O(outpatient_list, timeinterval) 

    for t in range(timeinterval):
        theta_c = []

        total_urgency = 0 
        total_action_demand = np.zeros((C))
        for pat in patient_waiting_list:
            total_urgency += U_cost[pat.urgency]
            total_action_demand[:] += P[pat.fullTreatmentPlan[0]-1]
        for tt in range(t):
            for pat in outp_trj[tt]:
                total_urgency += U_cost[pat.urgency]
                total_action_demand[:] += P[pat.fullTreatmentPlan[0]-1]
        for c in range(C):
            pos_options = pos_options_totaal[t][c]
            pos_options_cdf = pos_options_cdf_totaal[t][c]    
            if len(thetas) == 0:     
                current_demand = current_demands[c][t]
            else:
                current_demand = current_demands[t][c]+ np.sum(thetas, axis = 0)[c]

            if f[c]-current_demand-total_action_demand[c] >np.min(pos_options, axis=0) :
                if f[c]-current_demand-total_action_demand[c] < np.max(pos_options, axis=0):
                    pr = [pos_options_cdf[i] for i in range(len(pos_options)) if all(pos_options[i] <= f[c]-current_demand-total_action_demand[c])][-1]
                else: pr = 1
            else: pr  =0

            if total_action_demand[c] == 0 or ((total_urgency/total_action_demand[c])+C_cost[c]*(1-pr)) == 0:
                theta1 = C_cost[c]*(1-pr)/(C_cost[c]+C_cost[c]*(1-pr))
                theta2 = alpha*pr*C_cost[c]/(alpha*pr*C_cost[c]+alpha*pr*C_cost[c]+C_cost[c]*(1-pr))
            else: 
                theta1 = C_cost[c]*(1-pr)/((total_urgency/total_action_demand[c])+C_cost[c]*(1-pr))
                theta2 = alpha*pr*C_cost[c]/(alpha*pr*(total_urgency/total_action_demand[c])+alpha*pr*C_cost[c]+C_cost[c]*(1-pr))

            xp1 = [pos_options[i] for i in range(len(pos_options_cdf)) if pos_options_cdf[i]>=theta1]
            theta1_f = xp1[0] 
            xp2 = [pos_options[i] for i in range(len(pos_options_cdf)) if pos_options_cdf[i]>=theta2]
            theta2_f = xp2[0]
            
            if current_demand+total_action_demand[c]+theta1_f < f[c]:
                theta_c.append(theta2_f)
                if np.max(pos_options, axis=0)==0: 
                    rr = np.array([1])
                else: rr = theta2_f/np.max(pos_options, axis=0)
                rr_tot.append(rr)
            else: 
                theta_c.append(theta1_f)
                if np.max(pos_options, axis=0)==0: 
                    rr = np.array([1])
                else: rr = theta1_f/np.max(pos_options, axis=0)
                rr_tot.append(rr)
        thetas.append(theta_c)
    theta_sum = np.zeros((timeinterval, C))
    for i in range(timeinterval):
        for c in range(C):
            theta_sum[i][c] = np.sum(thetas[:i+1], axis = 0)[c]
    return current_demands + theta_sum#, rr_tot

def PARA_decinp_NEWS(patient_waiting_list, inpatient_list, outpatient_list, inp_trj, outp_trj, timeinterval, alpha):
    """
    Return the extra demand resulting from inpatients decision moments determined by the newsvendor model,
    seperate for each discipline
    """
    thetas =[ [np.zeros((1)) for c in range(C)]]
    rr_tot = []
    pos_options_totaal, _,pos_options_cdf_totaal = getPosOptions_I(inpatient_list, timeinterval)
    current_demands = getRealDemandOutpatient(outpatient_list, timeinterval) + getDemandList_I(inpatient_list, timeinterval) +getDemandNew_Inpatient(inpatient_list, inp_trj, timeinterval)
    for t in range(timeinterval-1):
        theta_c = []

        total_urgency = 0 
        total_action_demand = np.zeros((C))
        for pat in patient_waiting_list:
            total_urgency += U_cost[pat.urgency]
            total_action_demand[:] += P[pat.fullTreatmentPlan[0]-1]
        for tt in range(t):
            for pat in outp_trj[tt]:
                total_urgency += U_cost[pat.urgency]
                total_action_demand[:] += P[pat.fullTreatmentPlan[0]-1]
        for c in range(C):
            pos_options = pos_options_totaal[t][c]
            pos_options_cdf = pos_options_cdf_totaal[t][c]    
            if len(thetas) == 0:     
                current_demand = current_demands[t+1][c]
            else:
                current_demand = current_demands[t+1][c]+ np.sum(thetas, axis = 0)[c]

            if f[c]-current_demand-total_action_demand[c] >np.min(pos_options, axis=0) :
                if f[c]-current_demand-total_action_demand[c] < np.max(pos_options, axis=0):
                    pr = [pos_options_cdf[i] for i in range(len(pos_options)) if all(pos_options[i] <= f[c]-current_demand-total_action_demand[c])][-1]
                else: pr = 1
            else: pr  =0
            if total_action_demand[c] == 0 or ((total_urgency/total_action_demand[c])+C_cost[c]*(1-pr)) == 0:
                theta1 = C_cost[c]*(1-pr)/(C_cost[c]+C_cost[c]*(1-pr))
                theta2 = alpha*pr*C_cost[c]/(alpha*pr*C_cost[c]+alpha*pr*C_cost[c]+C_cost[c]*(1-pr))
            else: 
                theta1 = C_cost[c]*(1-pr)/((total_urgency/total_action_demand[c])+C_cost[c]*(1-pr))
                theta2 = alpha*pr*C_cost[c]/(alpha*pr*(total_urgency/total_action_demand[c])+alpha*pr*C_cost[c]+C_cost[c]*(1-pr))
            xp1 = [pos_options[i] for i in range(len(pos_options_cdf)) if pos_options_cdf[i]>=theta1]
            theta1_f = xp1[0] 
            xp2 = [pos_options[i] for i in range(len(pos_options_cdf)) if pos_options_cdf[i]>=theta2]
            theta2_f = xp2[0]
            
            if current_demand+total_action_demand[c]+theta1_f < f[c]:
                theta_c.append(theta2_f)
                if np.max(pos_options, axis=0)==0: 
                    rr = np.array([1])
                else: rr = theta2_f/np.max(pos_options, axis=0)
                rr_tot.append(rr)
            else: 
                theta_c.append(theta1_f)
                if np.max(pos_options, axis=0)==0: 
                    rr = np.array([1])
                else: rr = theta1_f/np.max(pos_options, axis=0)
                rr_tot.append(rr)
        thetas.append(theta_c)
    theta_sum = np.zeros((timeinterval, C))
    for i in range(timeinterval):
        for c in range(C):
            theta_sum[i][c] = np.sum(thetas[:i+1], axis = 0)[c]
    return current_demands + theta_sum#, rr_tot

def PARA_newDemand_inp(theta, timeinterval):
    dd = np.zeros((timeinterval,C))
    for t in range(timeinterval):
        for c in range(C):
            dd[t][c] = np.sum(theta[c]*t, axis = 0)
    return dd

def getExpectationMin(pos_prob, pos_values,  insert_):
    exp = 0
    for val in range(len(pos_values)):
        if pos_values[val] <= insert_:
            exp += pos_prob[val]*pos_values[val]
        else: exp += pos_prob[val]*insert_
    return exp
def getExpectation(pos_prob, pos_values):
    exp = 0
    for val in range(len(pos_values)):
        exp += pos_prob[val]*pos_values[val]

    return exp

def PARA_newDemand_inp_NEWS_iter(patient_waiting_list, inpatient_list, outpatient_list, outp_trj, timeinterval, alpha):
    N_iteration = 10
    thetas =[ [np.zeros((1)) for c in range(C)]]
    current_demands = getRealDemandOutpatient(outpatient_list, timeinterval) + getDemandInpatient_withoutNew(inpatient_list, timeinterval)
    
    nr_pat = np.zeros((timeinterval), dtype = np.int16)
    for pat in inpatient_list:
        for t in range(timeinterval):
            if pat.L  > t:
                nr_pat[t]+=1
    
    extra_pat = 0
    for t in range(timeinterval-1):
        total_urgency = 0 
        total_action_demand = np.zeros((C))
        for pat in patient_waiting_list:
            total_urgency += U_cost[pat.urgency]
            total_action_demand += P[pat.fullTreatmentPlan[0]-1]
        if t>0:
            for tt in range(t):
                for pat in outp_trj[tt]:
                    total_urgency += U_cost[pat.urgency]
                    total_action_demand += P[pat.fullTreatmentPlan[0]-1]
        theta_c =[]
        extra_pat_c = 0
        pr_c = []
        for c in range(C):
            pos_arrivals = pos_arrivals_nrpat[nr_pat[t]+extra_pat][c]
            pos_arrivals_cdf = pos_arrivals_cdf_nrpat[nr_pat[t]+extra_pat][c]
 
            if len(thetas) == C:     
                current_demand = current_demands[c][t]
            else:
                current_demand = current_demands[t][c]+ np.sum(thetas, axis = 0)[c]

            if f[c]-current_demand-total_action_demand[c] >np.min(pos_arrivals, axis=0) :

                if f[c]-current_demand-total_action_demand[c] < np.max(pos_arrivals, axis=0):
                    pr = [pos_arrivals_cdf[i] for i in range(len(pos_arrivals)) if pos_arrivals[i] <= f[c]-current_demand-total_action_demand[c]][-1]

                else: pr = 1
            else: pr  =0
            pr_c.append(pr)
            if total_action_demand[c] == 0 or ((total_urgency/total_action_demand[c])+C_cost[c]*(1-pr)) == 0:
                theta1 = C_cost[c]*(1-pr)/(C_cost[c]+C_cost[c]*(1-pr))
                theta2 = alpha*pr*C_cost[c]/(alpha*pr*C_cost[c]+alpha*pr*C_cost[c]+C_cost[c]*(1-pr))

                
            else: 
                theta1 = C_cost[c]*(1-pr)/((total_urgency/total_action_demand[c])+C_cost[c]*(1-pr))
                theta2 = alpha*pr*C_cost[c]/(alpha*pr*(total_urgency/total_action_demand[c])+alpha*pr*C_cost[c]+C_cost[c]*(1-pr))

            theta1_f = [pos_arrivals[i] for i in range(len(pos_arrivals_cdf)) if pos_arrivals_cdf[i]>=theta1][0]
            theta2_f = [pos_arrivals[i] for i in range(len(pos_arrivals_cdf)) if pos_arrivals_cdf[i]>=theta2][0]

            if current_demand+total_action_demand[c]+theta1_f < f[c]:
                theta_c.append(theta2_f)
                extra_pat_c += [nr_tot[nr_pat[t]+extra_pat][i] for i in range(len(prob_nr_cdf_tot[nr_pat[t]+extra_pat])) if prob_nr_cdf_tot[nr_pat[t]+extra_pat][i]>= theta2][0]
            else: 
                theta_c.append(theta1_f)

        extra_pat += round(extra_pat_c/3)
        eta  = 1
        for i in range(N_iteration):
            stop = False
            count = 0
            while True:
                Z = np.random.normal(size = (C,1))
                thetas_random = [np.round(i) for i in theta_c + eta*Z]
                if all(thetas_random[c]<= max(pos_arrivals_nrpat[nr_pat[t]+extra_pat][c]) for c in range(C)) and all(thetas_random[c]>= 0 for c in range(C)):
                    break
                count += 1
                if count == 5:
                    stop = True
                    break
            if stop:
                break
        
            thetas_check = [theta_c, thetas_random]
            
            objs = []
            for th in thetas_check:
                obj = 0
                for c in range(C):
                    current_demand = current_demands[t][c]+ np.sum(thetas, axis = 0)[c]
                    #regel 1
                    if current_demand+total_action_demand[c]+th[c] > f[c]:                     
                        if total_action_demand[c] !=0:
                            obj += (total_urgency/total_action_demand[c])*(-f[c] + (current_demand+total_action_demand[c]+th[c])- getExpectationMin(pos_arrivals_prob_nrpat[nr_pat[t]+extra_pat][c], pos_arrivals_nrpat[nr_pat[t]+extra_pat][c], th[c])[0] + getExpectationMin(pos_arrivals_prob_nrpat[nr_pat[t]+extra_pat][c], pos_arrivals_nrpat[nr_pat[t]+extra_pat][c],f[c]- current_demand-total_action_demand[c]) )
                        else: 
                            obj += (C_cost[c])*(-f[c] + (current_demand+total_action_demand[c]+th[c])- getExpectationMin(pos_arrivals_prob_nrpat[nr_pat[t]+extra_pat][c], pos_arrivals_nrpat[nr_pat[t]+extra_pat][c], th[c])[0] + getExpectationMin(pos_arrivals_prob_nrpat[nr_pat[t]+extra_pat][c], pos_arrivals_nrpat[nr_pat[t]+extra_pat][c],f[c]- current_demand-total_action_demand[c]) )
                            
                    #regel 2
                    if current_demand+total_action_demand[c]+th[c] > f[c]: 
                        obj+= (1-pr_c[c])*C_cost[c]*(getExpectation(pos_arrivals_prob_nrpat[nr_pat[t]+extra_pat][c], pos_arrivals_nrpat[nr_pat[t]+extra_pat][c])[0]- getExpectationMin(pos_arrivals_prob_nrpat[nr_pat[t]+extra_pat][c], pos_arrivals_nrpat[nr_pat[t]+extra_pat][c], th[c])[0])              
                    else:
                        obj+= (1-pr_c[c])*C_cost[c]*(getExpectation(pos_arrivals_prob_nrpat[nr_pat[t]+extra_pat][c], pos_arrivals_nrpat[nr_pat[t]+extra_pat][c])[0]- getExpectationMin(pos_arrivals_prob_nrpat[nr_pat[t]+extra_pat][c], pos_arrivals_nrpat[nr_pat[t]+extra_pat][c], th[c])[0]+f[c] - (current_demand+total_action_demand[c]+th[c]))              
                    
                    
                    #regel 3
                    if current_demand+total_action_demand[c]+th[c] <= f[c]: 
                        if total_action_demand[c] !=0:
                        
                            obj+= alpha*pr_c[c]*((total_urgency/total_action_demand[c])*th[c]+C_cost[c]*getExpectation(pos_arrivals_prob_nrpat[nr_pat[t]+extra_pat][c], pos_arrivals_nrpat[nr_pat[t]+extra_pat][c])[0]+(-1*(total_urgency/total_action_demand[c])-C_cost[c])*getExpectationMin(pos_arrivals_prob_nrpat[nr_pat[t]+extra_pat][c], pos_arrivals_nrpat[nr_pat[t]+extra_pat][c], th[c])[0])
                        else:
                            obj+= alpha*pr_c[c]*((C_cost[c])*th[c]+C_cost[c]*getExpectation(pos_arrivals_prob_nrpat[nr_pat[t]+extra_pat][c], pos_arrivals_nrpat[nr_pat[t]+extra_pat][c])[0]+(-2*C_cost[c])*getExpectationMin(pos_arrivals_prob_nrpat[nr_pat[t]+extra_pat][c], pos_arrivals_nrpat[nr_pat[t]+extra_pat][c], th[c])[0])
                            
                objs.append(obj)
            if objs[1]< objs[0]:
                theta_c = thetas_random
        thetas.append([np.round(th_c) for th_c in theta_c]) 
    theta_sum = np.zeros((timeinterval, C))
    for i in range(timeinterval):
        for c in range(C):
            theta_sum[i][c] = np.sum(thetas[:i+1], axis = 0)[c]
    return current_demands + theta_sum

def PARA_newDemand_inp_NEWS(patient_waiting_list, inpatient_list, outpatient_list, outp_trj, timeinterval, alpha):
    thetas =[ [np.zeros((1)) for c in range(C)]]
    current_demands = getRealDemandOutpatient(outpatient_list, timeinterval) + getDemandInpatient_withoutNew(inpatient_list, timeinterval)
    
    nr_pat = np.zeros((timeinterval), dtype = np.int16)
    for pat in inpatient_list:
        for t in range(timeinterval):
            if pat.L  >= t:
                nr_pat[t]+=1
    
    extra_pat = 0
    for t in range(timeinterval-1):
        total_urgency = 0 
        total_action_demand = np.zeros((C))
        for pat in patient_waiting_list:
            total_urgency += U_cost[pat.urgency]
            total_action_demand[:] += P[pat.fullTreatmentPlan[0]-1]
        if t>0:
            for tt in range(t-1):
                for pat in outp_trj[tt]:
                    total_urgency += U_cost[pat.urgency]
                    total_action_demand[:] += P[pat.fullTreatmentPlan[0]-1]
                

        theta_c =[]
        extra_pat_c = 0
        for c in range(C):
            pos_arrivals = pos_arrivals_nrpat[nr_pat[t]+extra_pat][c]
            pos_arrivals_cdf = pos_arrivals_cdf_nrpat[nr_pat[t]+extra_pat][c]
 
            if len(thetas) == C:     
                current_demand = current_demands[t+1][c]
            else:
                current_demand = current_demands[t+1][c]+ np.sum(thetas, axis = 0)[c]
            if f[c]-current_demand-total_action_demand[c] >np.min(pos_arrivals, axis=0) :

                if f[c]-current_demand-total_action_demand[c] < np.max(pos_arrivals, axis=0):
                    pr = [pos_arrivals_cdf[i] for i in range(len(pos_arrivals)) if pos_arrivals[i] <= f[c]-current_demand-total_action_demand[c]][-1]
                else: pr = 1
            else: pr  =0

            if total_action_demand[c] == 0 or ((total_urgency/total_action_demand[c])+C_cost[c]*(1-pr)) == 0:
                theta1 = C_cost[c]*(1-pr)/(C_cost[c]+C_cost[c]*(1-pr))
                theta2 = alpha*pr*C_cost[c]/(alpha*pr*C_cost[c]+alpha*pr*C_cost[c]+C_cost[c]*(1-pr))
            else: 
                theta1 = C_cost[c]*(1-pr)/((total_urgency/total_action_demand[c])+C_cost[c]*(1-pr))
                theta2 = alpha*pr*C_cost[c]/(alpha*pr*(total_urgency/total_action_demand[c])+alpha*pr*C_cost[c]+C_cost[c]*(1-pr))
            theta1_f = [pos_arrivals[i] for i in range(len(pos_arrivals_cdf)) if pos_arrivals_cdf[i]>=theta1][0]
            theta2_f = [pos_arrivals[i] for i in range(len(pos_arrivals_cdf)) if pos_arrivals_cdf[i]>=theta2][0]
            if current_demand+total_action_demand[c]+theta1_f < f[c]:
                theta_c.append(theta2_f)
            
                extra_pat_c += [nr_tot[nr_pat[t]+extra_pat][i] for i in range(len(prob_nr_cdf_tot[nr_pat[t]+extra_pat])) if prob_nr_cdf_tot[nr_pat[t]+extra_pat][i]>= theta2][0]
            else: 
                theta_c.append(theta1_f)
                extra_pat_c += [nr_tot[nr_pat[t]+extra_pat][i] for i in range(len(prob_nr_cdf_tot[nr_pat[t]+extra_pat])) if prob_nr_cdf_tot[nr_pat[t]+extra_pat][i]>= theta1][0]
                
        extra_pat += round(extra_pat_c/C)
        thetas.append(theta_c)
    theta_sum = np.zeros((timeinterval, C))

    for i in range(timeinterval):
        for c in range(C):
            theta_sum[i][c] = np.sum(thetas[:i+1], axis = 0)[c]
        
    return current_demands + theta_sum

def TUNE_demandDecision_O(outpatient_list, inpatient_list, patient_waiting_list, trj_inp, trj_outp, timeinterval, theta_O, alpha):
    demand_inp = getRealDemandInpatient(inpatient_list,trj_inp, timeinterval)
    demand_outp_known = getDemandList_O(outpatient_list, timeinterval)
    demand_outp_PARA = PARA_demandDecision_O(outpatient_list, timeinterval, theta_O)

    return [[demand_outp_known[tt][c] + demand_outp_PARA[tt][c] + demand_inp[tt][c] for c in range(C)]for tt in range(timeinterval)]

def TUNE_demandDecision_I(outpatient_list, inpatient_list, patient_waiting_list, trj_inp, trj_outp, timeinterval, theta_I, alpha ):
    demand_outp = getRealDemandOutpatient(outpatient_list, timeinterval)

    demand_inp_known = getDemandList_I(inpatient_list, timeinterval)
    demand_inp_new = getDemandNew_Inpatient(inpatient_list, trj_inp, timeinterval)
    demand_inp_PARA = PARA_demandDecision_I(inpatient_list, timeinterval, theta_I)

    return [[demand_outp[tt][c] + demand_inp_known[tt][c] + demand_inp_new[tt][c] + demand_inp_PARA[tt][c] for c in range(C)]for tt in range(timeinterval)]

def TUNE_demandDecision_I_seperate(outpatient_list, inpatient_list, patient_waiting_list, trj_inp, trj_outp, timeinterval, theta_I, alpha ):
    demand_outp = getRealDemandOutpatient(outpatient_list, timeinterval)

    demand_inp_known = getDemandList_I(inpatient_list, timeinterval)
    demand_inp_new = getDemandNew_Inpatient(inpatient_list, trj_inp, timeinterval)
    demand_inp_PARA = PARA_demandDecision_I_seperate(inpatient_list, timeinterval, theta_I)

    return [[demand_outp[tt][c] + demand_inp_known[tt][c] + demand_inp_new[tt][c] + demand_inp_PARA[tt][c] for c in range(C)]for tt in range(timeinterval)]

def TUNE_demandDecision_I_NEWS(outpatient_list, inpatient_list, patient_waiting_list, trj_inp, trj_outp, timeinterval, theta, alpha):
    return PARA_decinp_NEWS(patient_waiting_list, inpatient_list, outpatient_list, trj_inp, trj_outp, timeinterval, alpha)

def TUNE_demandDecision_O_NEWS(outpatient_list, inpatient_list, patient_waiting_list, trj_inp, trj_outp, timeinterval, theta, alpha):
    return PARA_decoutp_NEWS(patient_waiting_list, inpatient_list, outpatient_list, trj_inp, trj_outp, timeinterval, alpha)

def TUNE_newInpatient(outpatient_list, inpatient_list, patient_waiting_list, trj_inp, trj_outp, timeinterval, theta_I_new, alpha):
    demand_outp = getRealDemandOutpatient(outpatient_list, timeinterval)
    demand_inp_known = getDemandInpatient_withoutNew(inpatient_list, timeinterval)
    demand_inp_new_PARA = PARA_newDemand_inp(theta_I_new, timeinterval)
    
    return [[demand_outp[tt][c]+demand_inp_known[tt][c]+demand_inp_new_PARA[tt][c] for c in range(C)] for tt in range(timeinterval)]

def TUNE_newInpatient_NEWS(outpatient_list, inpatient_list, patient_waiting_list, trj_inp, trj_outp, timeinterval, theta_I_new, alpha):
    return PARA_newDemand_inp_NEWS(patient_waiting_list, inpatient_list, outpatient_list, trj_outp, timeinterval, alpha)

def TUNE_newInpatient_NEWS_iter(outpatient_list, inpatient_list, patient_waiting_list, trj_inp, trj_outp, timeinterval, theta_I_new, alpha):
    return PARA_newDemand_inp_NEWS_iter(patient_waiting_list, inpatient_list, outpatient_list, trj_outp, timeinterval, alpha)

def TUNE_alles(outpatient_list, inpatient_list, patient_waiting_list, trj_inp, trj_outp, timeinterval, theta, alpha ):
    demand_inp_known = getDemandList_I(inpatient_list, timeinterval)
    demand_outp_known = getDemandList_O(outpatient_list, timeinterval)
    
    PARA_new = [np.multiply(theta[0],t) for t in range(timeinterval)]
    PARA_dec_I = PARA_demandDecision_I(inpatient_list, timeinterval, theta[1])
    PARA_dec_O = PARA_demandDecision_O(outpatient_list, timeinterval, theta[2])
    
    return [[demand_inp_known[tt][c]+demand_outp_known[tt][c]+PARA_new[tt][c]+PARA_dec_I[tt][c]+PARA_dec_O[tt][c] for c in range(C)] for tt in range(timeinterval)]

def getProbCombination_check(list1, list2):
    results = []
    xx = np.sum(list(it.product(list1[0], list2[0])), 1)
    yy = np.prod(list(it.product(list1[1], list2[1])), axis = 1)
    if len(xx)==1:
        pos_options_comb =[np.array(xx[0])]
        pos_options_comb_prob = [np.array(yy[0])]
        
    else: 
        pos_options_comb = []
        pos_options_comb_prob =[]
        for m in xx:
            if len(find_indices(pos_options_comb, m)) ==0:
                ind = find_indices(xx, m)
                ss = 0
                for ii in ind:
                    ss+= yy[ii]
                pos_options_comb.append(m)
                pos_options_comb_prob.append(ss)
    
    combined_lists = list(zip(pos_options_comb, pos_options_comb_prob))
    sorted_lists = sorted(combined_lists, key=lambda x: x[0])

    pos_options_comb = [item[0] for item in sorted_lists]
    pos_options_comb_prob = [item[1] for item in sorted_lists]

    return [pos_options_comb,pos_options_comb_prob ]

def TUNE_alles_NEWS(outpatient_list, inpatient_list, patient_waiting_list, inpatients_trajectory, outpatients_trajectory, timeinterval, theta, alpha):

    pos_opt_total_I, pos_opt_prob_totaal_I, pos_opt_prob_cdf_totaal_I = getPosOptions_I(inpatient_list, timeinterval)
    pos_opt_total_O, pos_opt_prob_totaal_O, pos_opt_prob_cdf_totaal_O = getPosOptions_O(outpatient_list, timeinterval)
    
    nr_pat = np.zeros((timeinterval), dtype = np.int16)
    for pat in inpatient_list:
        for t in range(timeinterval):
            if pat.L  > t:
                nr_pat[t]+=1
    current_demands = getDemandList_I(inpatient_list, timeinterval) + getDemandList_O(outpatient_list, timeinterval)
    thetas =[[ np.zeros((1)) for c in range(C)]]

    extra_pat = 0
    
    for t in range(0, timeinterval-1):
        theta_c = []
        extra_pat_c = 0
        for c in range(C):   
            results = getProbCombination_check([pos_opt_total_I[t][c], pos_opt_prob_totaal_I[t][c]], [pos_opt_total_O[t][c], pos_opt_prob_totaal_O[t][c]]))
            pos_arrivals = pos_arrivals_nrpat[nr_pat[t]+extra_pat][c]
            pos_arrivals_prob= pos_arrivals_prob_nrpat[nr_pat[t]+extra_pat][c]
            list_arrivals =[pos_arrivals, pos_arrivals_prob]

            ttr  = getProbCombination_check(results, list_arrivals)
            pos_options = ttr[0]
            pos_options_prob = ttr[1]
            pos_options_cdf  = [np.sum(pos_options_prob[:i+1]) for i in range(len(pos_options_prob))]
            total_urgency = 0 
            total_action_demand = np.zeros((C))
            for pat in patient_waiting_list:
                total_urgency += U_cost[pat.urgency]
                total_action_demand[:] += pat.fullTreatmentPlan[0]
            for tt in range(t):
                for pat in outpatients_trajectory[tt]:
                    total_urgency += U_cost[pat.urgency]
                    total_action_demand[:] += pat.fullTreatmentPlan[0]  

            current_demand = current_demands[t][c]+ np.sum(thetas, axis = 0)[c]

            if f[c]-current_demand-total_action_demand[c] >np.min(pos_options, axis=0) :
                if f[c]-current_demand-total_action_demand[c] < np.max(pos_options, axis=0):
                    pr = [pos_options_cdf[i] for i in range(len(pos_options)) if pos_options[i] <= f[c]-current_demand-total_action_demand[c]][-1]
                else: pr = 1
            else: pr  =0

            if total_action_demand[0] == 0 or ((total_urgency/total_action_demand[c])+C_cost[c]*(1-pr)) == 0:

                theta1 = C_cost[c]*(1-pr)/(C_cost[c]+C_cost[c]*(1-pr))
                theta2 = alpha*pr*C_cost[c]/(alpha*pr*C_cost[c]+alpha*pr*C_cost[c]+C_cost[c]*(1-pr)) 
            else: 
                theta1 = C_cost[c]*(1-pr)/((total_urgency/total_action_demand[c])+C_cost[c]*(1-pr))
                theta2 = alpha*pr*C_cost[c]/(alpha*pr*(total_urgency/total_action_demand[c])+alpha*pr*C_cost[c]+C_cost[c]*(1-pr))

            xp1 = [pos_options[i] for i in range(len(pos_options_cdf)) if pos_options_cdf[i]>=theta1]
            if xp1 == [0]:
                theta1_f = xp1[0]
            else: theta1_f = xp1[0]
            xp2 = [pos_options[i] for i in range(len(pos_options_cdf)) if pos_options_cdf[i]>=theta2]
            if  xp2 == [0]:
                theta2_f = xp2[0]
            else: 
                theta2_f = xp2[0]
                
            if current_demand+total_action_demand[c]+theta1_f < f[c]:
                theta_c.append(theta2_f)
                extra_pat_c += [nr_tot[nr_pat[t]+extra_pat][i] for i in range(len(prob_nr_cdf_tot[nr_pat[t]+extra_pat])) if prob_nr_cdf_tot[nr_pat[t]+extra_pat][i]>= theta2][0]
            
            else: 
                theta_c.append(theta1_f)
                extra_pat_c += [nr_tot[nr_pat[t]+extra_pat][i] for i in range(len(prob_nr_cdf_tot[nr_pat[t]+extra_pat])) if prob_nr_cdf_tot[nr_pat[t]+extra_pat][i]>= theta1][0]
        extra_pat += int(round(extra_pat_c)/C)
        thetas.append(theta_c)

    theta_sum = np.zeros((timeinterval, C))
    for i in range(timeinterval):
        for c in range(C):
            theta_sum[i][c] = np.sum(thetas[:i+1], axis = 0)[c]

    return current_demands + theta_sum

def TUNE_alles_NEWS_iter(outpatient_list, inpatient_list, patient_waiting_list, inpatients_trajectory, outpatients_trajectory, timeinterval, theta, N_iteration, alpha):
    pos_opt_total_I, pos_opt_prob_totaal_I, pos_opt_prob_cdf_totaal_I = getPosOptions_I(inpatient_list, timeinterval)
    pos_opt_total_O, pos_opt_prob_totaal_O, pos_opt_prob_cdf_totaal_O = getPosOptions_O(outpatient_list, timeinterval)

    nr_pat = np.zeros((timeinterval), dtype = np.int16)
    for pat in inpatient_list:
        for t in range(timeinterval):
            if pat.L  > t:
                nr_pat[t]+=1
        
    current_demands = getDemandList_I(inpatient_list, timeinterval) + getDemandList_O(outpatient_list, timeinterval)
    thetas =[[ np.zeros((1)) for c in range(C)]]
    extra_pat = 0
    
    for t in range(0, timeinterval-1):

        theta_c = []
        extra_pat_c = 0
        pr_c = []
        pos_opt_c = []
        pos_opt_prob_c = []
        for c in range(C):
    
            results = getProbCombination_check([pos_opt_total_I[t][c], pos_opt_prob_totaal_I[t][c]], [pos_opt_total_O[t][c], pos_opt_prob_totaal_O[t][c]])

            pos_arrivals = pos_arrivals_nrpat[nr_pat[t]+extra_pat][c]
            pos_arrivals_prob= pos_arrivals_prob_nrpat[nr_pat[t]+extra_pat][c]
            list_arrivals =[pos_arrivals, pos_arrivals_prob]


            ttr  = getProbCombination_check(results, list_arrivals)
            pos_options = ttr[0]
            pos_options_prob = ttr[1]
            pos_options_cdf  = [np.sum(pos_options_prob[:i+1]) for i in range(len(pos_options_prob))]
            pos_opt_c.append(pos_options)
            pos_opt_prob_c.append(pos_options_prob)
            # print(pos_options_prob)
            total_urgency = 0 
            total_action_demand = np.zeros((C))
            for pat in patient_waiting_list:
                total_urgency += U_cost[pat.urgency]
                total_action_demand[:] += pat.fullTreatmentPlan[0]
            for tt in range(t):
                for pat in outpatients_trajectory[tt]:
                    total_urgency += U_cost[pat.urgency]
                    total_action_demand[:] += pat.fullTreatmentPlan[0]  
            current_demand = current_demands[t][c]+ np.sum(thetas, axis = 0)[c]

            if f[c]-current_demand-total_action_demand[c] >np.min(pos_options, axis=0) :
                if f[c]-current_demand-total_action_demand[c] < np.max(pos_options, axis=0):
                    pr = [pos_options_cdf[i] for i in range(len(pos_options)) if pos_options[i] <= f[c]-current_demand-total_action_demand[c]][-1]
                else: pr = 1
            else: pr  =0
            pr_c.append(pr)
            if total_action_demand[0] == 0 or ((total_urgency/total_action_demand[c])+C_cost[c]*(1-pr)) == 0:

                theta1 = C_cost[c]*(1-pr)/(C_cost[c]+C_cost[c]*(1-pr))
                theta2 = alpha*pr*C_cost[c]/(alpha*pr*C_cost[c]+alpha*pr*C_cost[c]+C_cost[c]*(1-pr))
                
            else: 
                theta1 = C_cost[c]*(1-pr)/((total_urgency/total_action_demand[c])+C_cost[c]*(1-pr))
                theta2 = alpha*pr*C_cost[c]/(alpha*pr*(total_urgency/total_action_demand[c])+alpha*pr*C_cost[c]+C_cost[c]*(1-pr))

            xp1 = [pos_options[i] for i in range(len(pos_options_cdf)) if pos_options_cdf[i]>=theta1]
            if xp1 == [0]:
                
                theta1_f = xp1[0]
            else: theta1_f = xp1[0]
            xp2 = [pos_options[i] for i in range(len(pos_options_cdf)) if pos_options_cdf[i]>=theta2]
            if  xp2 == [0]:
                
                theta2_f = xp2[0]
            else: 
                theta2_f = xp2[0]
            if current_demand+total_action_demand[c]+theta1_f < f[c]:
                theta_c.append(theta2_f)
                extra_pat_c += [nr_tot[nr_pat[t]+extra_pat][i] for i in range(len(prob_nr_cdf_tot[nr_pat[t]+extra_pat])) if prob_nr_cdf_tot[nr_pat[t]+extra_pat][i]>= theta2][0]
            
            else: 
                theta_c.append(theta1_f)
                extra_pat_c += [nr_tot[nr_pat[t]+extra_pat][i] for i in range(len(prob_nr_cdf_tot[nr_pat[t]+extra_pat])) if prob_nr_cdf_tot[nr_pat[t]+extra_pat][i]>= theta1][0]
            
        extra_pat += int(round(extra_pat_c)/C)
        
        eta  = 1
        for i in range(N_iteration):
            stop = False
            count = 0
            while True:
                Z = np.random.normal(size = (C,1))
                thetas_random = [i for i in theta_c + eta*Z]  
                if all(thetas_random[c]<= max(pos_opt_c[c]) for c in range(C)) and all(thetas_random[c]>= 0 for c in range(C)):
                    break
                count += 1
                if count == 5:
                    stop = True
                    break
            if stop:
                break
        
            thetas_check = [theta_c, thetas_random]
            
            objs = []
            for th in thetas_check:
                obj = 0
                for c in range(C):
                    current_demand = current_demands[t][c]+ np.sum(thetas, axis = 0)[c]
                    #regel 1
                    if current_demand+total_action_demand[c]+th[c] > f[c]:                     
                        if total_action_demand[c] !=0:
                            obj += (total_urgency/total_action_demand[c])*(-f[c] + (current_demand+total_action_demand[c]+th[c])- getExpectationMin(pos_opt_prob_c[c], pos_opt_c[c], th[c])[0] + getExpectationMin(pos_opt_prob_c[c], pos_opt_c[c],f[c]- current_demand-total_action_demand[c]) )
                        else: 
                            obj += (C_cost[c])*(-f[c] + (current_demand+total_action_demand[c]+th[c])- getExpectationMin(pos_opt_prob_c[c], pos_opt_c[c], th[c])[0] + getExpectationMin(pos_opt_prob_c[c], pos_opt_c[c],f[c]- current_demand-total_action_demand[c]) )
                            
                    #regel 2
                    if current_demand+total_action_demand[c]+th[c] > f[c]: 
                        obj+= (1-pr_c[c])*C_cost[c]*(getExpectation(pos_opt_prob_c[c], pos_opt_c[c])[0]- getExpectationMin(pos_opt_prob_c[c], pos_opt_c[c], th[c])[0])              
                    else:
                        obj+= (1-pr_c[c])*C_cost[c]*(getExpectation(pos_opt_prob_c[c], pos_opt_c[c])[0]- getExpectationMin(pos_opt_prob_c[c], pos_opt_c[c], th[c])[0]+f[c] - (current_demand+total_action_demand[c]+th[c]))              

                    
                    #regel 3
                    if current_demand+total_action_demand[c]+th[c] <= f[c]: 
                        if total_action_demand[c] !=0:
                        
                            obj+= alpha*pr_c[c]*((total_urgency/total_action_demand[c])*th[c]+C_cost[c]*getExpectation(pos_opt_prob_c[c], pos_opt_c[c])[0]+(-1*(total_urgency/total_action_demand[c])-C_cost[c])*getExpectationMin(pos_opt_prob_c[c], pos_opt_c[c], th[c])[0])
                        else:
                            obj+= alpha*pr_c[c]*((C_cost[c])*th[c]+C_cost[c]*getExpectation(pos_opt_prob_c[c], pos_opt_c[c])[0]+(-2*C_cost[c])*getExpectationMin(pos_opt_prob_c[c], pos_opt_c[c], th[c])[0])
                            
                objs.append(obj)
            if objs[1]< objs[0]:
                theta_c = thetas_random
        thetas.append(theta_c)

    theta_sum = np.zeros((timeinterval, C))
    for i in range(timeinterval):
        for c in range(C):
            theta_sum[i][c] = np.sum(thetas[:i+1], axis = 0)[c]
    return current_demands + theta_sum

def PARA_newDemand_inp_withzeros(theta, timeinterval,inpatient_list):
    nr_pat = np.zeros((timeinterval), dtype = np.int16)
    for pat in inpatient_list:
        for t in range(timeinterval):
            if pat.L  > t:
                nr_pat[t]+=1
    
    dd = np.zeros((timeinterval,C))
    for t in range(timeinterval):
        if nr_pat[t]<max_nr_i:
            for c in range(C):
                dd[t][c] = np.sum(theta[c]*t, axis = 0)
    return dd

def TUNE_newInpatient_withzeros(outpatient_list, inpatient_list, patient_waiting_list, trj_inp, trj_outp, timeinterval, theta_I_new, alpha):
    demand_outp = getRealDemandOutpatient(outpatient_list, timeinterval)
    demand_inp_known = getDemandInpatient_withoutNew(inpatient_list, timeinterval)
    demand_inp_new_PARA = PARA_newDemand_inp_withzeros(theta_I_new, timeinterval, inpatient_list)
    return [[demand_outp[tt][c]+demand_inp_known[tt][c]+demand_inp_new_PARA[tt][c] for c in range(C)] for tt in range(timeinterval)]
    
def TUNE_alles_withzeros(outpatient_list, inpatient_list, patient_waiting_list, trj_inp, trj_outp, timeinterval, theta, alpha ):
    demand_inp_known = getDemandList_I(inpatient_list, timeinterval)
    demand_outp_known = getDemandList_O(outpatient_list, timeinterval)
    
    PARA_new = PARA_newDemand_inp_withzeros(theta[0], timeinterval, inpatient_list)
    PARA_dec_I = PARA_demandDecision_I(inpatient_list, timeinterval, theta[1])
    PARA_dec_O = PARA_demandDecision_O(outpatient_list, timeinterval, theta[2])
    
    return [[demand_inp_known[tt][c]+demand_outp_known[tt][c]+PARA_new[tt][c]+PARA_dec_I[tt][c]+PARA_dec_O[tt][c] for c in range(C)] for tt in range(timeinterval)]

def determineTotalCost_function_TUNE_recourse(patient_waiting_list, outpatient_list, inpatient_list, outpatients_trajectory, inpatients_trajectory, theta, timerange, timeinterval, actionCostFunction, N_samples, alpha = 1, totalCostOnly = False):
    totalCost = 0

    costArray = []
    demandArray = []
    demandArrayOutp = []
    waitingCosts = []
    waitingTime = [[] for i in range(U)]
    
    for t in range(timerange):
        opt_actions = []
        opt_actions_count = []
        
        for n in range(N_samples):
            outpatients_trajectory_test =makeOutPatients(timeinterval+1, 'real')
            
            chosen_action = determineAction(actionCostFunction(outpatient_list, inpatient_list, patient_waiting_list, inpatients_trajectory, outpatients_trajectory_test, timeinterval+1, theta, alpha ), patient_waiting_list, outpatients_trajectory_test, timeinterval)
            for pat in chosen_action:
                waitingTime[pat.urgencyInitial].append(pat.timeWaitingList)
            n = 0
            for n in range(len(opt_actions)):
                if all([pat in opt_actions[n] for pat in chosen_action]):
                    opt_actions_count[n]+=1
                    break
                else: n+=1
            if n ==len(opt_actions):
                opt_actions.append(chosen_action)
                opt_actions_count.append(1)
        chosen_action_def = opt_actions[np.argmax(opt_actions_count)]
        x = getCost(patient_waiting_list, outpatient_list, inpatient_list, chosen_action_def)

        totalCost += x
        post_decision_waiting_list = []
        for pat in patient_waiting_list:
            if not any([pat.name ==pat2.name for pat2 in chosen_action]):
                post_decision_waiting_list.append(pat)

        waitingCost = sum([U_cost[pat.urgency] for pat in post_decision_waiting_list])
        waitingCosts.append(waitingCost)
        costArray.append(x)
        demandArray.append(getDemandNow(outpatient_list, inpatient_list))
        demandArrayOutp.append(getDemandList_I(inpatient_list, timeinterval)[0])

        patient_waiting_list, outpatient_list, inpatient_list = updatePatientsLists(patient_waiting_list, outpatient_list, inpatient_list, outpatients_trajectory[0], inpatients_trajectory[0], chosen_action )

        del inpatients_trajectory[0] 
        del outpatients_trajectory[0] ')
        
    if totalCostOnly:
        return totalCost
    else:
        return totalCost, costArray, waitingCosts, demandArray, demandArrayOutp, waitingTime#, demandArrayNames

def TUNE_demandReal(outpatient_list, inpatient_list, patient_waiting_list, trj_inp, trj_outp, timeinterval, theta, alpha ):
    demand_inp = theta*getRealDemandInpatient(inpatient_list, trj_inp, timeinterval)
    demand_out = theta*getRealDemandOutpatient(outpatient_list, timeinterval)
    return [[demand_inp[tt][c] + demand_out[tt][c] for c in range(C)] for tt in range(timeinterval)]

# ________________________________________________________________________________________________________
# _________________________POLICY FUNCTIONS _____________________________________________________________
def defineU_cost(per_urgency_cost, max_plan):
    max_len = len(per_urgency_cost)
    # print(max_len)
    lll = []
    for u in range(max_len):
        count = u
        ll = [0]
        for t in range(0,max_plan):
            # print(count)
            ll.append(ll[t]+per_urgency_cost[count])
            if count != max_len-1:
                count+=1
                
        lll.append(ll)
    return lll

def determineAction(demand, patient_waiting_list, outpatients_trajectory, timeinterval, action = True):
    max_plan = timeinterval+1
    T = max_plan
    U_cost_  = defineU_cost(U_cost, max_plan)

    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0))
        env.start()
        with gp.Model(env=env) as m:
            vars_ww = [[m.addVar(vtype= GRB.BINARY, name = pat.name) for tt in range(T)] for pat in patient_waiting_list ]
            vars_pp = [[[m.addVar(vtype = GRB.BINARY,  name = pat.name) for tt in range(T)] for pat in outpatients_trajectory[ti]] for ti in range(timeinterval)]
            vars2 = [[m.addVar(vtype = GRB.BINARY, name = 'r') for c in range(C)] for t in range(T)]
            
            for pat in range(len(patient_waiting_list)):
                m.addConstr(gp.quicksum(vars_ww[pat]) ==1)
                
            for ti in range(timeinterval):
                for pat in range(len(outpatients_trajectory[ti])):
                    m.addConstr(gp.quicksum(vars_pp[ti][pat])==1)
                    for tt in range(ti+1):
                        m.addConstr(vars_pp[ti][pat][tt]==0)
            
            obj = [[demand[tt][c] - f[c] for  c in range(C)] for tt in range(0, T)] 
            obj2 = 0
            
            for pat in range(len(patient_waiting_list)):
                obj2 += np.dot(U_cost_[patient_waiting_list[pat].urgency][:max_plan],vars_ww[pat][:max_plan])
                for tt in range(T):
                        for i in range(len(patient_waiting_list[pat].totalDemand())):
                            if tt+i+1 < T:
                                for c in range(C):
                                    obj[tt+i+1][c] += vars_ww[pat][tt]*patient_waiting_list[pat].totalDemand()[i][c]
            
            for ti in range(timeinterval):
                for pat in range(len(outpatients_trajectory[ti])):
                    obj2 += np.dot(U_cost_[outpatients_trajectory[ti][pat].urgency][:len(vars_pp[ti][pat][ti+1:])],vars_pp[ti][pat][ti+1:])               
                    for tt in range(T):                   
                            for i in range(len(outpatients_trajectory[ti][pat].totalDemand())):
                                if tt+1+i < T:
                                    for c in range(C):
                                        obj[tt+1+i][c] += vars_pp[ti][pat][tt]*outpatients_trajectory[ti][pat].totalDemand()[i][c]
            
            M=  10000
            for r in range(T):
                for c in range(C):
                    m.addConstr(-M*(1-vars2[r][c])<= obj[r][c])
                    m.addConstr(M*vars2[r][c]>= obj[r][c])
            m.update()
            m.setObjective(gp.quicksum([C_cost[c]*vars2[r][c]*obj[r][c]for c in range(C) for r in range(T-1)])+obj2, GRB.MINIMIZE)
            m.update()
            
            m.optimize()     
            if action: 
                chosen = []
                for pat in range(len(patient_waiting_list)):
                    if vars_ww[pat][0].X ==1:
                        chosen.append(patient_waiting_list[pat])
                return chosen    
            else:  
                actions = []                 
                for t in range(timeinterval):
                    chosen = []
                    for pat in range(len(patient_waiting_list)):
                        if vars_ww[pat][t].X ==1:
                            chosen.append(patient_waiting_list[pat])
                            
                    for ti in range(timeinterval):
                        if ti<t:
                            for pat in range(len(outpatients_trajectory[ti])):
                                if vars_pp[ti][pat][t].X ==1:

                                    chosen.append(outpatients_trajectory[ti][pat])
                    actions.append(chosen)
                return actions

#______________________________________________________________________________________
#____________________TOTAL COST FUNCTIONS______________________________________________
def determineTotalCost_function_TUNE(patient_waiting_list, outpatient_list, inpatient_list, outpatients_trajectory, inpatients_trajectory, theta, timerange, timeinterval, actionCostFunction, alpha = 1, totalCostOnly = False):
    totalCost = 0
    costArray = []
    demandArray = []
    demandArrayOutp = []
    waitingCosts = []
    waitingTime = [[] for i in range(U)]
    for t in range(timerange):
        chosen_action = determineAction(actionCostFunction(outpatient_list, inpatient_list, patient_waiting_list, inpatients_trajectory, outpatients_trajectory, timeinterval+1, theta, alpha ), patient_waiting_list, outpatients_trajectory, timeinterval)
        for pat in chosen_action:
            waitingTime[pat.urgencyInitial].append(pat.timeWaitingList)
        x = getCost(patient_waiting_list, outpatient_list, inpatient_list, chosen_action)

        totalCost += x
        post_decision_waiting_list = []
        for pat in patient_waiting_list:
            if not any([pat.name ==pat2.name for pat2 in chosen_action]):
                post_decision_waiting_list.append(pat)

        waitingCost = sum([U_cost[pat.urgency] for pat in post_decision_waiting_list])
        waitingCosts.append(waitingCost)
        costArray.append(x)
        demandArray.append(getDemandNow(outpatient_list, inpatient_list))
        demandArrayOutp.append(getDemandList_I(inpatient_list, timeinterval)[0])

        patient_waiting_list, outpatient_list, inpatient_list = updatePatientsLists(patient_waiting_list, outpatient_list, inpatient_list, outpatients_trajectory[0], inpatients_trajectory[0], chosen_action )

        del inpatients_trajectory[0] 
        del outpatients_trajectory[0]     
    if totalCostOnly:
        return totalCost
    else:
        return totalCost, costArray, waitingCosts, demandArray, demandArrayOutp, waitingTime#, demandArrayNames
    
def determineTotalCost_function_TUNE_iter(patient_waiting_list, outpatient_list, inpatient_list, outpatients_trajectory, inpatients_trajectory, theta, timerange, timeinterval, actionCostFunction, N_iter, alpha = 1, totalCostOnly = False):
    totalCost = 0
    costArray = []
    demandArray = []
    demandArrayOutp = []
    waitingCosts = []
    waitingTime = [[] for i in range(U)]
    for t in range(timerange):
        chosen_action = determineAction(actionCostFunction(outpatient_list, inpatient_list, patient_waiting_list, inpatients_trajectory, outpatients_trajectory, timeinterval+1, theta, N_iter, alpha ), patient_waiting_list, outpatients_trajectory, timeinterval)
        if t > 20:
        
            for pat in chosen_action:
                waitingTime[pat.urgencyInitial].append(pat.timeWaitingList)
        x = getCost(patient_waiting_list, outpatient_list, inpatient_list, chosen_action)

        totalCost += x
        post_decision_waiting_list = []
        for pat in patient_waiting_list:
            if not any([pat.name ==pat2.name for pat2 in chosen_action]):
                post_decision_waiting_list.append(pat)

        waitingCost = sum([U_cost[pat.urgency] for pat in post_decision_waiting_list])
        waitingCosts.append(waitingCost)
        costArray.append(x)
        demandArray.append(getDemandNow(outpatient_list, inpatient_list))
        demandArrayOutp.append(getDemandList_I(inpatient_list, timeinterval)[0])

        patient_waiting_list, outpatient_list, inpatient_list = updatePatientsLists(patient_waiting_list, outpatient_list, inpatient_list, outpatients_trajectory[0], inpatients_trajectory[0], chosen_action )

        del inpatients_trajectory[0] 
        del outpatients_trajectory[0] 
        
    if totalCostOnly:
        return totalCost
    else:
        return totalCost, costArray, waitingCosts, demandArray, demandArrayOutp, waitingTime#, demandArrayNames

    
def determineTotalCost_function_CHEAPEST(patient_waiting_list, outpatient_list, inpatient_list, outpatients_trajectory, inpatients_trajectory, timerange, timeinterval, totalCostOnly = False):
    totalCost = 0
    costArray = []
    demandArray = []
    demandArrayOutp = []
    waitingCosts = []

    for t in range(timerange):
        chosen_action = patient_waiting_list
        x = getCost(patient_waiting_list, outpatient_list, inpatient_list, chosen_action)

        totalCost += x
        post_decision_waiting_list = []
        for pat in patient_waiting_list:
            if not any([pat.name ==pat2.name for pat2 in chosen_action]):
                post_decision_waiting_list.append(pat)

        waitingCost = sum([U_cost[pat.urgency] for pat in post_decision_waiting_list])
        waitingCosts.append(waitingCost)
        costArray.append(x)
        demandArray.append(getDemandNow(outpatient_list, inpatient_list))
        demandArrayOutp.append(getDemandList_I(inpatient_list, timeinterval)[0][0])

        patient_waiting_list, outpatient_list, inpatient_list = updatePatientsLists(patient_waiting_list, outpatient_list, inpatient_list, outpatients_trajectory[0], inpatients_trajectory[0], chosen_action )
        del inpatients_trajectory[0] 
        del outpatients_trajectory[0] 
        
    if totalCostOnly:
        return totalCost
    else:
        return totalCost, costArray, waitingCosts, demandArray, demandArrayOutp#, demandArrayNames
    
def determineTotalCost_actionsList(patient_waiting_list, outpatient_list, inpatient_list, outpatients_trajectory, inpatients_trajectory, timerange, action_list, totalCostOnly = False):
    totalCost = 0    
    costArray = []
    demandArray = []
    waitingCosts = []
    demandArrayNames =[]
    for t in range(timerange):
        post_decision_waiting_list = []
        for pat in patient_waiting_list:
            if not any([pat.name ==pat2.name for pat2 in action_list[t]]):
                post_decision_waiting_list.append(pat)
        x = getCost(patient_waiting_list, outpatient_list, inpatient_list, action_list[t])
        totalCost += x
        
        waitingCost = sum([U_cost[pat.urgency] for pat in post_decision_waiting_list])
        waitingCosts.append(waitingCost)
        costArray.append(x)
        demandArray.append(getDemandNow(outpatient_list, inpatient_list))
        currentNames = [t]

        patient_waiting_list, outpatient_list, inpatient_list = updatePatientsLists(patient_waiting_list, outpatient_list, inpatient_list, outpatients_trajectory[0], inpatients_trajectory[0], action_list[t] )

        del inpatients_trajectory[0] 
        del outpatients_trajectory[0] 

    if totalCostOnly:
        return totalCost
    else:
        return totalCost, costArray, waitingCosts, sum(waitingCosts), demandArray#, demandArrayNames
    
def makeDeepCopyAll(patient_waiting_list_init, outpatient_list_init, inpatient_list_init, inpatients_trajectory_init, outpatients_trajectory_init):
    patient_waiting_list  = copy.deepcopy(patient_waiting_list_init)
    outpatient_list = copy.deepcopy(outpatient_list_init)
    inpatient_list = copy.deepcopy(inpatient_list_init)

    outpatients_trajectory = copy.deepcopy(outpatients_trajectory_init)
    inpatients_trajectory = copy.deepcopy(inpatients_trajectory_init)
    
    return patient_waiting_list, outpatient_list, inpatient_list, inpatients_trajectory, outpatients_trajectory

#__________________________________________________________________________________________________
## ______________________________ALGORITHMS__________________________________________________________
def ADAM_setup(timeinterval, timerange, actionCostFunction, theta0, size_all, eta, N_iteration, repeats, lr):
    theta = copy.copy(theta0)
    vv = np.zeros(size_all)
    vhat = np.zeros(size_all)
    ss = np.zeros(size_all)
    shat = np.zeros(size_all)
    
    beta1 = 0.9
    beta2 = 0.99
    # lr = 0.01
    
    k_count = 1
    G = 0
    gg = np.zeros(size_all)
    
    current_theta = []
    current_thetahat = []
    theta_hat = np.zeros(size_all)
    theta_bar = 0
    print('start')
    while True:
        # save theta
        theta = theta - gg
        theta_bar = beta2*theta_bar +(1-beta2)*theta
        theta_hat = theta_bar/(1-beta2**k_count)
        current_theta.append(theta)
        current_thetahat.append(theta_hat)
        
        # generate random theta
        Z = np.random.normal(size = size_all)
        theta_random = theta + eta*Z
        theta_check = [theta, theta_random]
        
        costs_m = []
        for mm in range(repeats):
            patient_waiting_list_init, outpatient_list_init, inpatient_list_init = makeInitialization_random(T=10)
            inpatients_trajectory_init = makeInPatients(timerange+timeinterval+1, 'real')
            outpatients_trajectory_init =makeOutPatients(timerange+timeinterval+1, 'real')

            costs = []
            
            for th in theta_check:
                patient_waiting_list, outpatient_list, inpatient_list, inpatients_trajectory, outpatients_trajectory = makeDeepCopyAll(patient_waiting_list_init, outpatient_list_init, inpatient_list_init, inpatients_trajectory_init, outpatients_trajectory_init)
                costs.append(determineTotalCost_function_TUNE(patient_waiting_list, outpatient_list, inpatient_list, outpatients_trajectory, inpatients_trajectory, th, timerange, timeinterval, actionCostFunction, totalCostOnly = True))         
            costs_m.append(costs)
        
        G = (np.average(costs_m, axis = 0)[1]-np.average(costs_m, axis = 0)[0])*Z/eta
        # if G != 0:
        vv = beta1*vv + (1-beta1)*G
        ss = beta2*ss + (1-beta2)*G**2
        vhat = vv/(1-beta1**k_count)
        shat = ss/(1-beta2**k_count)
        gg = lr*vhat/np.sqrt(shat+10**(-8))
        k_count += 1
        if k_count%(N_iteration/100)==0:
            print(str(k_count/(N_iteration)*100)+'%')
            np.savetxt('ADAM_out/ADAM_'+str(eta)+'_'+str(theta0)+'pending.out', current_theta, delimiter=',') 
            
        if k_count == N_iteration:
            break
        
    return current_theta, current_thetahat

def ADAM_setup_print(timeinterval, timerange, actionCostFunction, theta0, size_all, eta, N_iteration, repeats, lr):
    theta = copy.copy(theta0)
    vv = np.zeros(size_all)
    vhat = np.zeros(size_all)
    ss = np.zeros(size_all)
    shat = np.zeros(size_all)
    
    beta1 = 0.9
    beta2 = 0.99
    # lr = 0.01
    
    k_count = 1
    G = 0
    gg = np.zeros(size_all)
    
    current_theta = []
    current_thetahat = []
    theta_hat = np.zeros(size_all)
    theta_bar = 0
    print('start')
    while True:
        # save theta
        theta = theta - gg
        theta_bar = beta2*theta_bar +(1-beta2)*theta
        theta_hat = theta_bar/(1-beta2**k_count)
        current_theta.append(theta)
        current_thetahat.append(theta_hat)
        
        # generate random theta
        Z = np.random.normal(size = size_all)
        theta_random = theta + eta*Z
        theta_check = [theta, theta_random]
        print(theta_check)
        costs_m = []
        for mm in range(repeats):
            patient_waiting_list_init, outpatient_list_init, inpatient_list_init = makeInitialization_zeros()
            inpatients_trajectory_init = makeInPatients(timerange+timeinterval+1, 'real')
            outpatients_trajectory_init =makeOutPatients(timerange+timeinterval+1, 'real')

            costs = []
            
            for th in theta_check:
                patient_waiting_list, outpatient_list, inpatient_list, inpatients_trajectory, outpatients_trajectory = makeDeepCopyAll(patient_waiting_list_init, outpatient_list_init, inpatient_list_init, inpatients_trajectory_init, outpatients_trajectory_init)
                costs.append(determineTotalCost_function_TUNE(patient_waiting_list, outpatient_list, inpatient_list, outpatients_trajectory, inpatients_trajectory, th, timerange, timeinterval, actionCostFunction, totalCostOnly = True))         
            costs_m.append(costs)
        print(costs_m)
        G = (np.average(costs_m, axis = 0)[1]-np.average(costs_m, axis = 0)[0])*Z/eta
        # if G != 0:
        vv = beta1*vv + (1-beta1)*G
        ss = beta2*ss + (1-beta2)*G**2
        vhat = vv/(1-beta1**k_count)
        shat = ss/(1-beta2**k_count)
        gg = lr*vhat/np.sqrt(shat+10**(-8))
        k_count += 1
        if k_count%(N_iteration/100)==0:
            print(str(k_count/(N_iteration)*100)+'%')
        if k_count == N_iteration:
            break
        
    return current_theta, current_thetahat    
