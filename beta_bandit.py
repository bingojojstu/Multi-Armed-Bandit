from numpy import *
from scipy.stats import beta


class BetaBandit(object):
    def __init__(self, num_options=4, prior=(1.0,1.0,1.0,1.0)):
        self.trials = zeros(shape=(num_options,), dtype=int)
        self.successes = zeros(shape=(num_options,), dtype=int)
        self.num_options = num_options
        self.prior = prior

    def add_result(self, trial_id, success):
        self.trials[trial_id] = self.trials[trial_id] + 1
        if (success):
            self.successes[trial_id] = self.successes[trial_id] + 1

    def get_recommendation(self,trials1,successes1):
        sampled_theta = []
        for i in range(self.num_options):
            #Construct beta distribution for posterior
            dist = beta(self.prior[0]+successes1[i],
                        self.prior[0]+trials1[i]-successes1[i])
            #Draw sample from beta distribution
            sampled_theta += [ dist.rvs() ]
        # Return the index of the sample with the largest value
        return sampled_theta.index( max(sampled_theta) )