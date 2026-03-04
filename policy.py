import numpy as np

def random_policy_factory(prob):
    def random_policy(state):
        return np.random.binomial(n=1, p=prob)

    return random_policy