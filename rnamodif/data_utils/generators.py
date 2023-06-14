import torch
import numpy as np

def alternating_gen(generators):
    while True:
        for gen in generators:
            yield(next(gen))

def uniform_gen(generators):
    while True:
        gen = generators[torch.randint(high=len(generators), size=(1,))]
        yield next(gen)
        
def sequential_gen(generators):
    for gen in generators:
        for el in gen:
            yield el
    
def ratio_gen(generators, ratios):
    # Make sure the ratios add up to 1
    assert np.isclose(sum(ratios), 1), "Ratios must sum to 1"

    cumulative_ratios = np.cumsum(ratios) # create an array of cumulative sums of ratios
    while True: # keep generating indefinitely
        rand_num = torch.rand(size=(1,)) # generate a random number between 0 and 1

        # find the index of the first cumulative ratio that is greater than the random number
        for i, ratio in enumerate(cumulative_ratios):
            if rand_num < ratio:
                yield next(generators[i])
                break