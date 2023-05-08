import torch

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
    
