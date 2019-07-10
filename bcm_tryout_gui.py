 #tryout BCM

#matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import FileLink 
import nengo
import nengo.spa as spa
import bcm2_rule
import importlib #python3
import sys
import numpy
np.set_printoptions(threshold=sys.maxsize)

from ocl_sim import MyOCLsimulator
import os

import spa_mem_voja2_pes_hop_twolayers
#import compare_acc
import importlib
importlib.reload(spa_mem_voja2_pes_hop_twolayers)


if sys.platform == 'darwin':
    os.environ["PYOPENCL_CTX"] = "0:1"
else:
    os.environ["PYOPENCL_CTX"] = "0:0"
importlib.reload(bcm2_rule)
 
D = 128 #512 #128

rng_vocabs = np.random.RandomState(seed=1)
vocab = spa.Vocabulary(D,rng=rng_vocabs)
vocab.parse('CAT')
vocab.parse('DOG')
vocab.parse('CHAIR')
vocab.parse('TABLE')


def input_func(t):
    if t > 0 and t < .5:
        return 'CAT+DOG' 
    elif t > .5 and t < .7:
        return '0'
    elif t > .7 and t < 1.2:
        return 'CHAIR+TABLE'
    elif t > 1.2 and t < 1.4:
        return '0'
    elif t > 1.4 and t < 2.0:
        return 'CAT'
    elif t > 2 and t < 2.2:
        return '0'
    else:
        return 'CAT+TABLE'
        
        
        
def inhib_func(t):
    if t > .5 and t < .7:
        return -1
    elif t > 1.2 and t < 1.4:
        return -1
    elif t > 2 and t < 2.2:
        return -1
    else:
        return 0
        
        
      
model = spa.SPA(seed=1)
with model:

    #input layer 
    model.in_layer = spa.State(D,vocab=vocab,seed=1)
    model.correct = spa.State(D, vocab=vocab,seed=1)
    
    model.stim = spa.Input(in_layer=input_func,correct=input_func)         
 
    n_neurons_mem = 1000 #8000 #1000
    
                    
                                     
    model.mem = spa_mem_voja2_pes_hop_twolayers.SPA_Mem_Voja2_Pes_Hop_TwoLayers( 
                                            input_vocab=vocab,
                                            n_neurons=n_neurons_mem,
                                            n_neurons_out=50,
                                            dimensions=D,
                                            intercepts_mem=nengo.dists.Uniform(.1,.1), #=default Uniform(-1.0, 1.0)
                                            intercepts_out=nengo.dists.Uniform(-1,1), #.1,.1), #=default Uniform(-1.0, 1.0)
                                            voja2_rate=1e-1, #1.5e-2,
                                            voja2_bias=.5,
                                            pes_rate=.015,
                                            #bcm_rate=1e-9,  
                                            #bcm_max_weight=8e-4,  
                                            seed=1,
                                            ) 
                                           
    model.mem.mem.create_weight_probes() #save weights
    nengo.Connection(model.in_layer.output, model.mem.input, synapse=None)
    nengo.Connection(model.correct.output, model.mem.correct, synapse=None)



# run in python + save
nengo_gui_on = __name__ == 'builtins' #python3

if not(nengo_gui_on):

    #cur_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script path

    #sim = nengo.Simulator(model,seed=1)
    sim = MyOCLsimulator(model,seed=1)
    
    sim.run(1)
    
    model.mem.mem.save('test.npz',sim)


        

          

