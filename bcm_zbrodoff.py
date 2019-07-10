 #tryout BCM

#matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import FileLink 

import nengo
import nengo.spa as spa
#import bcm2_rule
import importlib #python3
import sys
import numpy
np.set_printoptions(threshold=sys.maxsize)

from ocl_sim import MyOCLsimulator
import os

import spa_mem_voja2_pes_hop_twolayers
import importlib
importlib.reload(spa_mem_voja2_pes_hop_twolayers)


if sys.platform == 'darwin':
    os.environ["PYOPENCL_CTX"] = "0:1"
else:
    os.environ["PYOPENCL_CTX"] = "0:0"
#importlib.reload(bcm2_rule)


#Semantic Pointer Dimensions 
D = 224 #512 #128
Dlow = 128

rng_vocabs = np.random.RandomState(seed=1)
vocab_problems = spa.Vocabulary(D,rng=rng_vocabs)


#determine the vocabulary of the model
letters = ['A', 'B', 'C', 'D']#, 'E', 'F', 'G', 'H', 'I', 'J', 'K']
numbers = ['ZERO', 'ONE', 'TWO']#, 'THREE', 'FOUR', 'FIVE']

for letter in letters:
    vocab_problems.parse(letter)
for number in numbers:
    vocab_problems.parse(number)

vocab_problems.parse('ITEM1')
vocab_problems.parse('ITEM2')
vocab_problems.parse('RESULT')

for letter in letters:
    vocab_problems.parse('ITEM1*%s' % (letter))
    vocab_problems.parse('RESULT*%s' % (letter))
for number in numbers: 
    vocab_problems.parse('ITEM2*%s' % (number))

    
#reset vocab
vocab_reset = spa.Vocabulary(Dlow,rng=rng_vocabs)
vocab_reset.parse('CLEAR + GO')



state = "recall"
#state = "training"

print(state)

    
def input_func(t):
    if t > 0 and t <= 1:
        if state is "recall":
             return 'ITEM1*A+ITEM2*ONE'
        else:
            return 'ITEM1*A+ITEM2*ONE+RESULT*B'
    elif t > .5 and t < .7:
        return '0'
    elif t > .7 and t < 1.5:
        if state is "recall":
                return 'ITEM1*A+ITEM2*TWO'
        else:
            return 'ITEM1*B+ITEM2*ONE+RESULT*C'
    elif t > 1.2 and t < 1.4:
        return '0'
    elif t > 1.4 and t < 2.0:
        return 'B+ONE'
    elif t > 2 and t < 2.2:
        return '0'
    else: 
        return 'ONE+TWO'
 
        
def inhib_func(t):
    if t > .5 and t <= 1:
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
    model.in_layer = spa.State(D,vocab=vocab_problems,seed=1)
    model.correct = spa.State(D, vocab=vocab_problems,seed=1)
    model.stoplearn = nengo.Node(0,size_out=1)

    model.stim = spa.Input(in_layer=input_func,correct=input_func)         
 
    n_neurons_mem = 1000 #8000 #1000
    
                    
                                     
    model.mem = spa_mem_voja2_pes_hop_twolayers.SPA_Mem_Voja2_Pes_Hop_TwoLayers( 
                                            input_vocab=vocab_problems,
                                            n_neurons=n_neurons_mem,
                                            n_neurons_out=50,
                                            dimensions=D,
                                            intercepts_mem=nengo.dists.Uniform(.1,.1), #=default Uniform(-1.0, 1.0)
                                            intercepts_out=nengo.dists.Uniform(-1,1), #.1,.1), #=default Uniform(-1.0, 1.0)
                                            #voja2_rate=1.5e-5, #1.5e-5,
                                            voja2_bias=1,   #1
                                            #pes_rate= .1, #.1
                                            bcm_rate=1e-3,  #1e-3
                                            bcm_max_weight=1e-8, #1e-8,  
                                            seed=1,
                                            load_from = "test"
                                            ) 
                                       
    model.mem.mem.create_weight_probes() #save weights
    nengo.Connection(model.in_layer.output, model.mem.input, synapse=None)
    nengo.Connection(model.correct.output, model.mem.correct, synapse=None)
    #nengo.Connection(model.stoplearn, model.mem.mem.stop_pes)

    #switch to accumulate
    #model.learn_state = spa.State(Dlow,vocab=vocab_reset)
    #model.do_learn = spa.AssociativeMemory(vocab_reset, default_output_key='CLEAR', threshold=.1,wta_output=True)
    #nengo.Connection(model.learn_state.output,model.do_learn.input)
    #nengo.Connection(model.do_learn.am.ensembles[-1], model.mem.mem.stop_pes, transform=-1, synapse=None)

# run in python + save
nengo_gui_on = __name__ == 'builtins' #python3

if not(nengo_gui_on):
    os.environ["PYOPENCL_CTX"] = "0:1"    #cur_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script path

    #sim = nengo.Simulator(model,seed=1)
    sim = MyOCLsimulator(model,seed=1)
    
    sim.run(1)
    
    model.mem.mem.save('test.npz',sim)


    

          

