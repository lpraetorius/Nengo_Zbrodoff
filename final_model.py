###### Import necessary packages & General Settings ######

import nengo
import nengo.spa as spa
import numpy as np
import assoc_mem_acc #associative memory with and status indicator that shows if something is retrieved
import compare_acc_zbrodoff #compare object that can compare two semantic pointers, and collects evidence in comparison_status
import datetime
import nengo_ocl
import pyopencl 
import os
import spa_mem_voja2_pes_hop_twolayers
from ocl_sim import MyOCLsimulator

os.environ["PYOPENCL_CTX"] = "0:0"
ctx = pyopencl.create_some_context()
print('\nSettings:')

import sys
print(sys.version_info)

fixed_seed = True #True #False
print('\nSettings:')

if fixed_seed:
    fseed = datetime.datetime.now()
    fseed = fseed.strftime("%Y%m%d")
    fseed = int(fseed)
    fseed = fseed + 0 #in case we want to change it
    fseed = 1 #works with fseed 1
    np.random.seed(fseed)
    #random.seed(fseed)
    print('\tFixed seed: %i' % fseed)
else:
    fseed = None
    print('\tRandom seed')
    
# Set dimensions of semantic pointers. Higher dimensions will make the model more reliable at
# the cost of more neurons. Must be divisible by 16
D = 272 #304 #272 #368  
Dlow = 112 #for goal and motor 
print('\tDimensions: %i, %i' % (D,Dlow))



####### Vocabularies #######

rng_vocabs = np.random.RandomState(seed=fseed)

letters = ['A', 'B', 'C', 'D']#, 'E', 'F', 'G', 'H', 'I', 'J', 'K']
numbers = ['ZERO', 'ONE', 'TWO']#, 'THREE', 'FOUR', 'FIVE']

vocab_concepts = spa.Vocabulary(D,rng=rng_vocabs)
for letter in letters:
    vocab_concepts.parse(letter)
for number in numbers:
    vocab_concepts.parse(number)

# The model also needs to know some slot names:
vocab_concepts.parse('ITEM1')
vocab_concepts.parse('ITEM2')
vocab_concepts.parse('RESULT')
vocab_concepts.parse('IDENTITY')
vocab_concepts.parse('NEXT')


# 2. Vocab with stored (correct) problem/answer combinations

# First, the concepts vocabulary is copied, to make sure that the same semantic pointers
# are used. Otherwise, this vocabulary would use different vectors for the same concepts.
vocab_problems = vocab_concepts.create_subset(vocab_concepts.keys) #copy the whole concepts vocab
vocab_problems.readonly = False #make it writable

#start with general knowledge (to use same vectors)
vocab_numbers = vocab_concepts.create_subset(numbers + ['IDENTITY', 'NEXT'])
vocab_numbers.readonly = False

cnt_number=0    
for number in numbers[0:(len(numbers)-1)]:
    cnt_number += 1
    vocab_numbers.add('%s->%s' % (number, numbers[cnt_number]),
        vocab_numbers.parse('IDENTITY*%s + NEXT*%s' % (number, numbers[cnt_number])))

#start with general knowledge (to use same vectors)
vocab_letters = vocab_concepts.create_subset(letters + ['IDENTITY', 'NEXT'])
vocab_letters.readonly = False

cnt_letter=0
for letter in letters[0:(len(letters)-1)]:
    cnt_letter += 1
    vocab_letters.add('%s->%s' % (letter, letters[cnt_letter]),
        vocab_letters.parse('IDENTITY*%s + NEXT*%s' % (letter, letters[cnt_letter])))


for letter in letters:
    vocab_problems.parse('ITEM1*%s' % (letter))
    vocab_problems.parse('RESULT*%s' % (letter))
for number in numbers: 
    vocab_problems.parse('ITEM2*%s' % (number))


#shortcut to write certain problems in l. 123
a1b = ['A', 'ONE', 'B']
a2b = ['A', 'TWO', 'B']
a1d = ['A', 'ONE', 'D']
a2d = ['A', 'TWO', 'D']
a1f = ['A', 'ONE', 'F']
a1c = ['A', 'ONE', 'C']
a2c = ['A', 'TWO', 'C']
a3d = ['A', 'THREE', 'D']
a4e = ['A', 'FOUR', 'E']
b1c = ['B', 'ONE', 'C']
b2d = ['B', 'TWO', 'D']
b1d = ['B', 'ONE', 'D']
b3e = ['B', 'THREE', 'E']
b4f = ['B', 'FOUR', 'F']
b2c = ['B', 'TWO', 'C']
c1d = ['C', 'ONE', 'D']
a1a = ['A', 'ONE', 'A']


#problem is presented here. the problem B+2=D must be given as ['B', 'TWO', 'D']
problem = a1b
#problem = ['A', 'ZERO', 'B']


# 3. Vocab with goal states, comparable to how we control an ACT-R model.
# This vocab uses a lower dimensionality, as it only has to store 5 different pointers.
vocab_goal = spa.Vocabulary(Dlow,rng=rng_vocabs)
vocab_goal.parse('START+RETRIEVE+RECALL+COMPARE+RESPOND+DONE+START_COUNTING+UPDATE_RESULT+STORE_RESULT+UPDATE_COUNT+STORE_COUNT+COMPARE_COUNT')

#vocab reset
vocab_reset = spa.Vocabulary(Dlow,rng=rng_vocabs)
vocab_reset.parse('CLEAR+GO')

# 4. Vocab_motor with two motor states.
vocab_motor = spa.Vocabulary(Dlow,rng=rng_vocabs)
vocab_motor.parse('YES')
vocab_motor.parse('NO')           


################## Set the problem that is presented ##################

# goal and imaginal input - this should vary trial by trial, probably want to use nodes for this or so, not just time

def vision_input_func(t):
    return'1*(ITEM1*%s + ITEM2*%s + RESULT*%s)' % (problem[0], problem[1], problem[2])

def goal_input_func(t):
    if t < .05:
        return 'START'
    else:
        return '0'


#The following three functions are used to present the problem to the model:
def arg1_func(t): #arg1 in imaginal
    return problem[0]
    
def arg2_func(t): #arg2 in imaginal
    return problem[1]

def target_func(t): #target in goal    
    return problem[2]

#This functions sets the goal state to start for the first 50 ms (cf. goal-focus in ACT-R)
def goal_func(t):
    if t < .05:
        return 'START'
    else:
        return '0'

#def input_func(t):
    #return'1*(ITEM1*%s + ITEM2*%s + RESULT*%s)' % (problem[0], problem[1], model.answer)
 #   return'1*(ITEM1*%s + ITEM2*%s + RESULT*%s)' % (problem[0], problem[1], problem[2])


################## Model ################## 

model = spa.SPA(seed=fseed)

with model:

    #Vision & Goal
    model.vision = spa.State(D,vocab=vocab_concepts) #visual input
   
    #Goal: a network with two slots: goal and target. 
    model.goalnet = nengo.Network(seed=fseed)
    with model.goalnet:
        model.goal = spa.State(Dlow,vocab=vocab_goal,feedback=1) #goal state    /for count feedback=.5
        model.count = spa.State(D,vocab=vocab_concepts,feedback=.8,feedback_synapse=.05) #the number of counts taken
        model.target = spa.State(D,vocab=vocab_concepts,feedback=0) #target answer from input   /for count feedback=1

    #Imaginal: a network with three slots: arg1, arg2, and answer.
    model.imaginal = nengo.Network(seed=fseed)
    with model.imaginal:
        model.arg1 = spa.State(D, vocab=vocab_concepts,feedback=0) #argument 1 from input   /for count feedback=1
        model.arg2 = spa.State(D, vocab=vocab_concepts,feedback=0) #argument 2 from input   /for count feedback=1
        model.answer = spa.State(D, vocab=vocab_concepts,feedback=1,feedback_synapse=.05) # /for count feedback=.8
        #model.answer = spa.State(D, vocab=vocab_concepts,feedback=1,feedback_synapse=.05) # /for count feedback=.8
    
            
    #set the inputs to the model (bypassing the need for a visual system)
    #model.input = spa.Input(goal=goal_func, target=target_func, arg1=arg1_func, arg2=arg2_func)
    #from count model
    model.input = spa.Input(vision=vision_input_func,goal=goal_input_func)

    #Number memory
    model.number_memory = assoc_mem_acc.AssociativeMemoryAccumulator(input_vocab = vocab_numbers, wta_output=True, status_scale=.7,threshold=.1,status_feedback=.3)

    #Alphabet memory
    model.letter_memory = assoc_mem_acc.AssociativeMemoryAccumulator(input_vocab = vocab_letters, wta_output=True, status_scale=.7,threshold=.2,status_feedback=.3)
        
    #Comparison
    model.comparison = compare_acc_zbrodoff.CompareAccumulator(vocab_compare = vocab_concepts,status_scale = .4,status_feedback = .2, status_feedback_synapse=.05, threshold_cleanup=.1)

    #final Comparison
    model.comparison2 = compare_acc_zbrodoff.CompareAccumulator(vocab_compare = vocab_concepts,status_scale = .6,status_feedback = .2, status_feedback_synapse=.05, threshold_cleanup=.1)

    #Motor
    model.motor = spa.State(Dlow,vocab=vocab_motor,feedback=1) #motor state       
    
  
    #bcm_model
    model.in_layer = spa.State(D,vocab=vocab_problems,seed=1)
    model.correct = spa.State(D, vocab=vocab_problems,seed=1)
    model.stoplearn = nengo.Node(0,size_out=1)

    #model.stim = spa.Input(in_layer=input_func,correct=input_func)         
 
    n_neurons_mem = 1000 #8000 #1000
    
    model.mem = spa_mem_voja2_pes_hop_twolayers.SPA_Mem_Voja2_Pes_Hop_TwoLayers( 
                                            input_vocab=vocab_problems,
                                            n_neurons=n_neurons_mem,
                                            n_neurons_out=50,
                                            dimensions=D,
                                            intercepts_mem=nengo.dists.Uniform(.1,.1), #=default Uniform(-1.0, 1.0)
                                            intercepts_out=nengo.dists.Uniform(-1,1), #.1,.1), #=default Uniform(-1.0, 1.0)
                                            voja2_rate=1.5e-5, #1.5e-5,
                                            voja2_bias=1,   #1
                                            pes_rate= .1, #.1
                                            bcm_rate=1e-3,  #1e-3
                                            bcm_max_weight=1e-8, #1e-8,  
                                            seed=1,
                                            #load_from = "test",
                                            ) 

    model.mem.mem.create_weight_probes() #save weights
    nengo.Connection(model.in_layer.output, model.mem.input, synapse=None)
    nengo.Connection(model.correct.output, model.mem.correct, synapse=None)
    nengo.Connection(model.stoplearn, model.mem.mem.stop_pes)

    #feedback for mem inputs
    nengo.Connection(model.correct.output, model.correct.input, transform=1, synapse=.1)#
    nengo.Connection(model.in_layer.output, model.in_layer.input, transform=1, synapse=.1)#


    #switch to accumulate
    model.learn_state = spa.State(Dlow,vocab=vocab_reset)
    model.do_learn = spa.AssociativeMemory(vocab_reset, threshold=.1,default_output_key='CLEAR', wta_output=True) #removed default_output_key
    #nengo.Connection(model.do_learn.output, model.do_learn.input)    
    nengo.Connection(model.learn_state.output,model.do_learn.input)
    nengo.Connection(model.do_learn.am.ensembles[-1], model.mem.mem.stop_pes, transform=-1, synapse=None)
    
    
### declarative retrieval from comparison input and output---------------

    d_comp = D
    model.dm_compare = spa.Compare(d_comp,neurons_per_multiply=150,input_magnitude=.8) #d_comp.neuronsmay be reduced, e.g 150
    direct_compare = True #keep that at true for now
    if direct_compare:
        for ens in model.dm_compare.all_ensembles:
            ens.neuron_type = nengo.Direct()

    nengo.Connection(model.mem.input[0:d_comp],model.dm_compare.inputA)
    nengo.Connection(model.mem.output[0:d_comp],model.dm_compare.inputB)

    #output ensemble of comparison
    model.dm_output_ens = nengo.Ensemble(n_neurons=1000,dimensions=1)
    nengo.Connection(model.dm_compare.output,model.dm_output_ens, synapse=None,transform=D/d_comp)

    #feedback (maybe not necessary, might make it more stable --> maybe remove)
    nengo.Connection(model.dm_output_ens, model.dm_output_ens, transform=.7, synapse=.1) #.8, .05

    threshold2 = 0 #.05 #-.1 # find accurate threshold
    def dec_func2(x):
        return x - threshold2


    #rep status accumulator
    model.dm_status = spa.State(1,feedback=1,neurons_per_dimension=1000,feedback_synapse=.01) #fb syn influences speed of acc
    nengo.Connection(model.dm_output_ens, model.dm_status.input,function=dec_func2,transform = 1) #set transformt o around 1

    #switch to accumulate #same as in mem_voja model
    model.do_dm = spa.AssociativeMemory(vocab_reset, threshold=.1, default_output_key='CLEAR', wta_output=True) #removed default_output_key
    nengo.Connection(model.do_dm.am.ensembles[-1], model.dm_status.all_ensembles[0].neurons, 
        transform=np.ones((model.dm_status.all_ensembles[0].n_neurons, 1)) * -3, synapse=0.02)

###--------------------------------------------------------
    
    #Basal Ganglia & Thalamus
    input_modifier = 6
    store_weight = 12 #12
    actions = spa.Actions(
        
        #encode the visual input
        a_encode = 'dot(goal,START) --> goal=RETRIEVE-.8*START, arg1=%g*~ITEM1*vision, arg2=%g*~ITEM2*vision, target=%g*~RESULT*vision' % (input_modifier, input_modifier, input_modifier),
        
        #compare visual input to Memory and retrieve answer
        b1_retrieve = 'dot(goal,RETRIEVE) --> goal=RECALL, do_dm=GO,  mem=5*ITEM1*arg1 + 5*ITEM2*arg2',
        b2_recall =   'dot(goal,RECALL) + dm_status -.3 --> goal=(.5*RECALL)+COMPARE, do_dm=GO, answer=8*~RESULT*mem',

        #ELSE start counting
        c_no_retrieval = 'dot(goal,RECALL) - dm_status -.2 --> goal=1.5*START_COUNTING, do_dm=CLEAR',# do_learn=2*GO,

        d_start_counting = 'dot(goal,START_COUNTING) - letter_memory_status - .3 --> goal=1.3*COUNT+START_COUNTING, count=ZERO, letter_memory=4*IDENTITY*arg1',#letter_memory=IDENTITY*arg1
        e_store_result_and_update_count =   'dot(goal,COUNT) + letter_memory_status - .4 --> goal=COUNT, answer=%i*~NEXT*letter_memory, number_memory=6*IDENTITY*count' % store_weight, 
        f_store_count_and_compare =    'dot(goal,COUNT) + 1.5*number_memory_status - .4 --> goal=1.4*COMPARE_COUNT+.6*COUNT, count=%i*~NEXT*number_memory, comparison_cleanA=arg2, comparison_cleanB=~NEXT*number_memory' % store_weight,
        g_update_result =    'dot(goal,COMPARE_COUNT) - 2*comparison_status -.9 --> goal=COUNT, letter_memory=3*IDENTITY*answer',
        h_answer_counted =   'dot(goal,COMPARE_COUNT) + 2*comparison_status - .3 --> goal=1.4*COMPARE-COUNT',
        
        #Compare and respond
        v_compare =           'dot(goal,COMPARE) --> goal=(.8*COMPARE)+RESPOND, comparison2_cleanA=2*target, comparison2_cleanB=2*answer, do_learn=GO, in_layer=ITEM1*arg1+ITEM2*arg2+RESULT*answer, correct=ITEM1*arg1+ITEM2*arg2+RESULT*answer',
        w_respond_match =     'dot(goal,RESPOND) + comparison2_status - 0.3  --> goal=DONE-COMPARE, motor=YES',
        x_respond_mismatch =  'dot(goal,RESPOND) - comparison2_status - 0.3 --> goal=DONE-COMPARE, motor=NO',
        
        y_done =              'dot(goal,DONE) + dot(motor,YES+NO) - .5 --> goal=2*DONE',
        z_threshold = '0.1 -->'      
        )

    model.bg = spa.BasalGanglia(actions)
    model.thalamus = spa.Thalamus(model.bg, synapse_channel = .04)

    #to show BG/thalamus rules
    vocab_actions = spa.Vocabulary(model.bg.output.size_out)
    for i, action in enumerate(model.bg.actions.actions):
        vocab_actions.add(action.name.upper(), np.eye(model.bg.output.size_out)[i])
    model.actions = spa.State(model.bg.output.size_out,subdimensions=model.bg.output.size_out,
                          vocab=vocab_actions)
    nengo.Connection(model.thalamus.output, model.actions.input,synapse=None)

    for net in model.networks:
        if net.label is not None and net.label.startswith('channel'):
            net.label = ''
    


# run in python + save
nengo_gui_on = __name__ == 'builtins' #python3

if not(nengo_gui_on):
    os.environ["PYOPENCL_CTX"] = "0:0"    #cur_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script path

    #sim = nengo.Simulator(model,seed=1)
    sim = MyOCLsimulator(model,seed=1)
    
    sim.run(1.5)
    
    model.mem.mem.save('test.npz',sim)

print('\t' + str(sum(ens.n_neurons for ens in model.all_ensembles)) + ' neurons')
print("Problem: " + str(problem))
