
#import necessary packages
import nengo
import nengo.spa as spa
import numpy as np
import assoc_mem_acc
import compare_acc_zbrodoff
import datetime
import nengo_ocl
import pyopencl
import os
os.environ["PYOPENCL_CTX"] = "0:1"
ctx = pyopencl.create_some_context()


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
    
    
#set dimensions. Values must be divisible by 16
D = 192  #192
Dlow = 96 #96
print('\tDimensions: %i, %i' % (D,Dlow))

#shortcuts to write certain problems in l. 58
a1b = ['A', 'ONE', 'B']
a2b = ['A', 'TWO', 'B']
a1d = ['A', 'ONE', 'D']
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

#The input problem. The problem B+2=D should be presented as ['B', 'TWO', 'D']
#problem = a2b
problem = ['A', 'TWO', 'C']

def vision_input_func(t):
    return'1*(ITEM1*%s + ITEM2*%s + RESULT*%s)' % (problem[0], problem[1], problem[2])

def goal_input_func(t):
    if t < .05:
        return 'START'
    else:
        return '0'




####### Vocabularies #######
rng_vocabs = np.random.RandomState(seed=fseed)


#general vocab of concepts the model knows about
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
numbers = ['ZERO', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE']

vocab_concepts = spa.Vocabulary(D,rng=rng_vocabs)
for letter in letters:
    vocab_concepts.parse(letter)

for number in numbers:
    vocab_concepts.parse(number)
   
vocab_concepts.parse('ITEM1')
vocab_concepts.parse('ITEM2')
vocab_concepts.parse('RESULT')
vocab_concepts.parse('IDENTITY')
vocab_concepts.parse('NEXT')


#vocabs with sequence information

#start with general knowledge (to use same vectors)
vocab_numbers = vocab_concepts.create_subset(numbers + ['IDENTITY', 'NEXT'])
vocab_numbers.readonly = False

cnt_number=0    
for number in numbers[0:(len(numbers)-1)]:
    cnt_number += 1
    vocab_numbers.add('%s->%s' % (number, numbers[cnt_number]),
        vocab_numbers.parse('IDENTITY*%s + NEXT*%s' % (number, numbers[cnt_number])))

#print vocab_numbers.keys

#start with general knowledge (to use same vectors)
vocab_letters = vocab_concepts.create_subset(letters + ['IDENTITY', 'NEXT'])
vocab_letters.readonly = False

cnt_letter=0
for letter in letters[0:(len(letters)-1)]:
    cnt_letter += 1
    vocab_letters.add('%s->%s' % (letter, letters[cnt_letter]),
        vocab_letters.parse('IDENTITY*%s + NEXT*%s' % (letter, letters[cnt_letter])))



#vocab goal
vocab_goal = spa.Vocabulary(Dlow,rng=rng_vocabs)
vocab_goal.parse('START+RETRIEVE+COMPARE+RESPOND+DONE+START_COUNTING+UPDATE_RESULT+STORE_RESULT+UPDATE_COUNT+STORE_COUNT+COMPARE_COUNT')

#vocab reset
vocab_reset = spa.Vocabulary(Dlow,rng=rng_vocabs)
vocab_reset.parse('CLEAR+GO')

#vocab_motor
vocab_motor = spa.Vocabulary(Dlow,rng=rng_vocabs)
vocab_motor.parse('YES')
vocab_motor.parse('NO')           



###### Model #######

model = spa.SPA(seed=fseed)

with model:

    
    #Vision & Goal
    model.vision = spa.State(D,vocab=vocab_concepts) #visual input
    
    model.goalnet = nengo.Network(seed=fseed)
    with model.goalnet:
        model.goal = spa.State(Dlow,vocab=vocab_goal,feedback=.5) #goal state #.75
        model.count = spa.State(D,vocab=vocab_concepts,feedback=.8,feedback_synapse=.05) #the number of counts taken
        model.target = spa.State(D,vocab=vocab_concepts,feedback=1) #target answer from vision
        
    model.input = spa.Input(vision=vision_input_func,goal=goal_input_func)
    
    #Imaginal
    model.imaginal = nengo.Network(seed=fseed)
    with model.imaginal:
        model.arg1 = spa.State(D, vocab=vocab_concepts,feedback=1) #argument 1 from vision
        model.arg2 = spa.State(D, vocab=vocab_concepts,feedback=1) #argument 2 from vision
        model.answer = spa.State(D, vocab=vocab_concepts,feedback=.8,feedback_synapse=.05) #result from retrieval or counting
    
    
    #Number memory
    model.number_memory = assoc_mem_acc.AssociativeMemoryAccumulator(input_vocab = vocab_numbers, wta_output=True, status_scale=.7,threshold=.1,status_feedback=.3)
    #nengo.Connection(model.number_memory.output,model.number_memory.input,transform=.2) #feedback on number memory 
    
    #Alphabet memory
    model.letter_memory = assoc_mem_acc.AssociativeMemoryAccumulator(input_vocab = vocab_letters, wta_output=True, status_scale=.7,threshold=.1,status_feedback=.3)
    #nengo.Connection(model.letter_memory.output,model.letter_memory.input,transform=.2) #feedback on letter memory
    
    #Comparison for count update
    model.comparison = compare_acc_zbrodoff.CompareAccumulator(vocab_compare = vocab_concepts,status_scale = .4,status_feedback = .2, status_feedback_synapse=.05, threshold_cleanup=.1)
    #model.comparison = compare_acc_zbrodoff.CompareAccumulator(vocab_compare = vocab_concepts,status_scale = .6,status_feedback = .2, status_feedback_synapse=.05, threshold_cleanup=.1)
    
    #final Comparison
    model.comparison2 = compare_acc_zbrodoff.CompareAccumulator(vocab_compare = vocab_concepts,status_scale = .6,status_feedback = .2, status_feedback_synapse=.05, threshold_cleanup=.1)

    # increase speed
    #for ens in model.vis_pair.all_ensembles:
    #    ens.neuron_type = nengo.Direct()
        
    #Motor
    model.motor = spa.State(Dlow,vocab=vocab_motor,feedback=1) #motor state       

    #Basal Ganglia & Thalamus
    input_modifier = 6
    store_weight = 12 #10
    actions = spa.Actions(
        
        #encode and retrieve
        a_encode =    'dot(goal,START)    --> goal=START_COUNTING-.8*START, arg1=%g*~ITEM1*vision, arg2=%g*~ITEM2*vision, target=%g*~RESULT*vision' % (input_modifier, input_modifier, input_modifier),
        #a_encode =    'dot(goal,START)    --> goal=START_COUNTING-.8*START, arg1=%g*~ITEM1*vision, arg2=%g*~ITEM2*vision, target=%g*~RESULT*vision' % (input_modifier, input_modifier, input_modifier),
        
        #counting
        d_start_counting = 'dot(goal,START_COUNTING) - letter_memory_status - .3 --> goal=COUNT+START_COUNTING, count=ZERO, letter_memory=IDENTITY*arg1',
        #d_start_counting = 'dot(goal,START_COUNTING) - letter_memory_status - .3 --> goal=COUNT+START_COUNTING, count=ZERO, letter_memory=IDENTITY*arg1',

        e_store_result_and_update_count =   'dot(goal,COUNT) + letter_memory_status - 0.4 --> goal=COUNT, answer=%i*~NEXT*letter_memory, number_memory=IDENTITY*count' % store_weight, 
        #e_store_result_and_update_count =   'dot(goal,COUNT) + letter_memory_status - 0.6 --> goal=COUNT, answer=%i*~NEXT*letter_memory, number_memory=IDENTITY*count' % store_weight, 
        
        f_store_count_and_compare =    'dot(goal,COUNT) + number_memory_status - 0.4 --> goal=1.3*COMPARE_COUNT+.6*COUNT, count=%i*~NEXT*number_memory, comparison_cleanA=2*arg2, comparison_cleanB=2*~NEXT*number_memory' % store_weight,
        #f_store_count_and_compare =    'dot(goal,COUNT) + number_memory_status - 0.6 --> goal=COMPARE_COUNT+.6*COUNT, count=%i*~NEXT*number_memory, comparison_cleanA=2*arg2, comparison_cleanB=2*~NEXT*number_memory' % store_weight,
        
        g_update_result =    'dot(goal,COMPARE_COUNT) - 3*comparison_status -.9 --> goal=COUNT, letter_memory=IDENTITY*answer',
        #g_update_result =    'dot(goal,COMPARE_COUNT) - 3*comparison_status - .6 --> goal=COUNT, letter_memory=IDENTITY*answer',
        
        h_answer_counted =   'dot(goal,COMPARE_COUNT) + 2*comparison_status - .5 --> goal=1.4*COMPARE-COUNT', #- COUNT  here we should store the problem-solution-combination
        #h_answer_counted =   'dot(goal,COMPARE_COUNT) + comparison_status - .5 --> goal=1.2*COMPARE-COUNT', #- COUNT  here we should store the problem-solution-combination
         
        #compare and respond
        v_compare =           'dot(goal,COMPARE) --> goal=(.8*COMPARE)+RESPOND, comparison2_cleanA=2*target, comparison2_cleanB=2*answer',
        #v_compare =           'dot(goal,COMPARE) --> goal=(.8*COMPARE)+RESPOND, comparison_cleanA=2*target, comparison_cleanB=2*answer',
        w_respond_match =     'dot(goal,RESPOND) + comparison2_status - 0.3  --> goal=DONE-COMPARE, motor=YES',
        #w_respond_match =     'dot(goal,RESPOND) + comparison_status - 0.3 --> goal=DONE-COMPARE, motor=YES',
        x_respond_mismatch =  'dot(goal,RESPOND) - comparison2_status - 0.3 --> goal=DONE-COMPARE, motor=NO',
        #x_respond_mismatch =  'dot(goal,RESPOND) - comparison_status - 0.3 --> goal=DONE-COMPARE, motor=NO',
        
        #finished
        y_done =              'dot(goal,DONE) + dot(motor,YES+NO) - .5 --> goal=2*DONE',
        #y_done =              'dot(goal,DONE) + dot(motor,YES+NO) - .5 --> goal=2*DONE',
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

#with nengo_ocl.Simulator(model) as sim:
 #   sim.run(10)
print('\t' + str(sum(ens.n_neurons for ens in model.all_ensembles)) + ' neurons')

