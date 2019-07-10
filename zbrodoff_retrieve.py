###### Import necessary packages & General Settings ######

import nengo
import nengo.spa as spa
import numpy as np

# The following two packages are more advanced nengo components that I developed
# to make cognitive processing more stable and easier to control.
import assoc_mem_acc #associative memory with and status indicator that shows if something is retrieved
import compare_acc_zbrodoff #compare object that can compare two semantic pointers, and collects evidence in comparison_status


print('\nSettings:')

# Use a fixed random seed, so building the model goes faster (after running it the first time).
# The model should work with different random seeds, but might make different mistakes. The seed
# determines, for example, the tuning curves of the neurons.
fixed_seed = True #False
if fixed_seed:
    fseed = 2 #
    np.random.seed(fseed)
    print('\tFixed seed: %i' % fseed)
else:
    fseed = None
    print('\tRandom seed')
    
    
# Set dimensions of semantic pointers. Higher dimensions will make the model more reliable at
# the cost of more neurons.
D = 128  #for numbers and letters
Dlow = 16 #for goal and motor
print('\tDimensions: %i, %i' % (D,Dlow))



####### Vocabularies #######

rng_vocabs = np.random.RandomState(seed=fseed)

# The model uses four different vocabularies: all concepts, learned facts, goal states, and motor states. 
# See the assignment text for an explanation of vocabularies.

# 1. A vocabulary is created that contains all the concepts the model knows about.

# The model knows about these letters and numbers. If you add more, your might need
# to increase the dimesionality of the semantic pointers.

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
numbers = ['ZERO', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE']

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

# Second, add A+2=C facts. The model knows about A-F + 2-4 - it will not recognize other problems,
# unless you add them. If you do, you might need to increase the dimensionality. 
# Problems are encoded as ITEM1*A + ITEM2*TWO + RESULT*C: the circular convolutions of 
# slot names ITEM1, ITEM2, and RESULT, with the contents of the slots.
cnt_letter=0
for letter in letters[0:6]:
    cnt_letter += 1
    cnt_number=1
    for number in numbers[2:5]:
        cnt_number += 1
        vocab_problems.add('%s+%s=%s' % (letter, number, letters[cnt_letter+cnt_number-1]),
            vocab_problems.parse('ITEM1*%s + ITEM2*%s + RESULT*%s' % (letter, number, letters[cnt_letter+cnt_number-1]))) 


# 3. Vocab with goal states, comparable to how we control an ACT-R model.
# This vocab uses a lower dimensionality, as it only has to store 5 different pointers.
vocab_goal = spa.Vocabulary(Dlow,rng=rng_vocabs)
vocab_goal.parse('START+RETRIEVE+COMPARE+RESPOND+DONE') # add them all at once


# 4. Vocab_motor with two motor states.
vocab_motor = spa.Vocabulary(Dlow,rng=rng_vocabs)
vocab_motor.parse('YES')
vocab_motor.parse('NO')           


################## Set the problem that is presented ##################
#syntax for the problem B+2=E is ['B', '2', 'D']
problem = ['C', 'ONE', 'D'] #change this line to present other problems.


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
        




################## Model ################## 

model = spa.SPA(seed=fseed)

with model:
   
    #Goal: a network with two slots: goal and target. 
    model.goalnet = nengo.Network(seed=fseed)
    with model.goalnet:
        model.goal = spa.State(Dlow,vocab=vocab_goal,feedback=1) #goal state
        model.target = spa.State(D,vocab=vocab_concepts,feedback=0) #target answer from input

    #Imaginal: a network with three slots: arg1, arg2, and answer.
    model.imaginal = nengo.Network(seed=fseed)
    with model.imaginal:
        model.arg1 = spa.State(D, vocab=vocab_concepts,feedback=0) #argument 1 from input
        model.arg2 = spa.State(D, vocab=vocab_concepts,feedback=0) #argument 2 from input
        model.answer = spa.State(D, vocab=vocab_concepts,feedback=1,feedback_synapse=.05) #result from retrieval (or counting in the full model)
    
            
    #set the inputs to the model (bypassing the need for a visual system)
    model.input = spa.Input(goal=goal_func, target=target_func, arg1=arg1_func, arg2=arg2_func)


    #Declarative Memory
    model.declarative = assoc_mem_acc.AssociativeMemoryAccumulator(input_vocab = vocab_problems, wta_output=True, wta_inhibit_scale=10, threshold=.5)

    #Comparison
    model.comparison = compare_acc_zbrodoff.CompareAccumulator(vocab_compare = vocab_concepts,status_scale = .6,status_feedback = .6)

    #Motor
    model.motor = spa.State(dimensions=Dlow, vocab=vocab_motor)
    
    #Basal Ganglia & Thalamus
    actions = spa.Actions(
        
        #encode and retrieve
        a_retrieve =  'dot(goal,START) - declarative_status -.2 --> goal=START+RETRIEVE, declarative=ITEM1*arg1 + ITEM2*arg2', 
        #a_retrieve= goal equals START, - retrieval status - .2 --> switch goal to retrieve, declare current problem as arg1 + arg2
        
        b_answer_retrieved =   'dot(goal,RETRIEVE) + declarative_status - .3 --> goal=(.8*RETRIEVE)+COMPARE, answer=4*~RESULT*declarative',
        # b_answer_retrieved= if goal =retrieve - retrieval status - .3 --> switch goal to COMPARE, retrieve answer corresponding to current problem 
         
        #compare
        v_compare =           'dot(goal,COMPARE) --> goal=(.8*COMPARE)+RESPOND, comparison_cleanA=2*target, comparison_cleanB=4*answer',
        #v_compare =    if goal = COMPARE --> switch goal to RESPOND, compare target with answer
        
        #respond
        
        w_yes = 'dot(goal, RESPOND) --> goal=RESPOND+DONE, motor=YES',
        x_no = 'dot(goal, RESPOND) - comparison_status --> goal=RESPOND+DONE, motor=NO',
        #finished
        y_done =              'dot(goal,DONE) + dot(motor,YES+NO) - .5 --> goal=2*DONE',
    ) 

    model.bg = spa.BasalGanglia(actions)
    model.thalamus = spa.Thalamus(model.bg, synapse_channel = .04)

print('\t' + str(sum(ens.n_neurons for ens in model.all_ensembles)) + ' neurons')

"""
Assignment Part 2

1. The model is not trained for recognizing problems that include the semantic pointer ONE. 
As a consequence, it does not move past the retrieval state, as it looks for a problem in memory including ONE.
We can see that the declarative module is "stuck" at C+TWO=E, because it finds the next best problem that starts with C.

2. The model would need to be able to transform LETTERS into NUMBERS. 
For this, it would have to know about the order of the alphabet (C comes after B comes after A).
Initially, the model would determine the value of a letter by going through the alphabet in order.
If a letter was presented multiple times, the model could retrieve the corresponding value from memory, increasing the speed.
Afterwards, the model should be able to add the values together.
Finally, the result needs to be transformed back into the corresponding letter(again, first by counting and, with increasing training, with memory).

3. The idea behind this is, that the basal ganglia is not directly responible for comparing the components, which we would like to include in the cognitive architecture.

4. 
"""