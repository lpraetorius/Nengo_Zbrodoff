 # Short-term plasticity example

import nengo
import numpy as np
import nengo.spa as spa
import os, inspect, csv

D=32

vocab = spa.Vocabulary(D)
vocab.parse('LEFT')
vocab.parse('RIGHT')
vocab.parse('IMPULSE')

def input_func(t):
    t=t-.5
    if t > 0 and t < .25:
        return '.4*%s' % stim
    elif t > 1.00 and t < 1.050:
        return '.4*IMPULSE'
    else:
        return '0'
    
def learn_func(t):
    t=t-.5
    if t > 0 and t < .25:
        return 0
    elif t > 2.00 and t < 2.050:
        return 0
    else:
        return 1   
    
    
model = spa.SPA(seed=1)
with model:
    
    #vision
    model.vision = spa.State(D,vocab=vocab)
    model.input = spa.Input(vision=input_func)         

    #Working Memory
    model.WM_input = nengo.Ensemble(n_neurons=D*50, dimensions=D,intercepts=nengo.dists.Uniform(-.01,.1))
    model.WM_output = nengo.Ensemble(n_neurons=D*50, dimensions=D,intercepts=nengo.dists.Uniform(-.01,.1))
    
    #recurrent connection
    nengo.Connection(model.WM_input,model.WM_input,transform=.4,synapse=.1)

    #learning connection
    model.learn_conn = nengo.Connection(model.WM_input, model.WM_output, transform=np.zeros((D, D), dtype=float),
        learning_rule_type=nengo.PES(learning_rate=5e-3 ))
                                  
 
    
    #learning control
    model.learn_control = nengo.Node(lambda t, x: x[:-1] if x[-1] < 0.5 else x[:-1]*0, size_in=D+1)
    nengo.Connection(model.learn_control, model.learn_conn.learning_rule)
    nengo.Connection(model.WM_output, model.learn_control[:-1], synapse=None)
    nengo.Connection(model.WM_input, model.learn_control[:-1], transform=-1, synapse=None)
    
    model.stop_pes = nengo.Node(learn_func, size_out=1)
    nengo.Connection(model.stop_pes, model.learn_control[-1],synapse=None)
    
    
    #connect vision
    nengo.Connection(model.vision.output, model.WM_input)#,synapse=.04)
    
    
    #check representations
    model.WM_in_state = spa.State(D,vocab=vocab)
    for ens in model.WM_in_state.all_ensembles:
        ens.neuron_type = nengo.Direct()
        nengo.Connection(model.WM_input, model.WM_in_state.input,synapse=None)
        
    model.WM_out_state = spa.State(D,vocab=vocab)
    for ens in model.WM_out_state.all_ensembles:
        ens.neuron_type = nengo.Direct()
        nengo.Connection(model.WM_output, model.WM_out_state.input,synapse=None)
        
    #collect activity
    model.act_WM = nengo.Node(None, size_in=1)
    nengo.Connection(model.WM_input.neurons, model.act_WM, transform=np.ones((1, model.WM_input.n_neurons)),synapse=None)
    nengo.Connection(model.WM_output.neurons, model.act_WM, transform=np.ones((1, model.WM_output.n_neurons)),synapse=None)

    #probes
    model.syn_weights_probe = nengo.Probe(model.learn_conn, 'weights')
    model.act_wm_probe = nengo.Probe(model.act_WM,synapse=.01)
    model.WM_in_probe = nengo.Probe(model.WM_input.neurons,synapse=.01)
    model.WM_out_probe = nengo.Probe(model.WM_output.neurons,synapse=.01)


# run in python + save
nengo_gui_on = __name__ == 'builtins' #python3
stim = "LEFT"

if not(nengo_gui_on):

    cur_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script path
    
    sim = nengo.Simulator(model,seed=1)
    
    #trial 1
    stim = 'LEFT'
    sim.run(3)
        
    #save weights
    np.savetxt("WM_in_probe_%s.csv" % stim, sim.data[model.WM_in_probe], delimiter=",")
    np.savetxt("WM_out_probe_%s.csv" % stim, sim.data[model.WM_in_probe], delimiter=",")
    np.savetxt("act_wm_probe_%s.csv" % stim, sim.data[model.act_wm_probe], delimiter=",")

    #doesn't work becasue 3D - np.savetxt("foo.csv", sim.data[model.syn_weights_probe], delimiter=",")
   
    #reset simulator, clean probes thoroughly
    sim.reset()
    for probe in sim.model.probes:
        del sim._probe_outputs[probe][:]
    del sim.data
    sim.data = nengo.simulator.ProbeDict(sim._probe_outputs)   
   
   
    #trial 2
    stim = 'RIGHT'
    sim.run(3)
        
    #save weights
    np.savetxt("WM_in_probe_%s.csv" % stim, sim.data[model.WM_in_probe], delimiter=",")
    np.savetxt("WM_out_probe_%s.csv" % stim, sim.data[model.WM_in_probe], delimiter=",")
    np.savetxt("act_wm_probe_%s.csv" % stim, sim.data[model.act_wm_probe], delimiter=",")

        

          
