#import nengo
import nengo.spa as spa
#from nengo.network import Network

import mem_voja2_pes_hop_twolayers
import importlib
#from nengo.spa.vocab import Vocabulary

importlib.reload(mem_voja2_pes_hop_twolayers)
import assoc_mem_acc
#from assoc_mem_acc import AssociativeMemoryAccumulator
from nengo.spa.assoc_mem import AssociativeMemory

class SPA_Mem_Voja2_Pes_Hop_TwoLayers(assoc_mem_acc.AssociativeMemoryAccumulator, spa.module.Module):
    #input_vocab = Vocabulary(
    #    'input_vocab', default=None, readonly=True)
    #output_vocab = Vocabulary(
    #    'output_vocab', default=None, readonly=True)    

    def __init__(self, n_neurons=1000, dimensions=None, input_vocab=None, output_vocab=None,
                 label=None, seed=None,
                 add_to_container=None, **keys):
        super(spa.module.Module, self).__init__(label=label, seed=seed,
                            add_to_container=add_to_container)
        if output_vocab == None:
            output_vocab = input_vocab
        
        with self:
            
            self.mem =  mem_voja2_pes_hop_twolayers.Mem_Voja2_Pes_Hop_TwoLayers(
                            n_neurons=n_neurons,
                            dimensions=dimensions,
                            vocab=input_vocab,
                            seed=seed,
                            **keys)
            

        self.input = self.mem.input
        self.output = self.mem.output_layer.output

        if hasattr(self.mem, 'correct'):
            self.correct = self.mem.correct
        self.inputs = dict(default=(self.input, input_vocab))
        
        self.outputs = dict(default=(self.output, output_vocab))
        
        #self.declare_input(self.input, self.input_vocab)
        #self.declare_output(self.output, self.output_vocab)
        
        
        
        
    def __init__(self, status_scale=.8, status_feedback=.8,status_feedback_synapse=.1, threshold_input_detect=.6, bias=0, **kwargs):
        super(assoc_mem_acc.AssociativeMemoryAccumulator, self).__init__(**kwargs)

        with self:
            #memory status indicator
            self.memory_status = nengo.Ensemble(50,1)
            nengo.Connection(self.memory_status,self.memory_status, transform=status_feedback, synapse=status_feedback_synapse)
            
            #positive source for status accumulator: summed similarity
            self.summed_similarity = nengo.Ensemble(n_neurons=100,dimensions=1)
            
            #original version based on similarity
            #nengo.Connection(self.am.elem_output, self.summed_similarity, transform=np.ones((1, self.am.elem_output.size_out))) #take sum

            #new version based on neural activity
            self.summed_act = nengo.Node(None,size_in=1) #node to collect sum of activity to normalize
        
            #connect all ensembles to summed_act    
            for i, ens in enumerate(self.am.am_ensembles):
                mr = (ens.max_rates.high + ens.max_rates.low)/2 #calc average max rate
                nengo.Connection(ens.neurons, self.summed_act, transform=np.ones((1, ens.n_neurons))/(ens.n_neurons*mr),synapse=None)
            
            nengo.Connection(self.summed_act,self.summed_similarity, transform=1) #connect to summed sim without changing it
            nengo.Connection(self.summed_similarity,self.memory_status, transform=status_scale+bias) #use bias to scale this
        
            #negative source for status: switched on when input present
            n_signal = 10 #number of  dimensions used to determine whether input is present
    
            self.switch_in = nengo.Ensemble(n_neurons=n_signal*50,dimensions=n_signal,radius=1) #50 neurons per dimension
            nengo.Connection(self.am.input[0:10],self.switch_in,transform=10,synapse=None) #no synapse, as input signal directly 

            self.switch_detect = nengo.Ensemble(n_neurons=200,dimensions=1)
            nengo.Connection(self.switch_in,self.switch_detect,function=lambda x: np.sqrt(np.sum(x*x)),eval_points=nengo.dists.Gaussian(0,.1))
    
            self.negative_source = nengo.Ensemble(n_neurons=100,dimensions=1)
            nengo.Connection(self.switch_detect,self.negative_source, function=lambda x: 1 if x > threshold_input_detect else 0,synapse=.05)
            nengo.Connection(self.negative_source, self.memory_status, transform=-((status_scale-bias)/2)) 
 
            #bias
            #nengo.Connection(self.negative_source, self.memory_status, transform=bias)
 
            #outputs we can use on LHS
            self.outputs['status'] = (self.memory_status,1)
            self.outputs['summed_sim'] = (self.summed_similarity,1)

