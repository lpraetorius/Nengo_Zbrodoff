import nengo
import nengo.spa as spa
#from nengo.network import Network

import mem_voja2_pes_hop_twolayers
import importlib
#from nengo.spa.vocab import Vocabulary

importlib.reload(mem_voja2_pes_hop_twolayers)

class SPA_Mem_Voja2_Pes_Hop_TwoLayers(spa.module.Module):
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

