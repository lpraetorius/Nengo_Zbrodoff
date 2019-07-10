import nengo
import nengo.spa as spa
import numpy as np
import importlib

import voja2_rule
importlib.reload(voja2_rule)

import bcm2_rule
importlib.reload(bcm2_rule)

from nengo.utils.builder import default_n_eval_points
import scipy
import scipy.special
import scipy.sparse

class Mem_Voja2_Pes_Hop_TwoLayers(nengo.Network):
    def __init__(self, n_neurons=1000, n_neurons_out=1000, dimensions=None,vocab=None,
                 intercepts_mem=nengo.dists.Uniform(0,0),
                 intercepts_out=nengo.dists.Uniform(0,0),
                 voja_tau=0.005,
                 voja2_rate=None, #1e-3,
                 voja2_bias=1, #1 is no bias
                 pes_rate=None, #1e-3, None means no output layer (for PES/BCM learning), 0 no learning
                 bcm_rate=None, #1e-10, None means no recurrent connections, 0 no learning
                 bcm_theta=1,
                 bcm_max_weight=1e-5,
                 bcm_diagonal0=True,
                 label=None,
                 dec_ini = 0,
                 output_radius = 1,
                 seed=None,
                 load_from=None,
                 fwd_dens=.05,
                 fwd_multi=0, #1 
                 fb = 0,
                 ):
        super(Mem_Voja2_Pes_Hop_TwoLayers, self).__init__(label=label)
        
        #print('SEED %i' % seed)

        if (n_neurons is None or dimensions is None) and load_from is None:
            error('Either provide load_from or n_neurons and dimensions.')
     
        if load_from is not None:
        
            data = np.load(load_from + '.npz', allow_pickle=True)
            encoders = data['enc']
            decoders = data['dec']
            hop_weights = data['hop']
            intercepts_mem = data['intercepts_mem']
            #intercepts_out = data['intercepts_out']
            dimensions = decoders.shape[0]
            n_neurons = decoders.shape[1]
            n_neurons_out = int(data['n_neurons_out'])
            output_radius = data['output_radius']
            fwd_multi = data['fwd_multi']
            fwd_matrices= data['fwd_matrices']

            if seed is None:
                seed = data['seed']
            else:
                assert seed == int(data['seed'])
        
            
        else:
            rng = np.random
            rng.seed(seed)
            dist =  nengo.dists.UniformHypersphere()
            encoders = dist.sample(n_neurons, dimensions, rng=rng)
            if dec_ini == 0:
                decoders = np.zeros((dimensions, n_neurons), dtype=float)
            else:
                #decoders = np.random.normal(0, dec_ini, size=(dimensions, n_neurons))
                decoders = np.random.uniform(-dec_ini, dec_ini, size=(dimensions, n_neurons))
            hop_weights = np.zeros((n_neurons, n_neurons))
            fwd_matrices = None
        
        
        self.seed = seed
        self.voja2_rate = voja2_rate
        self.pes_rate = pes_rate
        self.bcm_rate = bcm_rate
        self.decoders = decoders
        self.hop_weights = hop_weights
        self.n_neurons_out = n_neurons_out
        self.intercepts_out = intercepts_out
        self.output_radius = output_radius
        self.fwd_multi = fwd_multi
        self.fwd_matrices = fwd_matrices
        
        with self:
           
           
            self.input = nengo.Node(None, size_in=dimensions)

            #initialize memory
            self.mem = nengo.Ensemble(n_neurons=n_neurons,
                                      dimensions=dimensions,
                                      intercepts=intercepts_mem,
                                      encoders=encoders,
                                      seed=seed
                                      )
                              
            
            #build output layer 
      
            rad = output_radius
            if fb > 0:
                self.output_layer = spa.State(dimensions,vocab=vocab,neurons_per_dimension=n_neurons_out, label='retrieval_out',seed=seed,feedback=fb,feedback_synapse=.05)
            else:
                self.output_layer = spa.State(dimensions,vocab=vocab,neurons_per_dimension=n_neurons_out, label='retrieval_out',seed=seed)#n_neurons_out, 
        
            
            #set intercepts and radius
            for ens in self.output_layer.all_ensembles:
                ens.intercepts=intercepts_out
                ens.radius *= rad
                
            for c in self.output_layer.all_connections:
                if c.post_obj is self.output_layer.output:
                    #c.scale_eval_points=False
                    ens = c.pre_obj
                    n_eval_points = default_n_eval_points(ens.n_neurons, ens.dimensions)
                    c.eval_points=ens.eval_points.sample(n_eval_points, ens.dimensions)/rad
        
            
            #current forwarding to output layer.
            if fwd_multi > 0:
                
                #via node
                self.act_node_in = nengo.Node(None, size_in=1)
                nengo.Connection(self.mem.neurons, self.act_node_in, transform=np.ones((1, self.mem.n_neurons))/self.mem.n_neurons*fwd_multi,synapse=None)
       
                density = fwd_dens
                conn_matrices = []
                for ens_out in self.output_layer.all_ensembles:
                    if fwd_matrices is None:
                        connection_matrix = scipy.sparse.random(ens_out.n_neurons,1,density=density)
                        connection_matrix = connection_matrix != 0
                        nengo.Connection(self.act_node_in,ens_out.neurons,transform = connection_matrix.toarray())
                        conn_matrices.append(connection_matrix.toarray())
                    else:
                        nengo.Connection(self.act_node_in,ens_out.neurons,transform = fwd_matrices.pop(0))
                self.fwd_matrices = conn_matrices
                    
            
            #encoder learning
            if voja2_rate is None or voja2_rate == 0: #if no encoder learning, make default connection
                self.conn_in = nengo.Connection(self.input, self.mem,synapse=0)
            else:

                #voja 2 rule version
                print('Voja 2 - rule!')
                learning_rule_type = voja2_rule.Voja2(post_tau=voja_tau,
                                                learning_rate=voja2_rate,
                                                bias=voja2_bias)
        
                self.conn_in = nengo.Connection(self.input, self.mem,
                             learning_rule_type=learning_rule_type,synapse=0)


            #decoder learning
            if pes_rate is not None and pes_rate > 0:

                self.conn_out = nengo.Connection(self.mem.neurons, self.output_layer.input,
                transform=decoders,
                learning_rule_type=nengo.PES(learning_rate=pes_rate)
                )
            else:
                self.conn_out = nengo.Connection(self.mem.neurons, self.output_layer.input,
                transform=decoders
                )

            #if pes_rate is not None and pes_rate > 0:
            self.correct = nengo.Node(None, size_in=dimensions)

            self.learn_control = nengo.Node(
                lambda t, x: x[:-1] if x[-1] < 0.5 else x[:-1]*0,
            size_in=dimensions+1)
            if pes_rate is not None and pes_rate > 0:
                nengo.Connection(self.learn_control, self.conn_out.learning_rule,)
            nengo.Connection(self.output_layer.output, self.learn_control[:-1], synapse=None)
            nengo.Connection(self.correct, self.learn_control[:-1], transform=-1, synapse=None)
            self.stop_pes = nengo.Node(None, size_in=1)
            nengo.Connection(self.stop_pes, self.learn_control[-1], synapse=None)


            #hopfield learning/BCM
            if bcm_rate is not None: #recur connection
                self.conn_hop = nengo.Connection(self.mem.neurons, self.mem.neurons, transform=hop_weights, synapse=.05)
                
                if bcm_rate > 0:
                    self.conn_hop.learning_rule_type = bcm2_rule.BCM2(learning_rate=bcm_rate, theta_tau=bcm_theta, max_weight=bcm_max_weight,diagonal0=bcm_diagonal0)
                

 
    def create_weight_probes(self):
        with self:
            self.probe_encoder = nengo.Probe(self.mem, 'scaled_encoders',
                                             sample_every=1)
            if self.pes_rate is not None and self.pes_rate > 0:
                self.probe_decoder = nengo.Probe(self.conn_out, 'weights',
                                             sample_every=1)
            if self.bcm_rate is not None and self.bcm_rate > 0:
                self.probe_hop = nengo.Probe(self.conn_hop, 'weights',
                                            sample_every=1)


    def save(self, filename, sim):
        #print(sim.data[self.probe_encoder])
    
        enc = sim.data[self.probe_encoder][-1]
        
        if self.pes_rate is not None and self.pes_rate > 0:
            dec = sim.data[self.probe_decoder][-1]
        else:
            dec = self.decoders
        
        if self.bcm_rate is not None and self.bcm_rate > 0:
            hop = sim.data[self.probe_hop][-1]
        else:
            hop = self.hop_weights
        
        #print(sim.data)
        #print(sim.data[self.mem])
        #print(sim.[self.mem.output_layer])
            
        np.savez(filename, enc=enc, dec=dec, hop=hop,
            seed=self.seed,
            intercepts_mem=sim.data[self.mem].intercepts.copy(),
            #intercepts_out = self.intercepts_out, doesn't work as it's not an array
            n_neurons_out=self.n_neurons_out,
            output_radius=self.output_radius,
            fwd_multi=self.fwd_multi,
            fwd_matrices=self.fwd_matrices
            )
            


     
