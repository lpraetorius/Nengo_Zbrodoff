#import warnings

#from nengo.exceptions import ValidationError
from nengo.params import NumberParam #, IntParam, FrozenObject,  Parameter
#from nengo.utils.compat import is_iterable, is_string, itervalues
import nengo.learning_rules
from nengo.synapses import Lowpass
import numpy as np
from nengo.builder.operator import Reset #,DotInc, ElementwiseInc, Copy
from nengo.builder import Operator, Builder, Signal
#from nengo_ocl import Simulator
from nengo_ocl.utils import as_ascii #, indent, round_up
from mako.template import Template
import pyopencl as cl
from nengo_ocl.plan import Plan



class Voja2(nengo.learning_rules.LearningRuleType):
    """Vector Oja 2 learning rule version 2
    
    Combines positive voja changes for highly responding
    neurons with negative voja changes for weakly responding neurons. 
    bias = 0 is default voja, bias = 1 is full negative voja.

    Modifies an ensemble's encoders to be selective to its inputs.

    A connection to the learning rule will provide a scalar weight for the
    learning rate, minus 1. For instance, 0 is normal learning, -1 is no
    learning, and less than -1 causes anti-learning or "forgetting".

    Parameters
    ----------
    post_tau : float, optional (Default: 0.005)
        Filter constant on activities of neurons in post population.
    learning_rate : float, optional (Default: 1e-2)
        A scalar indicating the rate at which encoders will be adjusted.
    bias : float 0-1, optional (Default: .5)
        Biases learning to default voja (0) or fully negative voja (1).

    Attributes
    ----------
    learning_rate : float
        A scalar indicating the rate at which encoders will be adjusted.
    post_tau : float
        Filter constant on activities of neurons in post population.
    """

    modifies = 'encoders'
    probeable = ('post_filtered', 'scaled_encoders', 'delta')

    post_tau = NumberParam('post_tau', low=0, low_open=True, optional=True)

    def __init__(self, post_tau=0.005, learning_rate=1e-2, bias=.5):
        self.post_tau = post_tau
        self.bias = bias
        super(Voja2, self).__init__(learning_rate, size_in=1)

    @property
    def _argreprs(self):
        args = []
        if self.post_tau is None:
            args.append("post_tau=%s" % self.post_tau)
        elif self.post_tau != 0.005:
            args.append("post_tau=%g" % self.post_tau)
        if self.learning_rate != 1e-2:
            args.append("learning_rate=%g" % self.learning_rate)
        if self.bias != .5:
            args.append("bias=%g" % self.bias)
        return args



class SimVoja2(Operator):
    r"""Simulates a simplified version of Oja's rule in the vector space, version 2.

    Parameters
    ----------
    pre_decoded : Signal
        Decoded activity from presynaptic ensemble, :math:`a_i`.
    post_filtered : Signal
        Filtered postsynaptic activity signal.
    scaled_encoders : Signal
        2d array of encoders, multiplied by ``scale``.
    delta : Signal
        The synaptic weight change to be applied, :math:`\Delta \omega_{ij}`.
    scale : ndarray
        The length of each encoder.
    learning_signal : Signal
        Scalar signal to be multiplied by ``learning_rate``. Expected to be
        either 0 or 1 to turn learning off or on, respectively.
    learning_rate : float
        The scalar learning rate.
    tag : str, optional (Default: None)
        A label associated with the operator, for debugging purposes.

    Attributes
    ----------
    delta : Signal
        The synaptic weight change to be applied, :math:`\Delta \omega_{ij}`.
    learning_rate : float
        The scalar learning rate.
    learning_signal : Signal
        Scalar signal to be multiplied by ``learning_rate``. Expected to be
        either 0 or 1 to turn learning off or on, respectively.
    post_filtered : Signal
        Filtered postsynaptic activity signal.
    pre_decoded : Signal
        Decoded activity from presynaptic ensemble, :math:`a_i`.
    scale : ndarray
        The length of each encoder.
    scaled_encoders : Signal
        2d array of encoders, multiplied by ``scale``.
    tag : str or None
        A label associated with the operator, for debugging purposes.

    Notes
    -----
    1. sets ``[]``
    2. incs ``[]``
    3. reads ``[pre_decoded, post_filtered, scaled_encoders, learning_signal, mags]``
    4. updates ``[delta]``
    """

    def __init__(self, pre_decoded, post_filtered, scaled_encoders, delta,
                 scale, learning_signal, learning_rate, bias, tag=None): #mags
        super(SimVoja2, self).__init__(tag=tag)
        self.scale = scale
        self.learning_rate = learning_rate
        #self.threshold = int(bias*200)
        self.threshold = int(bias*200)

        self.sets = []
        self.incs = []
        self.reads = [
            pre_decoded, post_filtered, scaled_encoders, learning_signal] #,mags
        self.updates = [delta]

    @property
    def delta(self):
        return self.updates[0]
 
    @property
    def learning_signal(self):
        return self.reads[3]

    @property
    def pre_decoded(self):
        return self.reads[0]

    @property
    def post_filtered(self):
        return self.reads[1]

    @property
    def scaled_encoders(self):
        return self.reads[2]

    @property
    def weights(self):
        return self.reads[2]

    def _descstr(self):
        return 'pre=%s, post=%s -> %s' % (
            self.pre_decoded, self.post_filtered, self.delta)

    def make_step(self, signals, dt, rng):
        pre_decoded = signals[self.pre_decoded]
        post_filtered = signals[self.post_filtered]
        scaled_encoders = signals[self.scaled_encoders]
        delta = signals[self.delta]
        learning_signal = signals[self.learning_signal]
        alpha = self.learning_rate * dt
        scale = self.scale[:, np.newaxis]
        threshold = self.threshold
        
        def step_simvoja2():
        
            shifted_post = 1/(post_filtered-threshold)
            pre_delta = alpha * learning_signal * (scale * np.outer(shifted_post, pre_decoded) - shifted_post[:, np.newaxis] * scaled_encoders)
            
            mod_enc = pre_delta + scaled_encoders
            
            #normalising factor keeping encoders inside the radius of the ensemble
            mag = np.linalg.norm(mod_enc, axis=1)
            
            mod_enc = 1*scale / mag[:, None] * mod_enc #assumes radius is 1

            #final delta
            delta[...] = mod_enc - scaled_encoders
           
        return step_simvoja2



@Builder.register(Voja2)
def build_voja2(model, voja2, rule):
    """Builds a `.Voja2` object into a model.

    Calls synapse build functions to filter the post activities,
    and adds a `.SimVoja2` operator to the model to calculate the delta.

    Parameters
    ----------
    model : Model
        The model to build into.
    voja : Voja
        Learning rule type to build.
    rule : LearningRule
        The learning rule object corresponding to the neuron type.

    Notes
    -----
    Does not modify ``model.params[]`` and can therefore be called
    more than once with the same `.Voja` instance.
    """

    conn = rule.connection

    # Filtered post activity
    post = conn.post_obj
    if voja2.post_tau is not None:
        post_filtered = model.build(
            Lowpass(voja2.post_tau), model.sig[post]['out'])
    else:
        post_filtered = model.sig[post]['out']

    # Learning signal, defaults to 1 in case no connection is made
    # and multiplied by the learning_rate * dt
    learning = Signal(np.zeros(rule.size_in), name="Voja2:learning")
    assert rule.size_in == 1
    model.add_op(Reset(learning, value=1.0))
    model.sig[rule]['in'] = learning  # optional connection will attach here

    scaled_encoders = model.sig[post]['encoders']
    

    # The gain and radius are folded into the encoders during the ensemble
    # build process, so we need to make sure that the deltas are proportional
    # to this scaling factor
    encoder_scale = model.params[post].gain / post.radius
    assert post_filtered.shape == encoder_scale.shape

    model.add_op(
        SimVoja2(pre_decoded=model.sig[conn]['out'],
                post_filtered=post_filtered,
                scaled_encoders=scaled_encoders,
                delta=model.sig[rule]['delta'],
                scale=encoder_scale,
                learning_signal=learning,
                learning_rate=voja2.learning_rate,
                bias=voja2.bias))

    model.sig[rule]['scaled_encoders'] = scaled_encoders
    model.sig[rule]['post_filtered'] = post_filtered




#OCL - not finished
def plan_voja2(queue, pre, post, enc, new_enc, learn, scale, alpha, threshold, tag=None):
    assert (len(pre) == len(post) == len(enc) == len(new_enc) == len(learn) == alpha.size == len(scale))
    N = len(pre)

    for arr in (learn,):  # scalars
        assert (arr.shape0s == 1).all()
        assert (arr.shape1s == 1).all()
    for arr in (pre, post, scale):  # vectors
        assert (arr.shape1s == 1).all()
    for arr in (enc,new_enc):  # matrices
        assert (arr.stride1s == 1).all()

    assert (post.shape0s == enc.shape0s).all()
    assert (pre.shape0s == enc.shape1s).all()

    assert (pre.ctype == post.ctype == enc.ctype ==
            learn.ctype == scale.ctype == alpha.ctype)

    
    text = """
    __kernel void voja2(
        __global const int *shape0s,
        __global const int *shape1s,
        __global const int *pre_stride0s,
        __global const int *pre_starts,
        __global const ${type} *pre_data,
        __global const int *post_stride0s,
        __global const int *post_starts,
        __global const ${type} *post_data,
        __global const int *enc_stride0s,
        __global const int *enc_starts,
        __global const ${type} *enc_data,
        __global const int *new_enc_stride0s,
        __global const int *new_enc_starts,
        __global ${type} *new_enc_data,
        __global const int *learn_starts,
        __global const ${type} *learn_data,
        __global const int *scale_stride0s,
        __global const int *scale_starts,
        __global const ${type} *scale_data,
        __global const ${type} *alphas,
        __global const ${type} *thresholds
    )
    {
        const int ij = get_global_id(0); //indexes of cores
        const int k = get_global_id(1); // index of cores

        const int shape0 = shape0s[k];
        const int shape1 = shape1s[k];
        const int i = ij / shape1;
        const int j = ij % shape1;

        //__global ${type} *delta = delta_data + delta_starts[k];
        
        const ${type} pre = pre_data[pre_starts[k] + j*pre_stride0s[k]];
        const ${type} post = post_data[post_starts[k] + i*post_stride0s[k]];
        const ${type} enc = enc_data[enc_starts[k] + i*enc_stride0s[k] + j];
        __global ${type} *new_enc = new_enc_data + new_enc_starts[k];

        const ${type} learn = learn_data[learn_starts[k]];
        const ${type} scale = scale_data[scale_starts[k] + i*scale_stride0s[k]];
        
        const ${type} alpha = alphas[k];
        const ${type} threshold = thresholds[k];
        
        if (i < shape0) {
            new_enc[i*new_enc_stride0s[k] + j] = alpha * learn *  (1 / (post-threshold)) * (scale * pre - enc) + enc; 
            //printf(" %f ", new_enc[i*new_enc_stride0s[k] + j]);
        }
    }
    """
#sort of for loop over k, i, j #threshold, + enc
    textconf = dict(type=pre.ctype)
    text = as_ascii(Template(text, output_encoding='ascii').render(**textconf))

    full_args = (
        new_enc.cl_shape0s, new_enc.cl_shape1s,
        pre.cl_stride0s, pre.cl_starts, pre.cl_buf,
        post.cl_stride0s, post.cl_starts, post.cl_buf,
        enc.cl_stride0s, enc.cl_starts, enc.cl_buf,
        new_enc.cl_stride0s, new_enc.cl_starts, new_enc.cl_buf,
        learn.cl_starts, learn.cl_buf,
        scale.cl_stride0s, scale.cl_starts, scale.cl_buf,
        alpha, threshold,
    )
    _fn = cl.Program(queue.context, text).build().voja2
    _fn.set_args(*[arr.data for arr in full_args])

    lsize = None
    gsize = (new_enc.sizes.max(), N)
    plan = Plan(queue, _fn, gsize, lsize=lsize, name="cl_voja2", tag=tag)
    plan.full_args = full_args     # prevent garbage-collection
    plan.flops_per_call = 5 * new_enc.sizes.sum()
    plan.bw_per_call = (pre.nbytes + post.nbytes + enc.nbytes + new_enc.nbytes +
                        learn.nbytes + scale.nbytes + alpha.nbytes + threshold.nbytes)
    return plan



def plan_voja2_prep_norm(queue, new_enc, mag, tag=None):
    assert (len(new_enc) == len(mag))
    N = len(mag)

   
    for arr in (mag,):  # vectors
        assert (arr.shape1s == 1).all()
    for arr in (new_enc,):  # matrices
        assert (arr.stride1s == 1).all()

    assert (new_enc.ctype == mag.ctype)
    
    
    text = """
    __kernel void voja2_prep_norm(
        __global const int *shape0s,
        __global const int *shape1s,
        __global const int *new_enc_stride0s,
        __global const int *new_enc_starts,
        __global const ${type} *new_enc_data,
        __global const int *mag_stride0s,
        __global const int *mag_starts,
        __global ${type} *mag_data
    )
    {
        const int i = get_global_id(0);
        const int k = get_global_id(1);

        const int shape0 = shape0s[k];
        const int shape1 = shape1s[k];
        
        __global ${type} *mag = mag_data + mag_starts[k];
        __global const ${type} *new_enc = new_enc_data + new_enc_starts[k];
        
        //printf(" shape0 %i shape1 %i", shape0, shape1);
        //shape 0 = 1000
        //shape 1 = 512
        
        if (i < shape0) {
            //printf("  i %i ", i);
            ${type} norm = 0.;
            for (int d = 0; d < shape1; ++d) {
                //printf("%f ", new_enc[i*new_enc_stride0s[k] + d]);
                ${type} enc_val = new_enc[i*new_enc_stride0s[k] + d];
                norm += enc_val * enc_val;
                //printf("  d %i ", d);
            }
            //printf("sqrt %f", sqrt(norm));
            //mag[i*mag_stride0s[k]] = sqrt(norm);   
            //printf("mag %f", mag[i*mag_stride0s[k]]);  
            mag[i] = sqrt(norm);    
        }          
    }
    """


    textconf = dict(type=new_enc.ctype)
    text = as_ascii(Template(text, output_encoding='ascii').render(**textconf))

    full_args = (
        new_enc.cl_shape0s, new_enc.cl_shape1s,
        new_enc.cl_stride0s, new_enc.cl_starts, new_enc.cl_buf,
        mag.cl_stride0s, mag.cl_starts, mag.cl_buf,
    )
    _fn = cl.Program(queue.context, text).build().voja2_prep_norm
    _fn.set_args(*[arr.data for arr in full_args])

    lsize = None
    gsize = (mag.shape0s.max(), N)
    plan = Plan(queue, _fn, gsize, lsize=lsize, name="cl_voja2_prep_norm", tag=tag)
    plan.full_args = full_args     # prevent garbage-collection
    plan.flops_per_call = 5 * mag.sizes.sum()
    plan.bw_per_call = (new_enc.nbytes + mag.nbytes)
    return plan
    
    
#     #normalising factor keeping encoders inside the radius of the ensemble
#     mag = np.linalg.norm(mod_enc, axis=1)


def plan_voja2_apply_norm(queue, enc, new_enc, delta, mag, scale, tag=None):
    assert (len(enc) == len(new_enc) == len(delta) == len(mag) == len(scale))
    N = len(enc)

    for arr in (mag, scale):  # vectors
        assert (arr.shape1s == 1).all()
    for arr in (enc,new_enc,delta):  # matrices
        assert (arr.stride1s == 1).all()

  
    assert (enc.ctype == new_enc.ctype == delta.ctype ==
            mag.ctype == scale.ctype)

    text = """
    __kernel void voja2_apply_norm(
        __global const int *shape0s,
        __global const int *shape1s,
        __global const int *enc_stride0s,
        __global const int *enc_starts,
        __global const ${type} *enc_data,
        __global const int *new_enc_stride0s,
        __global const int *new_enc_starts,
        __global const ${type} *new_enc_data,
        __global const int *delta_stride0s,
        __global const int *delta_starts,
        __global ${type} *delta_data,
        __global const int *mag_stride0s,
        __global const int *mag_starts,
        __global const ${type} *mag_data,
        __global const int *scale_stride0s,
        __global const int *scale_starts,
        __global const ${type} *scale_data
    )
    {
        const int ij = get_global_id(0);
        const int k = get_global_id(1);

        const int shape0 = shape0s[k];
        const int shape1 = shape1s[k];
        const int i = ij / shape1;
        const int j = ij % shape1;

        __global ${type} *delta = delta_data + delta_starts[k];
        const ${type} enc = enc_data[enc_starts[k] + i*enc_stride0s[k] + j];
        const ${type} new_enc = new_enc_data[new_enc_starts[k] + i*new_enc_stride0s[k] + j];
        const ${type} mag = mag_data[mag_starts[k] + i*mag_stride0s[k]];
        //const ${type} mag = mag_data[mag_starts[k] + j];

        const ${type} scale = scale_data[scale_starts[k] + i*scale_stride0s[k]];

        if (i < shape0) {
            //printf(" mag %f", mag);
            delta[i*delta_stride0s[k] + j] = 1 * scale / mag * new_enc - enc;
            //delta[i*delta_stride0s[k] + j] = new_enc - enc;
            
        }
    }
    """
 
    textconf = dict(type=enc.ctype)
    text = as_ascii(Template(text, output_encoding='ascii').render(**textconf))

    full_args = (
        delta.cl_shape0s, delta.cl_shape1s,
        enc.cl_stride0s, enc.cl_starts, enc.cl_buf,
        new_enc.cl_stride0s, new_enc.cl_starts, new_enc.cl_buf,
        delta.cl_stride0s, delta.cl_starts, delta.cl_buf,
        mag.cl_stride0s, mag.cl_starts, mag.cl_buf,
        scale.cl_stride0s, scale.cl_starts, scale.cl_buf,
    )
    _fn = cl.Program(queue.context, text).build().voja2_apply_norm
    _fn.set_args(*[arr.data for arr in full_args])

    lsize = None
    gsize = (delta.sizes.max(), N)
    plan = Plan(queue, _fn, gsize, lsize=lsize, name="cl_voja2_apply_norm", tag=tag)
    plan.full_args = full_args     # prevent garbage-collection
    plan.flops_per_call = 5 * delta.sizes.sum()
    plan.bw_per_call = (enc.nbytes + new_enc.nbytes + delta.nbytes + mag.nbytes + scale.nbytes)
    return plan

    
# class MyOCLsimulator(Simulator):
# 
#     def plan_SimVoja2(self, ops):
#             pre = self.all_data[[self.sidx[op.pre_decoded] for op in ops]]
#             post = self.all_data[[self.sidx[op.post_filtered] for op in ops]]
#             encoders = self.all_data[[self.sidx[op.scaled_encoders] for op in ops]]
# 
#             new_encoders = self.RaggedArray([np.zeros(op.scaled_encoders.shape) for op in ops], dtype=np.float32)
# 
#             delta = self.all_data[[self.sidx[op.delta] for op in ops]]
#             learning_signal = self.all_data[[self.sidx[op.learning_signal] for op in ops]]
#             scale = self.RaggedArray([op.scale for op in ops], dtype=np.float32)
#             alpha = self.Array([op.learning_rate * self.model.dt for op in ops])
#             threshold = self.Array([op.threshold for op in ops]) 
#             mag = self.RaggedArray([np.zeros(op.scale.shape) for op in ops], dtype=np.float32) #correct? (1,1000)
# 
#             return [
#                 plan_voja2(self.queue, pre, post, encoders, new_encoders, learning_signal, scale, alpha, threshold),
#                 plan_voja2_prep_norm(self.queue, new_encoders, mag),
#                 plan_voja2_apply_norm(self.queue, encoders, new_encoders, delta, mag, scale)]
