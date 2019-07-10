import warnings

from nengo.config import SupportDefaultsMixin
from nengo.exceptions import ValidationError
from nengo.params import (
    Default,
    IntParam,
    FrozenObject,
    NumberParam,
    Parameter,
    Unconfigurable,
    NdarrayParam,
    BoolParam,
)
from nengo.synapses import Lowpass, SynapseParam
from nengo.utils.numpy import is_iterable

from nengo.learning_rules import LearningRuleType

import numpy as np

from nengo.builder import Builder, Operator, Signal
from nengo.builder.operator import DotInc, ElementwiseInc, Copy, Reset
from nengo.connection import LearningRule
from nengo.ensemble import Ensemble, Neurons
from nengo.exceptions import BuildError
#from nengo.learning_rules import BCM, Oja, PES, Voja
from nengo.node import Node

#from nengo_ocl import Simulator
from nengo_ocl.utils import as_ascii #, indent, round_up
from mako.template import Template
import pyopencl as cl
from nengo_ocl.plan import Plan


class BCM2(LearningRuleType):
    """Bienenstock-Cooper-Munroe learning rule.
    Modifies connection weights as a function of the presynaptic activity
    and the difference between the postsynaptic activity and the average
    postsynaptic activity.
    Notes
    -----
    The BCM rule is dependent on pre and post neural activities,
    not decoded values, and so is not affected by changes in the
    size of pre and post ensembles. However, if you are decoding from
    the post ensemble, the BCM rule will have an increased effect on
    larger post ensembles because more connection weights are changing.
    In these cases, it may be advantageous to scale the learning rate
    on the BCM rule by ``1 / post.n_neurons``.
    Parameters
    ----------
    learning_rate : float, optional (Default: 1e-9)
        A scalar indicating the rate at which weights will be adjusted.
    pre_synapse : `.Synapse`, optional \
                  (Default: ``nengo.synapses.Lowpass(tau=0.005)``)
        Synapse model used to filter the pre-synaptic activities.
    post_synapse : `.Synapse`, optional (Default: ``None``)
        Synapse model used to filter the post-synaptic activities.
        If None, ``post_synapse`` will be the same as ``pre_synapse``.
    theta_synapse : `.Synapse`, optional \
                    (Default: ``nengo.synapses.Lowpass(tau=1.0)``)
        Synapse model used to filter the theta signal.
    max_weights : float, optional (Default: None)
    
    Attributes
    ----------
    learning_rate : float
        A scalar indicating the rate at which weights will be adjusted.
    post_synapse : `.Synapse`
        Synapse model used to filter the post-synaptic activities.
    pre_synapse : `.Synapse`
        Synapse model used to filter the pre-synaptic activities.
    theta_synapse : `.Synapse`
        Synapse model used to filter the theta signal.
    """

    modifies = 'weights'
    probeable = ('theta', 'pre_filtered', 'post_filtered', 'delta')

    learning_rate = NumberParam(
        'learning_rate', low=0, readonly=True, default=1e-9)
    pre_synapse = SynapseParam(
        'pre_synapse', default=Lowpass(tau=0.005), readonly=True)
    post_synapse = SynapseParam(
        'post_synapse', default=None, readonly=True)
    theta_synapse = SynapseParam(
        'theta_synapse', default=Lowpass(tau=1.0), readonly=True)
    max_weight = NumberParam(
         'max_weight', readonly=True, default=None)
    diagonal0 = BoolParam('diagonal0', readonly=True, default=True)

    def __init__(self, learning_rate=Default, pre_synapse=Default,
                 post_synapse=Default, theta_synapse=Default, max_weight=Default,diagonal0=Default,
                 pre_tau=Unconfigurable, post_tau=Unconfigurable,
                 theta_tau=Unconfigurable):
        super().__init__(learning_rate, size_in=0)

        self.max_weight=max_weight
        self.diagonal0 = diagonal0
        
        if pre_tau is Unconfigurable:
            self.pre_synapse = pre_synapse
        else:
            self.pre_tau = pre_tau

        if post_tau is Unconfigurable:
            self.post_synapse = (self.pre_synapse if post_synapse is Default
                                 else post_synapse)
        else:
            self.post_tau = post_tau

        if theta_tau is Unconfigurable:
            self.theta_synapse = theta_synapse
        else:
            self.theta_tau = theta_tau

    @property
    def _argdefaults(self):
        return (('learning_rate', BCM2.learning_rate.default),
                ('pre_synapse', BCM2.pre_synapse.default),
                ('post_synapse', self.pre_synapse),
                ('theta_synapse', BCM2.theta_synapse.default),
                ('max_weight', BCM2.max_weight.default),
                ('diagonal0', BCM2.diagonal0.default))



class SimBCM2(Operator):
    r"""Calculate connection weight change according to the BCM rule.
    Implements the Bienenstock-Cooper-Munroe learning rule of the form
    .. math:: \Delta \omega_{ij} = \kappa a_j (a_j - \theta_j) a_i
    where
    * :math:`\kappa` is a scalar learning rate,
    * :math:`a_j` is the activity of a postsynaptic neuron,
    * :math:`\theta_j` is an estimate of the average :math:`a_j`, and
    * :math:`a_i` is the activity of a presynaptic neuron.
    Parameters
    ----------
    pre_filtered : Signal
        The presynaptic activity, :math:`a_i`.
    post_filtered : Signal
        The postsynaptic activity, :math:`a_j`.
    theta : Signal
        The modification threshold, :math:`\theta_j`.
    delta : Signal
        The synaptic weight change to be applied, :math:`\Delta \omega_{ij}`.
    learning_rate : float
        The scalar learning rate, :math:`\kappa`.
    tag : str, optional (Default: None)
        A label associated with the operator, for debugging purposes.
    Attributes
    ----------
    delta : Signal
        The synaptic weight change to be applied, :math:`\Delta \omega_{ij}`.
    learning_rate : float
        The scalar learning rate, :math:`\kappa`.
    post_filtered : Signal
        The postsynaptic activity, :math:`a_j`.
    pre_filtered : Signal
        The presynaptic activity, :math:`a_i`.
    tag : str or None
        A label associated with the operator, for debugging purposes.
    theta : Signal
        The modification threshold, :math:`\theta_j`.
    Notes
    -----
    1. sets ``[]``
    2. incs ``[]``
    3. reads ``[pre_filtered, post_filtered, theta]``
    4. updates ``[delta]``
    """

    def __init__(self, pre_filtered, post_filtered, theta, delta, weights,
                 learning_rate, max_weight,diagonal0, tag=None):
        super().__init__(tag=tag)
        self.learning_rate = learning_rate
        self.max_weight = max_weight
        self.diagonal0 = diagonal0
        self.sets = []
        self.incs = []
        self.reads = [pre_filtered, post_filtered, theta, weights]
        self.updates = [delta]

    @property
    def delta(self):
        return self.updates[0]

    @property
    def pre_filtered(self):
        return self.reads[0]

    @property
    def post_filtered(self):
        return self.reads[1]

    @property
    def theta(self):
        return self.reads[2]

    @property
    def weights(self):
        return self.reads[3]
        
    def _descstr(self):
        return 'pre=%s, post=%s -> %s' % (
            self.pre_filtered, self.post_filtered, self.delta)

    def make_step(self, signals, dt, rng):
        pre_filtered = signals[self.pre_filtered]
        post_filtered = signals[self.post_filtered]
        theta = signals[self.theta]
        delta = signals[self.delta]
        weights = signals[self.weights]
        alpha = self.learning_rate * dt
        max_weight = self.max_weight
        diagonal0 = self.diagonal0

        #print(max_weights)
        if max_weight is not None:
            def step_simbcm():
                delta[...] = np.outer(
                    alpha * post_filtered * (post_filtered - theta), pre_filtered)
                delta[np.abs(weights+delta) > max_weight] = 0
                #print('here')
                if diagonal0:
                    #print('butnothere')
                    delta[np.eye(delta.shape[0])==1] = 0
                #delta = delta * (-np.eye(delta.shape)+1) #set diagonal to 0
                
        else: #None
            def step_simbcm():
                delta[...] = np.outer(
                    alpha * post_filtered * (post_filtered - theta), pre_filtered)
                #print(np.max(weights))
            
        return step_simbcm



def get_pre_ens(conn):
    return (conn.pre_obj if isinstance(conn.pre_obj, Ensemble)
            else conn.pre_obj.ensemble)


def get_post_ens(conn):
    return (conn.post_obj if isinstance(conn.post_obj, (Ensemble, Node))
            else conn.post_obj.ensemble)

def build_or_passthrough(model, obj, signal):
    """Builds the obj on signal, or returns the signal if obj is None."""
    return signal if obj is None else model.build(obj, signal)



@Builder.register(BCM2)
def build_bcm2(model, bcm, rule):
    """Builds a `.BCM` object into a model.
    Calls synapse build functions to filter the pre and post activities,
    and adds a `.SimBCM` operator to the model to calculate the delta.
    Parameters
    ----------
    model : Model
        The model to build into.
    bcm : BCM
        Learning rule type to build.
    rule : LearningRule
        The learning rule object corresponding to the neuron type.
    Notes
    -----
    Does not modify ``model.params[]`` and can therefore be called
    more than once with the same `.BCM` instance.
    """

    conn = rule.connection
    pre_activities = model.sig[get_pre_ens(conn).neurons]['out'][conn.pre_slice]
    post_activities = model.sig[get_post_ens(conn).neurons]['out'][conn.post_slice]
    pre_filtered = build_or_passthrough(model, bcm.pre_synapse, pre_activities)
    post_filtered = build_or_passthrough(model, bcm.post_synapse, post_activities)
    theta = build_or_passthrough(model, bcm.theta_synapse, post_activities)
   
    #weights = model.sig[get_post_ens(conn)]['encoders'][conn.post_slice]
    #weights = conn.solver.values
    
    model.add_op(SimBCM2(pre_filtered,
                        post_filtered,
                        theta,
                        model.sig[rule]['delta'],
                        model.sig[conn]['weights'],
                        learning_rate=bcm.learning_rate,
                        max_weight=bcm.max_weight,
                        diagonal0=bcm.diagonal0))

    # expose these for probes
    model.sig[rule]['theta'] = theta
    model.sig[rule]['pre_filtered'] = pre_filtered
    model.sig[rule]['post_filtered'] = post_filtered

## OCL


def plan_bcm2(queue, pre, post, theta, delta, alpha, tag=None): #weights, max_weight,
    assert len(pre) == len(post) == len(theta) == len(delta) == alpha.size
    N = len(pre)

    for arr in (pre, post, theta):  # vectors
        assert (arr.shape1s == 1).all()
    for arr in (delta,):  # matrices
        assert (arr.stride1s == 1).all()

    assert (post.shape0s == delta.shape0s).all()
    assert (pre.shape0s == delta.shape1s).all()
    assert (post.shape0s == theta.shape0s).all()

    assert (pre.ctype == post.ctype == theta.ctype == delta.ctype ==
            alpha.ctype)

    text = """
    __kernel void bcm2(
        __global const int *shape0s,
        __global const int *shape1s,
        __global const int *pre_stride0s,
        __global const int *pre_starts,
        __global const ${type} *pre_data,
        __global const int *post_stride0s,
        __global const int *post_starts,
        __global const ${type} *post_data,
        __global const int *theta_stride0s,
        __global const int *theta_starts,
        __global const ${type} *theta_data,
        __global const int *delta_stride0s,
        __global const int *delta_starts,
        __global ${type} *delta_data,
        __global const ${type} *alphas
        //__global const int *weights_stride0s,
        //__global const int *weights_starts,
        //__global const ${type} *weights_data,
        //__global const ${type} *max_weights
    )
    {
        const int ij = get_global_id(0);
        const int k = get_global_id(1);
        const int shape0 = shape0s[k];
        const int shape1 = shape1s[k];
        const int i = ij / shape1;
        const int j = ij % shape1;
        __global ${type} *delta = delta_data + delta_starts[k];
        const ${type} pre = pre_data[pre_starts[k] + j*pre_stride0s[k]];
        const ${type} post = post_data[post_starts[k] + i*post_stride0s[k]];
        const ${type} theta = theta_data[
            theta_starts[k] + i*theta_stride0s[k]];
        const ${type} alpha = alphas[k];
        
        //__global const ${type} *weights = weights_data + weights_starts[k];

        //const ${type} max_weight = max_weights[k];

        if (i < shape0) {
            delta[i*delta_stride0s[k] + j]  =
                alpha * post * (post - theta) * pre;

  
            //if (i==j) {
            //    delta[i*delta_stride0s[k] + j]  = 0;
            //} else {
            //    
            //    delta[i*delta_stride0s[k] + j]  = alpha * post * (post - theta) * pre;
            //
            //   if (fabs(weights[i*weights_stride0s[k] + j] + delta[i*delta_stride0s[k] + j]) > max_weight) { 
            //        delta[i*delta_stride0s[k] + j] = 0;
            //    }
            //}
        }
    }
    """

    textconf = dict(type=pre.ctype)
    text = as_ascii(Template(text, output_encoding='ascii').render(**textconf))

    full_args = (
        delta.cl_shape0s, delta.cl_shape1s,
        pre.cl_stride0s, pre.cl_starts, pre.cl_buf,
        post.cl_stride0s, post.cl_starts, post.cl_buf,
        theta.cl_stride0s, theta.cl_starts, theta.cl_buf,
        delta.cl_stride0s, delta.cl_starts, delta.cl_buf,
        alpha,
    )
        #weights.cl_stride0s, weights.cl_starts, weights.cl_buf, #max_weight,
    
    _fn = cl.Program(queue.context, text).build().bcm2
    _fn.set_args(*[arr.data for arr in full_args])

    lsize = None
    gsize = (delta.sizes.max(), N)
    plan = Plan(queue, _fn, gsize, lsize=lsize, name="cl_bcm2", tag=tag)
    plan.full_args = full_args     # prevent garbage-collection
    plan.flops_per_call = 4 * delta.sizes.sum()
    plan.bw_per_call = (pre.nbytes + post.nbytes + theta.nbytes +
                        delta.nbytes + alpha.nbytes) # + weights.nbytes + max_weight.nbytes)
    return plan
    


def plan_bcm2_threshold(queue, delta, weights, max_weight, tag=None):
    N = len(delta)

    for arr in (delta,):  # matrices
        assert (arr.stride1s == 1).all()

    text = """
    __kernel void bcm2_threshold(
        __global const int *shape0s,
        __global const int *shape1s,
        __global const int *delta_stride0s,
        __global const int *delta_starts,
        __global ${type} *delta_data,
        __global const int *weights_stride0s,
        __global const int *weights_starts,
        __global const ${type} *weights_data,
        __global const ${type} *max_weights
    )
    {
        const int ij = get_global_id(0);
        const int k = get_global_id(1);
        const int shape0 = shape0s[k];
        const int shape1 = shape1s[k];
        const int i = ij / shape1;
        const int j = ij % shape1;
        __global ${type} *delta = delta_data + delta_starts[k];
        __global const ${type} *weights = weights_data + weights_starts[k];
        const ${type} max_weight = max_weights[k];

        if (i < shape0) {
        
            if (i==j) {
                delta[i*delta_stride0s[k] + j]  = 0;
            } else {
            
               if (fabs(weights[i*weights_stride0s[k] + j] + delta[i*delta_stride0s[k] + j]) > max_weight) { 
                    delta[i*delta_stride0s[k] + j] = 0;
                }
            }
        }
    }
    """

    textconf = dict(type=delta.ctype)
    text = as_ascii(Template(text, output_encoding='ascii').render(**textconf))

    full_args = (
        delta.cl_shape0s, delta.cl_shape1s,
        delta.cl_stride0s, delta.cl_starts, delta.cl_buf,
        weights.cl_stride0s, weights.cl_starts, weights.cl_buf,
        max_weight,
    )
    _fn = cl.Program(queue.context, text).build().bcm2_threshold
    _fn.set_args(*[arr.data for arr in full_args])

    lsize = None
    gsize = (delta.sizes.max(), N)
    plan = Plan(queue, _fn, gsize, lsize=lsize, name="cl_bcm2_threshold", tag=tag)
    plan.full_args = full_args     # prevent garbage-collection
    plan.flops_per_call = 4 * delta.sizes.sum()
    plan.bw_per_call = (delta.nbytes + weights.nbytes + max_weight.nbytes)
    return plan
    
def plan_bcm2_threshold_diagonal1(queue, delta, weights, max_weight, tag=None):
    N = len(delta)

    for arr in (delta,):  # matrices
        assert (arr.stride1s == 1).all()

    text = """
    __kernel void bcm2_threshold_diagonal1(
        __global const int *shape0s,
        __global const int *shape1s,
        __global const int *delta_stride0s,
        __global const int *delta_starts,
        __global ${type} *delta_data,
        __global const int *weights_stride0s,
        __global const int *weights_starts,
        __global const ${type} *weights_data,
        __global const ${type} *max_weights
    )
    {
        const int ij = get_global_id(0);
        const int k = get_global_id(1);
        const int shape0 = shape0s[k];
        const int shape1 = shape1s[k];
        const int i = ij / shape1;
        const int j = ij % shape1;
        __global ${type} *delta = delta_data + delta_starts[k];
        __global const ${type} *weights = weights_data + weights_starts[k];
        const ${type} max_weight = max_weights[k];

        if (i < shape0) {            
           if (fabs(weights[i*weights_stride0s[k] + j] + delta[i*delta_stride0s[k] + j]) > max_weight) { 
                delta[i*delta_stride0s[k] + j] = 0;
            }
        }
    }
    """

    textconf = dict(type=delta.ctype)
    text = as_ascii(Template(text, output_encoding='ascii').render(**textconf))

    full_args = (
        delta.cl_shape0s, delta.cl_shape1s,
        delta.cl_stride0s, delta.cl_starts, delta.cl_buf,
        weights.cl_stride0s, weights.cl_starts, weights.cl_buf,
        max_weight,
    )
    _fn = cl.Program(queue.context, text).build().bcm2_threshold_diagonal1
    _fn.set_args(*[arr.data for arr in full_args])

    lsize = None
    gsize = (delta.sizes.max(), N)
    plan = Plan(queue, _fn, gsize, lsize=lsize, name="cl_bcm2_threshold_diagonal1", tag=tag)
    plan.full_args = full_args     # prevent garbage-collection
    plan.flops_per_call = 4 * delta.sizes.sum()
    plan.bw_per_call = (delta.nbytes + weights.nbytes + max_weight.nbytes)
    return plan 
    
# class MyOCLsimulator(Simulator):
#   def plan_SimBCM2(self, ops):
#         pre = self.all_data[[self.sidx[op.pre_filtered] for op in ops]]
#         post = self.all_data[[self.sidx[op.post_filtered] for op in ops]]
#         theta = self.all_data[[self.sidx[op.theta] for op in ops]]
#         delta = self.all_data[[self.sidx[op.delta] for op in ops]]
#         alpha = self.Array([op.learning_rate * self.model.dt for op in ops])
#         # new bcm2
#         weights = self.all_data[[self.sidx[op.weights] for op in ops]]
#         max_weight = self.Array([op.max_weight for op in ops])
# 
#         #print(max_weights)
#         #max_weights = self.all_data[[self.sidx[op.max_weights] for op in ops]]
# 
#         #return [plan_bcm2(self.queue, pre, post, theta, delta, alpha, weights, max_weight)]
#         if ops[0].diagonal0:
#             return [plan_bcm2(self.queue, pre, post, theta, delta, alpha),
#                 plan_bcm2_threshold(self.queue, delta, weights, max_weight)]
#         else:
#             return [plan_bcm2(self.queue, pre, post, theta, delta, alpha),
#                 plan_bcm2_threshold_diagonal1(self.queue, delta, weights, max_weight)]
#         

   
