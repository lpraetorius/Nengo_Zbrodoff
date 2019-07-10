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

from nengo_ocl import Simulator
from nengo_ocl.utils import as_ascii #, indent, round_up
from mako.template import Template
import pyopencl as cl
from nengo_ocl.plan import Plan



import voja2_rule
import bcm2_rule
from voja2_rule import plan_voja2, plan_voja2_prep_norm, plan_voja2_apply_norm
from bcm2_rule import plan_bcm2, plan_bcm2_threshold

class MyOCLsimulator(Simulator):
    def plan_SimBCM2(self, ops):
        pre = self.all_data[[self.sidx[op.pre_filtered] for op in ops]]
        post = self.all_data[[self.sidx[op.post_filtered] for op in ops]]
        theta = self.all_data[[self.sidx[op.theta] for op in ops]]
        delta = self.all_data[[self.sidx[op.delta] for op in ops]]
        alpha = self.Array([op.learning_rate * self.model.dt for op in ops])
        # new bcm2
        weights = self.all_data[[self.sidx[op.weights] for op in ops]]
        max_weight = self.Array([op.max_weight for op in ops])
        #print(max_weights)
        #max_weights = self.all_data[[self.sidx[op.max_weights] for op in ops]]

        #return [plan_bcm2(self.queue, pre, post, theta, delta, alpha, weights, max_weight)]
        return [plan_bcm2(self.queue, pre, post, theta, delta, alpha),
                plan_bcm2_threshold(self.queue, delta, weights, max_weight)]
   
   
    def plan_SimVoja2(self, ops):
        pre = self.all_data[[self.sidx[op.pre_decoded] for op in ops]]
        post = self.all_data[[self.sidx[op.post_filtered] for op in ops]]
        encoders = self.all_data[[self.sidx[op.scaled_encoders] for op in ops]]

        new_encoders = self.RaggedArray([np.zeros(op.scaled_encoders.shape) for op in ops], dtype=np.float32)

        delta = self.all_data[[self.sidx[op.delta] for op in ops]]
        learning_signal = self.all_data[[self.sidx[op.learning_signal] for op in ops]]
        scale = self.RaggedArray([op.scale for op in ops], dtype=np.float32)
        alpha = self.Array([op.learning_rate * self.model.dt for op in ops])
        threshold = self.Array([op.threshold for op in ops]) 
        mag = self.RaggedArray([np.zeros(op.scale.shape) for op in ops], dtype=np.float32) #correct? (1,1000)

        return [
            plan_voja2(self.queue, pre, post, encoders, new_encoders, learning_signal, scale, alpha, threshold),
            plan_voja2_prep_norm(self.queue, new_encoders, mag),
            plan_voja2_apply_norm(self.queue, encoders, new_encoders, delta, mag, scale)]
