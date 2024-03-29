from __future__ import print_function

import inspect
import logging
import os
import warnings
from collections import defaultdict

import numpy as np
import pyopencl as cl

import nengo
import nengo.version
import nengo.utils.numpy as npext
from nengo.cache import get_default_decoder_cache
from nengo.exceptions import ReadonlyError, SimulatorClosed, ValidationError
from nengo.simulator import ProbeDict
from nengo.builder.builder import Model
from nengo.builder.operator import Reset
from nengo.builder.signal import SignalDict
from nengo.utils.compat import iteritems, StringIO, range, ResourceWarning
from nengo.utils.progress import ProgressTracker, Progress
from nengo.utils.stdlib import groupby

from nengo_ocl.raggedarray import RaggedArray
from nengo_ocl.clraggedarray import CLRaggedArray, to_device
from nengo_ocl.clra_gemv import plan_block_gemv
from nengo_ocl.clra_nonlinearities import (
    plan_timeupdate, plan_reset, plan_copy, plan_slicedcopy,
    plan_direct, plan_lif, plan_lif_rate, plan_rectified_linear,
    plan_spiking_rectified_linear, plan_sigmoid,
    plan_probes, plan_linearfilter, plan_elementwise_inc,
    create_rngs, init_rngs, get_dist_enums_params, plan_whitenoise,
    plan_presentinput, plan_conv2d, plan_pool2d, plan_bcm, plan_oja, plan_voja)
from nengo_ocl.operators import MultiDotInc
from nengo_ocl.plan import BasePlan, PythonPlan, Plans
from nengo_ocl.planners import greedy_planner
from nengo_ocl.ast_conversion import OCL_Function
from nengo_ocl.utils import get_closures, indent, split, stable_unique, Timer
from nengo_ocl.version import (
    bad_nengo_versions, latest_nengo_version, latest_nengo_version_info)

import voja2_rule
import bcm2_rule
from voja2_rule import plan_voja2, plan_voja2_prep_norm, plan_voja2_apply_norm
from bcm2_rule import plan_bcm2, plan_bcm2_threshold



logger = logging.getLogger(__name__)
PROFILING_ENABLE = cl.command_queue_properties.PROFILING_ENABLE


class ViewBuilder(object):

    def __init__(self, bases, rarray):
        self.sidx = {bb: ii for ii, bb in enumerate(bases)}
        assert len(bases) == len(self.sidx)
        self.rarray = rarray

        self.starts = []
        self.shape0s = []
        self.shape1s = []
        self.stride0s = []
        self.stride1s = []
        self.names = []

        self._A_views = {}
        self._X_views = {}
        self._YYB_views = {}

    def append_view(self, obj):
        if obj in self.sidx:
            return  # we already have this view

        if not obj.is_view:
            # -- it is not a view, and not OK
            raise ValueError('can only append views of known signals', obj)

        assert obj.size and obj.ndim <= 2
        idx = self.sidx[obj.base]
        shape0 = obj.shape[0] if obj.ndim > 0 else 1
        shape1 = obj.shape[1] if obj.ndim > 1 else 1
        self.starts.append(self.rarray.starts[idx] + obj.elemoffset)
        self.shape0s.append(shape0)
        self.shape1s.append(shape1)
        self.stride0s.append(obj.elemstrides[0] if shape0 > 1 else 1)
        self.stride1s.append(obj.elemstrides[1] if shape1 > 1 else 1)
        self.names.append(getattr(obj, 'name', ''))
        self.sidx[obj] = len(self.sidx)

    def add_views_to(self, rarray):
        rarray.add_views(self.starts, self.shape0s, self.shape1s,
                         self.stride0s, self.stride1s, names=self.names)

    def setup_views(self, ops):
        all_views = [sig for op in ops for sig in op.all_signals]
        for op in (op for op in ops if isinstance(op, MultiDotInc)):
            A_views, X_views, Y_view, Y_in_view, beta_view = op.get_views()
            all_views.extend(A_views + X_views + [Y_view, Y_in_view] +
                             ([beta_view] if beta_view else []))
            self._A_views[op] = A_views
            self._X_views[op] = X_views
            self._YYB_views[op] = [Y_view, Y_in_view, beta_view]

        for view in all_views:
            self.append_view(view)


class Simulator(object):
    """Simulator for running Nengo models in OpenCL.

    Parameters
    ----------
    network, dt, seed, model
        These parameters are the same as in `nengo.Simulator`.
    context : `pyopencl.Context` (optional)
        OpenCL context specifying which device(s) to run on. By default, we
        will create a context by calling `pyopencl.create_some_context`
        and use this context as the default for all subsequent instances.
    n_prealloc_probes : int (optional)
        Number of timesteps to buffer when probing. Larger numbers mean less
        data transfer with the device (faster), but use more device memory.
    profiling : boolean (optional)
        If ``True``, ``print_profiling()`` will show profiling information.
        By default, will check the environment variable ``NENGO_OCL_PROFILING``
    if_python_code : 'none' | 'warn' | 'error'
        How the simulator should react if a Python function cannot be converted
        to OpenCL code.
    planner : callable
        A function to plan operator order. See ``nengo_ocl.planners``.
    """

    # --- Store the result of create_some_context so we don't recreate it
    some_context = None

    # 'unsupported' defines features unsupported by a simulator.
    # The format is a list of tuples of the form `(test, reason)` with `test`
    # being a string with wildcards (*, ?, [abc], [!abc]) matched against Nengo
    # test paths and names, and `reason` is a string describing why the feature
    # is not supported by the backend. For example:
    #     unsupported = [('test_pes*', 'PES rule not implemented')]
    # would skip all test whose names start with 'test_pes'.
    unsupported = [
        # advanced indexing
        ('nengo/tests/test_connection.py:test_list_indexing*',
         "Advanced indexing with repeated indices not implemented"),

        # neuron types
        ('nengo/tests/test_neurons.py:test_izhikevich',
         "Izhikevich neurons not implemented"),
        ('nengo/tests/test_neurons.py:test_lif_min_voltage*',
         "Min voltage not implemented"),

        # nodes
        ('nengo/tests/test_node.py:test_none',
         "No error if nodes output None"),
        ('nengo/tests/test_node.py:test_invalid_values*',
         "No error for invalid node values"),
        ('nengo/tests/test_neurons.py:test_direct_mode_nonfinite_value',
         "No error for non-finite values"),

        # processes
        ('nengo/tests/test_processes.py:test_brownnoise',
         "Filtered noise processes not yet implemented"),
        ('nengo/tests/test_ensemble.py:test_noise_copies_ok*',
         "Filtered noise processes not yet implemented"),
        ('nengo/tests/test_simulator.py:test_noise_copies_ok',
         "Filtered noise processes not yet implemented"),
        ('nengo/tests/test_processes.py:TestPiecewise.test_interpolation_?d',
         "float32 rounding issues"),

        # synapses
        ('nengo/tests/test_synapses.py:test_triangle',
         "Only linear filters implemented"),

        # learning rules
        ('nengo/tests/test_learning_rules.py:test_custom_type',
         "Copying 2-D arrays not implemented"),

        # simulator features
        ('nengo/tests/test_simulator.py:test_probe_cache',
         "Changing simulator seed not implemented"),

        # specific to nengo.Simulator (functionality does not need testing)
        ('nengo/tests/test_builder.py:test_commonsig_readonly',
         "Specific to nengo.Simulator"),
    ]

    def Array(self, val, dtype=np.float32):
        return to_device(self.queue, np.asarray(val, dtype=dtype))

    def RaggedArray(self, listofarrays, **kwargs):
        return CLRaggedArray.from_arrays(self.queue, listofarrays, **kwargs)

    def __init__(self, network, dt=0.001, seed=None, model=None, context=None,
                 n_prealloc_probes=32, profiling=None, if_python_code='none',
                 planner=greedy_planner, progress_bar=True):
        # --- check version
        if nengo.version.version_info in bad_nengo_versions:
            raise ValueError(
                "This simulator does not support Nengo version %s. Upgrade "
                "with 'pip install --upgrade --no-deps nengo'."
                % nengo.__version__)
        elif nengo.version.version_info > latest_nengo_version_info:
            warnings.warn("This version of `nengo_ocl` has not been tested "
                          "with your `nengo` version (%s). The latest fully "
                          "supported version is %s" % (
                              nengo.__version__, latest_nengo_version))

        # --- create these first since they are used in __del__
        self.closed = False
        self.model = None

        # --- arguments/attributes
        if context is None and Simulator.some_context is None:
            print('No context argument was provided to nengo_ocl.Simulator')
            print("Calling pyopencl.create_some_context() for you now:")
            Simulator.some_context = cl.create_some_context()
        if profiling is None:
            profiling = int(os.getenv("NENGO_OCL_PROFILING", 0))
        self.context = Simulator.some_context if context is None else context
        self.profiling = profiling
        self.queue = cl.CommandQueue(
            self.context, properties=PROFILING_ENABLE if self.profiling else 0)

        if if_python_code not in ['none', 'warn', 'error']:
            raise ValueError("%r not a valid value for `if_python_code`"
                             % if_python_code)
        self.if_python_code = if_python_code
        self.n_prealloc_probes = n_prealloc_probes
        self.progress_bar = progress_bar

        # --- Nengo build
        with Timer() as nengo_timer:
            if model is None:
                self.model = Model(dt=float(dt),
                                   label="%s, dt=%f" % (network, dt),
                                   decoder_cache=get_default_decoder_cache())
            else:
                self.model = model

            if network is not None:
                # Build the network into the model
                self.model.build(network)

        logger.info("Nengo build in %0.3f s" % nengo_timer.duration)

        # --- operators
        with Timer() as planner_timer:
            operators = list(self.model.operators)

            # convert DotInc and Copy to MultiDotInc
            operators = list(map(MultiDotInc.convert_to, operators))
            operators = MultiDotInc.compress(operators)

            # plan the order of operations, combining where appropriate
            op_groups = planner(operators)
            assert len([typ for typ, _ in op_groups if typ is Reset]) < 2, (
                "All resets not planned together")

            self.operators = operators
            self.op_groups = op_groups

        logger.info("Planning in %0.3f s" % planner_timer.duration)

        with Timer() as signals_timer:
            # Initialize signals
            all_signals = stable_unique(
                sig for op in operators for sig in op.all_signals)
            all_bases = stable_unique(sig.base for sig in all_signals)

            sigdict = SignalDict()  # map from Signal.base -> ndarray
            for op in operators:
                op.init_signals(sigdict)

            # Add built states to the probe dictionary
            self._probe_outputs = dict(self.model.params)

            # Provide a nicer interface to probe outputs
            self.data = ProbeDict(self._probe_outputs)

            # Create data on host and add views
            self.all_data = RaggedArray(
                [sigdict[sb] for sb in all_bases],
                names=[getattr(sb, 'name', '') for sb in all_bases],
                dtype=np.float32)

            view_builder = ViewBuilder(all_bases, self.all_data)
            view_builder.setup_views(operators)
            for probe in self.model.probes:
                view_builder.append_view(self.model.sig[probe]['in'])
            view_builder.add_views_to(self.all_data)

            self.all_bases = all_bases
            self.sidx = {
                k: np.int32(v) for k, v in iteritems(view_builder.sidx)}
            self._A_views = view_builder._A_views
            self._X_views = view_builder._X_views
            self._YYB_views = view_builder._YYB_views
            del view_builder

            # Copy data to device
            self.all_data = CLRaggedArray(self.queue, self.all_data)

        logger.info("Signals in %0.3f s" % signals_timer.duration)

        # --- set seed
        self.seed = np.random.randint(npext.maxint) if seed is None else seed
        self.rng = np.random.RandomState(self.seed)

        # --- create list of plans
        self._raggedarrays_to_reset = {}
        self._cl_rngs = {}
        self._python_rngs = {}

        plans = []
        
        
        
        with Timer() as plans_timer:
            for op_type, op_list in op_groups:
                plans.extend(self.plan_op_group(op_type, op_list))
            plans.extend(self.plan_probes())

        logger.info("Plans in %0.3f s" % plans_timer.duration)

        # -- create object to execute list of plans
        self._plans = Plans(plans, self.profiling)

        self.rng = None  # all randomness set, should no longer be used
        self._reset_probes()  # clears probes from previous model builds

    def _create_cl_rngs(self, seeds):
        seeds = [self.rng.randint(npext.maxint) if s is None else s
                 for s in seeds]
        cl_rngs = create_rngs(self.queue, len(seeds))
        init_rngs(self.queue, cl_rngs, seeds)
        self._cl_rngs[cl_rngs] = seeds
        return cl_rngs

    def _reset_rngs(self):
        for rngs, seeds in iteritems(self._cl_rngs):
            init_rngs(self.queue, rngs, seeds)

        for rng, state in iteritems(self._python_rngs):
            rng.set_state(state)

    def __del__(self):
        """Raise a ResourceWarning if we are deallocated while open."""
        if not self.closed:
            warnings.warn(
                "Simulator with model=%s was deallocated while open. Please "
                "close simulators manually to ensure resources are properly "
                "freed." % self.model, ResourceWarning)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __getitem__(self, item):
        """
        Return internally shaped signals, which are always 2d
        """
        return self.all_data[self.sidx[item]]

    @property
    def dt(self):
        """(float) The time step of the simulator."""
        return self.model.dt

    @dt.setter
    def dt(self, dummy):
        raise ReadonlyError(attr='dt', obj=self)

    @property
    def n_steps(self):
        """(int) The current time step of the simulator."""
        return self._n_steps

    @property
    def time(self):
        """(float) The current time of the simulator."""
        return self._time

    @property
    def signals(self):
        """Get/set [properly-shaped] signal value (either 0d, 1d, or 2d)
        """
        class Accessor(object):

            def __iter__(_):
                return iter(self.all_bases)

            def __getitem__(_, item):
                raw = self.all_data[self.sidx[item]]
                if item.ndim == 0:
                    return raw[0, 0]
                elif item.ndim == 1:
                    return raw.ravel()
                elif item.ndim == 2:
                    return raw
                else:
                    raise NotImplementedError()

            def __setitem__(_, item, val):
                incoming = np.asarray(val)
                if item.ndim == 0:
                    assert incoming.size == 1
                    self.all_data[self.sidx[item]] = incoming
                elif item.ndim == 1:
                    assert (item.size,) == incoming.shape
                    self.all_data[self.sidx[item]] = incoming[:, None]
                elif item.ndim == 2:
                    assert item.shape == incoming.shape
                    self.all_data[self.sidx[item]] = incoming
                else:
                    raise NotImplementedError()

            def __str__(self_):
                sio = StringIO()
                for k in self_:
                    print(k, self_[k], file=sio)
                return sio.getvalue()

        return Accessor()

    # --- Simulation functions (see ``nengo.Simulator`` for interface)
    def close(self):
        self.closed = True
        self.context = None
        self.queue = None
        self.all_data = None
        self._plans = None
        self._raggedarrays_to_reset = None
        self._cl_rngs = None
        self._cl_probe_plan = None

    def _probe(self):
        """Copy all probed signals to buffers"""
        self._probe_step_time()

        plan = self._cl_probe_plan
        if plan is None:
            return  # nothing to probe

        self.queue.finish()
        bufpositions = plan.cl_bufpositions.get()
        for i, probe in enumerate(self.model.probes):
            shape = self.model.sig[probe]['in'].shape
            n_buffered = bufpositions[i]
            if n_buffered:
                # XXX: this syntax retrieves *ALL* of Y from the device
                #      because the :n_buffered only works on the ndarray
                #      *after* it has been transferred.
                raw = plan.Y[i][:n_buffered]
                shaped = raw.reshape((n_buffered,) + shape)
                self._probe_outputs[probe].extend(shaped)
        plan.cl_bufpositions.fill(0)
        self.queue.finish()

    def _probe_step_time(self):
        self._n_steps = self.signals[self.model.step].item()
        self._time = self.signals[self.model.time].item()

    def _reset_probes(self):
        if self._cl_probe_plan is not None:
            self._cl_probe_plan.cl_bufpositions.fill(0)

        for probe in self.model.probes:
            self._probe_outputs[probe] = []
        self.data.reset()

        self._probe_step_time()

    def reset(self, seed=None):
        if self.closed:
            raise SimulatorClosed("Cannot reset closed Simulator.")

        if seed is not None:
            raise NotImplementedError("Seed changing not implemented")

        # reset signals
        for base in self.all_bases:
            # TODO: copy all data on at once
            if not base.readonly:
                self.all_data[self.sidx[base]] = base.initial_value

        for clra, ra in iteritems(self._raggedarrays_to_reset):
            # TODO: copy all data on at once
            for i in range(len(clra)):
                clra[i] = ra[i]

        self._reset_rngs()
        self._reset_probes()

    def run(self, time_in_seconds, progress_bar=None):
        """Simulate for the given length of time.

        If the given length of time is not a multiple of ``dt``,
        it will be rounded to the nearest ``dt``. For example, if ``dt``
        is 0.001 and ``run`` is called with ``time_in_seconds=0.0006``,
        the simulator will advance one timestep, resulting in the actual
        simulator time being 0.001.

        The given length of time must be positive. The simulator cannot
        be run backwards.

        Parameters
        ----------
        time_in_seconds : float
            Amount of time to run the simulation for. Must be positive.
        progress_bar : bool or `.ProgressBar` or `.ProgressUpdater`, optional \
                       (Default: True)
            Progress bar for displaying the progress of the simulation run.

            If True, the default progress bar will be used.
            If False, the progress bar will be disabled.
            For more control over the progress bar, pass in a `.ProgressBar`
            or `.ProgressUpdater` instance.
        """
        if time_in_seconds < 0:
            raise ValidationError("Must be positive (got %g)"
                                  % (time_in_seconds,), attr="time_in_seconds")

        steps = int(np.round(float(time_in_seconds) / self.dt))

        if steps == 0:
            warnings.warn("%g results in running for 0 timesteps. Simulator "
                          "still at time %g." % (time_in_seconds, self.time))
        else:
            logger.info("Running %s for %f seconds, or %d steps",
                        self.model.label, time_in_seconds, steps)
            self.run_steps(steps, progress_bar=progress_bar)

    def run_steps(self, N, progress_bar=True):
        if self.closed:
            raise SimulatorClosed("Simulator cannot run because it is closed.")

        if self.n_steps + N >= 2**24:
            # since n_steps is float32, point at which `n_steps == n_steps + 1`
            raise ValueError("Cannot handle more than 2**24 steps")

        if self._cl_probe_plan is not None:
            # -- precondition: the probe buffers have been drained
            bufpositions = self._cl_probe_plan.cl_bufpositions.get()
            assert np.all(bufpositions == 0)

        if progress_bar is None:
            progress_bar = self.progress_bar
        try:
            progress = ProgressTracker(progress_bar, Progress(
                "Simulating", "Simulation", N))
        except TypeError:
            try:
                progress = ProgressTracker(N, progress_bar, "Simulating")
            except TypeError:
                progress = ProgressTracker(N, progress_bar)

        with progress:
            # -- we will go through N steps of the simulator
            #    in groups of up to B at a time, draining
            #    the probe buffers after each group of B
            while N:
                B = min(N, self._max_steps_between_probes)
                self._plans.call_n_times(B)
                self._probe()
                N -= B
                if hasattr(progress, 'total_progress'):
                    progress.total_progress.step(n=B)
                else:
                    progress.step(n=B)

        if self.profiling > 1:
            self.print_profiling()

    def step(self):
        return self.run_steps(1, progress_bar=False)

    def trange(self, dt=None):
        """Create a vector of times matching probed data.

        Note that the range does not start at 0 as one might expect, but at
        the first timestep (i.e., ``dt``).

        Parameters
        ----------
        dt : float, optional (Default: None)
            The sampling period of the probe to create a range for.
            If None, the simulator's ``dt`` will be used.
        """
        dt = self.dt if dt is None else dt
        n_steps = int(self.n_steps * (self.dt / dt))
        return dt * np.arange(1, n_steps + 1)

    # --- Planning
    def plan_probes(self):
        if len(self.model.probes) == 0:
            self._max_steps_between_probes = self.n_prealloc_probes
            self._cl_probe_plan = None
            return []
        else:
            n_prealloc = self.n_prealloc_probes

            probes = self.model.probes
            periods = [1 if p.sample_every is None else
                       p.sample_every / self.dt
                       for p in probes]

            X = self.all_data[
                [self.sidx[self.model.sig[p]['in']] for p in probes]]
            Y = self.RaggedArray(
                [np.zeros((n_prealloc, self.model.sig[p]['in'].size))
                 for p in probes], dtype=np.float32)

            cl_plan = plan_probes(self.queue, periods, X, Y)
            self._max_steps_between_probes = n_prealloc * int(min(periods))
            self._cl_probe_plan = cl_plan
            return [cl_plan]

    def plan_op_group(self, op_type, ops):
        return getattr(self, 'plan_' + op_type.__name__)(ops)

    def plan_PreserveValue(self, ops):  # LEGACY
        # This op was removed in Nengo version 2.3.1+, but remains here
        # for compatibility with older versions of Nengo.
        return []  # do nothing

    def plan_MultiDotInc(self, ops):
        constant_bs = [op for op in ops if op._float_beta is not None]
        vector_bs = [op for op in ops
                     if op._signal_beta is not None
                     and op._signal_beta.ndim == 1]
        if len(constant_bs) + len(vector_bs) != len(ops):
            raise NotImplementedError()

        args = (
            lambda op: self._A_views[op],
            lambda op: self._X_views[op],
            lambda op: self._YYB_views[op][0],
            lambda op: self._YYB_views[op][1],
            )
        constant_b_gemvs = self._sig_gemv(
            constant_bs, *args,
            beta=[op._float_beta for op in constant_bs],
            gamma=[op.gamma for op in constant_bs],
            tag='c-beta-%d' % len(constant_bs))
        vector_b_gemvs = self._sig_gemv(
            vector_bs, *args,
            beta=lambda op: self._YYB_views[op][2],
            gamma=[op.gamma for op in vector_bs],
            tag='v-beta-%d' % len(vector_bs))
        return constant_b_gemvs + vector_b_gemvs

    def _sig_gemv(self, ops, A_js_fn, X_js_fn, Y_fn, Y_in_fn=None,
                  alpha=1.0, beta=1.0, gamma=0.0, tag=None):
        if len(ops) == 0:
            return []

        all_data, sidx = self.all_data, self.sidx
        A_js = RaggedArray([[sidx[ss] for ss in A_js_fn(op)] for op in ops])
        X_js = RaggedArray([[sidx[ss] for ss in X_js_fn(op)] for op in ops])
        Y_sigs = [Y_fn(item) for item in ops]
        Y_in_sigs = [Y_in_fn(item) for item in ops] if Y_in_fn else Y_sigs
        Y = all_data[[sidx[sig] for sig in Y_sigs]]
        Y_in = all_data[[sidx[sig] for sig in Y_in_sigs]]
        if callable(beta):
            beta = RaggedArray([sidx[beta(o)] for o in ops], dtype=np.float32)

        rval = plan_block_gemv(
            self.queue, alpha, all_data, A_js, all_data, X_js, beta, Y,
            Y_in=Y_in, gamma=gamma, tag=tag)
        return rval.plans

    def plan_TimeUpdate(self, ops):
        op, = ops
        step = self.all_data[[self.sidx[op.step]]]
        time = self.all_data[[self.sidx[op.time]]]
        return [plan_timeupdate(self.queue, step, time, self.model.dt)]

    def plan_Reset(self, ops):
        targets = self.all_data[[self.sidx[op.dst] for op in ops]]
        values = self.Array([op.value for op in ops])
        return [plan_reset(self.queue, targets, values)]

    def plan_SlicedCopy(self, ops):  # LEGACY
        # This op was removed in Nengo version 2.3.1+, but remains here
        # for compatibility with older versions of Nengo.
        return self.plan_Copy(ops, legacy=True)

    def plan_Copy(self, ops, legacy=False):
        noslice = Ellipsis if legacy else None  # LEGACY
        copies, ops = split(ops, lambda op: (
            op.src_slice is noslice and op.dst_slice is noslice))

        plans = []
        if copies:
            X = self.all_data[[self.sidx[op.src] for op in copies]]
            Y = self.all_data[[self.sidx[op.dst] for op in copies]]
            incs = np.array([op.inc for op in copies], dtype=np.int32)
            plans.append(plan_copy(self.queue, X, Y, incs))

        if ops:
            inds = lambda ary, i: np.arange(ary.size, dtype=np.int32)[
                Ellipsis if i is None else i]
            xinds = [inds(op.src, op.src_slice) for op in ops]
            yinds = [inds(op.dst, op.dst_slice) for op in ops]

            dupl = lambda s: (
                s is not None
                and not (isinstance(s, np.ndarray) and s.dtype == np.bool)
                and len(s) != len(set(s)))
            if any(dupl(i) for i in xinds) or any(dupl(i) for i in yinds):
                raise NotImplementedError("Duplicates in indices")

            X = self.all_data[[self.sidx[op.src] for op in ops]]
            Y = self.all_data[[self.sidx[op.dst] for op in ops]]
            Xinds = self.RaggedArray(xinds)
            Yinds = self.RaggedArray(yinds)
            incs = np.array([op.inc for op in ops], dtype=np.int32)
            plans.append(plan_slicedcopy(self.queue, X, Y, Xinds, Yinds, incs))

        return plans

    def plan_ElementwiseInc(self, ops):
        A = self.all_data[[self.sidx[op.A] for op in ops]]
        X = self.all_data[[self.sidx[op.X] for op in ops]]
        Y = self.all_data[[self.sidx[op.Y] for op in ops]]
        return [plan_elementwise_inc(self.queue, A, X, Y)]

    def plan_SimPyFunc(self, ops):
        groups = groupby(ops, lambda op: op.fn)
        # ^ NOTE: Groups functions based on equality `==`, not identity `is`.
        #   I think this is what we want in all cases.
        plans = []
        for fn, group in groups:
            plans.extend(self._plan_python_fn(
                fn, ts=[op.t for op in group], xs=[op.x for op in group],
                ys=[op.output for op in group]))
        return plans

    def _plan_python_fn(self, fn, ts, xs, ys):
        assert len(ts) == len(xs) == len(ys)
        assert all(t is None for t in ts) or all(t is not None for t in ts)
        assert all(x is None for x in xs) or all(x is not None for x in xs)
        assert all(y is None for y in ys) or all(y is not None for y in ys)
        if ts[0] is not None:
            assert all(t is self.model.time for t in ts)

        signal_size = lambda sig: sig.size if sig is not None else None

        fn_name = fn.__name__ if inspect.isfunction(fn) else type(fn).__name__

        # group by number of x dims
        signals = zip(ts, xs, ys)
        groups = groupby(signals, lambda s: signal_size(s[1]))

        # --- try to turn Python function into OCL code
        plans = []
        unplanned_signals = []
        for x_dim, group in groups:
            tt, xx, yy = zip(*group)

            # if any functions have no output, must do them in Python
            y_dim = signal_size(yy[0])
            if y_dim is None:
                self._found_python_code(
                    "Function %r could not be converted to OCL "
                    "since it has no outputs." % (fn_name))
                unplanned_signals.extend(zip(tt, xx, yy))
                continue

            # try to get OCL code
            if self.if_python_code == 'error':
                plans.append(self._plan_fn_in_ocl(fn, tt, xx, yy, fn_name))
            else:
                try:
                    plans.append(self._plan_fn_in_ocl(fn, tt, xx, yy, fn_name))
                except Exception as e:
                    self._found_python_code(
                        "Function %r could not be converted to OCL due to %s%s"
                        % (fn_name, type(e).__name__, e.args))
                    unplanned_signals.extend(zip(tt, xx, yy))

        # --- do remaining unplanned signals in Python
        if len(unplanned_signals) > 0:
            tt, xx, yy = zip(*unplanned_signals)
            plans.append(self._plan_fn_in_python(fn, tt, xx, yy, fn_name))

        return plans

    def _found_python_code(self, message):
        if self.if_python_code == 'error':
            raise RuntimeError(message)
        elif self.if_python_code == 'warn':
            warnings.warn(message, RuntimeWarning)

    def _plan_fn_in_ocl(self, fn, tt, xx, yy, fn_name):
        signal_size = lambda sig: sig.size if sig is not None else None
        vector_dims = lambda shape, dim: len(shape) == 1 and shape[0] == dim
        unit_stride = lambda s, es: len(es) == 1 and (s[0] == 1 or es[0] == 1)

        t_in = tt[0] is not None
        x_in = xx[0] is not None
        x_dim = signal_size(xx[0])
        y_dim = signal_size(yy[0])
        assert x_dim != 0 and y_dim != 0  # should either be None or > 0
        assert all(signal_size(x) == x_dim for x in xx)
        assert all(signal_size(y) == y_dim for y in yy)

        # check signal input and output shape (implicitly checks
        # for indexing errors)
        if x_in:
            assert all(vector_dims(x.shape, x_dim) for x in xx)
            assert all(unit_stride(x.shape, x.elemstrides) for x in xx)

        assert all(vector_dims(y.shape, y_dim) for y in yy)
        assert all(unit_stride(y.shape, y.elemstrides) for y in yy)

        # try to get OCL code
        in_dims = ([1] if t_in else []) + ([x_dim] if x_in else [])
        ocl_fn = OCL_Function(fn, in_dims=in_dims, out_dim=y_dim)
        input_names = ocl_fn.translator.arg_names
        inputs = []
        if t_in:  # append time
            inputs.append(self.all_data[[self.sidx[t] for t in tt]])
        if x_in:  # append x
            inputs.append(self.all_data[[self.sidx[x] for x in xx]])
        output = self.all_data[[self.sidx[y] for y in yy]]

        return plan_direct(self.queue, ocl_fn.code, ocl_fn.init,
                           input_names, inputs, output, tag=fn_name)

    def _plan_fn_in_python(self, fn, tt, xx, yy, fn_name):
        t_in = tt[0] is not None
        t_idx = self.sidx[self.model.time]
        x_idx = [self.sidx[x] if x is not None else None for x in xx]
        y_idx = [self.sidx[y] if y is not None else None for y in yy]
        ix_iy = list(zip(x_idx, y_idx))

        def m2v(x):  # matrix to vector, if appropriate
            return x[:, 0] if x.ndim == 2 and x.shape[1] == 1 else x

        def v2m(x):  # vector to matrix, if appropriate
            return x[:, None] if x.ndim == 1 else x

        def step():
            t = float(self.all_data[t_idx][0, 0] if t_in else 0)
            for ix, iy in ix_iy:
                args = [t] if t_in else []
                args += [m2v(self.all_data[ix])] if ix is not None else []
                y = fn(*args)
                if iy is not None:
                    self.all_data[iy] = v2m(np.asarray(y))

        return PythonPlan(step, name='python_fn', tag=fn_name)

    def plan_SimNeurons(self, all_ops):
        groups = groupby(all_ops, lambda op: op.neurons.__class__)
        plans = []
        for neuron_class, ops in groups:
            attr_name = '_plan_%s' % neuron_class.__name__
            if hasattr(self, attr_name):
                plans.extend(getattr(self, attr_name)(ops))
            else:
                raise ValueError("Unsupported neuron type '%s'"
                                 % neuron_class.__name__)
        return plans

    def _plan_LIF(self, ops):
        if not all(op.neurons.min_voltage == 0 for op in ops):
            raise NotImplementedError("LIF min voltage")
        dt = self.model.dt
        J = self.all_data[[self.sidx[op.J] for op in ops]]
        V = self.all_data[[self.sidx[op.states[0]] for op in ops]]
        W = self.all_data[[self.sidx[op.states[1]] for op in ops]]
        S = self.all_data[[self.sidx[op.output] for op in ops]]
        ref = self.RaggedArray([op.neurons.tau_ref * np.ones(op.J.size)
                                for op in ops], dtype=J.dtype)
        tau = self.RaggedArray([op.neurons.tau_rc * np.ones(op.J.size)
                                for op in ops], dtype=J.dtype)
        amp = self.RaggedArray([op.neurons.amplitude * np.ones(op.J.size)
                                for op in ops], dtype=J.dtype)
        return [plan_lif(self.queue, dt, J, V, W, S, ref, tau, amp)]

    def _plan_LIFRate(self, ops):
        dt = self.model.dt
        J = self.all_data[[self.sidx[op.J] for op in ops]]
        R = self.all_data[[self.sidx[op.output] for op in ops]]
        ref = self.RaggedArray([op.neurons.tau_ref * np.ones(op.J.size)
                                for op in ops], dtype=J.dtype)
        tau = self.RaggedArray([op.neurons.tau_rc * np.ones(op.J.size)
                                for op in ops], dtype=J.dtype)
        amp = self.RaggedArray([op.neurons.amplitude * np.ones(op.J.size)
                                for op in ops], dtype=J.dtype)
        return [plan_lif_rate(self.queue, dt, J, R, ref, tau, amp)]

    def _plan_AdaptiveLIF(self, ops):
        dt = self.model.dt
        J = self.all_data[[self.sidx[op.J] for op in ops]]
        V = self.all_data[[self.sidx[op.states[0]] for op in ops]]
        W = self.all_data[[self.sidx[op.states[1]] for op in ops]]
        N = self.all_data[[self.sidx[op.states[2]] for op in ops]]
        S = self.all_data[[self.sidx[op.output] for op in ops]]
        ref = self.RaggedArray([op.neurons.tau_ref * np.ones(op.J.size)
                                for op in ops], dtype=J.dtype)
        tau = self.RaggedArray([op.neurons.tau_rc * np.ones(op.J.size)
                                for op in ops], dtype=J.dtype)
        amp = self.RaggedArray([op.neurons.amplitude * np.ones(op.J.size)
                                for op in ops], dtype=J.dtype)
        tau_n = self.RaggedArray([op.neurons.tau_n * np.ones(op.J.size)
                                  for op in ops], dtype=J.dtype)
        inc_n = self.RaggedArray([op.neurons.inc_n * np.ones(op.J.size)
                                  for op in ops], dtype=J.dtype)
        return [plan_lif(self.queue, dt, J, V, W, S, ref, tau, amp,
                         N=N, tau_n=tau_n, inc_n=inc_n)]

    def _plan_AdaptiveLIFRate(self, ops):
        dt = self.model.dt
        J = self.all_data[[self.sidx[op.J] for op in ops]]
        R = self.all_data[[self.sidx[op.output] for op in ops]]
        N = self.all_data[[self.sidx[op.states[0]] for op in ops]]
        ref = self.RaggedArray([op.neurons.tau_ref * np.ones(op.J.size)
                                for op in ops], dtype=J.dtype)
        tau = self.RaggedArray([op.neurons.tau_rc * np.ones(op.J.size)
                                for op in ops], dtype=J.dtype)
        amp = self.RaggedArray([op.neurons.amplitude * np.ones(op.J.size)
                                for op in ops], dtype=J.dtype)
        tau_n = self.RaggedArray([op.neurons.tau_n * np.ones(op.J.size)
                                  for op in ops], dtype=J.dtype)
        inc_n = self.RaggedArray([op.neurons.inc_n * np.ones(op.J.size)
                                  for op in ops], dtype=J.dtype)
        return [plan_lif_rate(self.queue, dt, J, R, ref, tau, amp,
                              N=N, tau_n=tau_n, inc_n=inc_n)]

    def _plan_RectifiedLinear(self, ops):
        J = self.all_data[[self.sidx[op.J] for op in ops]]
        R = self.all_data[[self.sidx[op.output] for op in ops]]
        amp = self.RaggedArray([op.neurons.amplitude * np.ones(op.J.size)
                                for op in ops], dtype=J.dtype)
        return [plan_rectified_linear(self.queue, J, R, amp)]

    def _plan_SpikingRectifiedLinear(self, ops):
        dt = self.model.dt
        J = self.all_data[[self.sidx[op.J] for op in ops]]
        V = self.all_data[[self.sidx[op.states[0]] for op in ops]]
        S = self.all_data[[self.sidx[op.output] for op in ops]]
        amp = self.RaggedArray([op.neurons.amplitude * np.ones(op.J.size)
                                for op in ops], dtype=J.dtype)
        return [plan_spiking_rectified_linear(self.queue, dt, J, V, S, amp)]

    def _plan_Sigmoid(self, ops):
        J = self.all_data[[self.sidx[op.J] for op in ops]]
        R = self.all_data[[self.sidx[op.output] for op in ops]]
        ref = self.RaggedArray([op.neurons.tau_ref * np.ones(op.J.size)
                                for op in ops], dtype=J.dtype)
        return [plan_sigmoid(self.queue, J, R, ref)]

    def plan_SimProcess(self, all_ops):
        class_groups = groupby(all_ops, lambda op: type(op.process))
        plan_groups = defaultdict(list)
        python_ops = []
        for process_class, ops in class_groups:
            for cls in process_class.__mro__:
                attrname = '_plan_' + cls.__name__
                if hasattr(self, attrname):
                    plan_groups[attrname].extend(ops)
                    break
            else:
                python_ops.extend(ops)

        process_plans = [p for attr, ops in iteritems(plan_groups)
                         for p in getattr(self, attr)(ops)]
        python_plans = [p for op in python_ops
                        for p in self._plan_python_process(op)]
        return process_plans + python_plans

    def _plan_python_process(self, op):
        shape = lambda s: s.shape if s is not None else (0,)
        rng = op.process.get_rng(self.rng)
        fn = op.process.make_step(
            shape(op.input), shape(op.output), self.model.dt, rng=rng)
        plans = self._plan_python_fn(fn, [op.t], [op.input], [op.output])
        plan, = plans  # should only be one
        self._python_rngs[rng] = rng.get_state()
        return plans

    def _plan_LinearFilter(self, ops):
        steps = [op.process.make_step(op.input.shape, op.output.shape,
                                      self.model.dt, rng=None) for op in ops]
        A = self.RaggedArray([f.den for f in steps], dtype=np.float32)
        B = self.RaggedArray([f.num for f in steps], dtype=np.float32)
        X = self.all_data[[self.sidx[op.input] for op in ops]]
        Y = self.all_data[[self.sidx[op.output] for op in ops]]
        Xbuf0 = RaggedArray(
            [np.zeros(shape) for shape in zip(B.sizes, X.sizes)],
            dtype=np.float32)
        Ybuf0 = RaggedArray(
            [np.zeros(shape) for shape in zip(A.sizes, Y.sizes)],
            dtype=np.float32)
        Xbuf = CLRaggedArray(self.queue, Xbuf0)
        Ybuf = CLRaggedArray(self.queue, Ybuf0)
        self._raggedarrays_to_reset[Xbuf] = Xbuf0
        self._raggedarrays_to_reset[Ybuf] = Ybuf0
        return plan_linearfilter(self.queue, X, Y, A, B, Xbuf, Ybuf)

    def _plan_WhiteNoise(self, ops):
        assert all(op.input is None for op in ops)

        seeds = [op.process.seed for op in ops]
        cl_rngs = self._create_cl_rngs(seeds)

        Y = self.all_data[[self.sidx[op.output] for op in ops]]
        scale = self.Array([op.process.scale for op in ops], dtype=np.int32)
        inc = self.Array([op.mode == 'inc' for op in ops], dtype=np.int32)
        enums, params = get_dist_enums_params([op.process.dist for op in ops])
        enums = CLRaggedArray(self.queue, enums)
        params = CLRaggedArray(self.queue, params)
        dt = self.model.dt
        return [plan_whitenoise(
            self.queue, Y, enums, params, scale, inc, dt, cl_rngs)]

    def _plan_FilteredNoise(self, ops):
        raise NotImplementedError()

    def _plan_WhiteSignal(self, ops):
        Y = self.all_data[[self.sidx[op.output] for op in ops]]
        t = self.all_data[[self.sidx[self.model.step] for _ in ops]]

        dt = self.model.dt
        signals = []
        for op in ops:
            assert op.input is None and op.output is not None
            rng = op.process.get_rng(self.rng)
            f = op.process.make_step((0,), op.output.shape, dt, rng)
            signals.append(get_closures(f)['signal'])

        signals = self.RaggedArray(signals, dtype=np.float32)
        return [plan_presentinput(self.queue, Y, t, signals, dt)]

    def _plan_PresentInput(self, ops):
        ps = [op.process for op in ops]
        Y = self.all_data[[self.sidx[op.output] for op in ops]]
        t = self.all_data[[self.sidx[self.model.step] for _ in ops]]
        inputs = self.RaggedArray([p.inputs.reshape(p.inputs.shape[0], -1)
                                   for p in ps], dtype=np.float32)
        pres_t = self.Array([p.presentation_time for p in ps])
        dt = self.model.dt
        return [plan_presentinput(self.queue, Y, t, inputs, dt, pres_t=pres_t)]

    def _plan_Conv2d(self, ops):
        plans = []
        for op in ops:
            p, f, b = op.process, op.process.filters, op.process.biases
            assert f.ndim in [4, 6]
            conv = (f.ndim == 4)
            kernel_shape = f.shape[-2:]
            X = self.all_data.getitem_device(self.sidx[op.input])
            Y = self.all_data.getitem_device(self.sidx[op.output])
            ftrans = np.asarray(np.transpose(
                f, (0, 1, 2, 3) if conv else (0, 3, 4, 5, 1, 2)), order='C')
            F = self.Array(ftrans.ravel())
            B = self.Array((np.zeros(p.shape_out) + b).ravel())
            plans.append(plan_conv2d(
                self.queue, X, Y, F, B, p.shape_in, p.shape_out,
                kernel_shape, conv, p.padding, p.strides))

        return plans

    def _plan_Pool2d(self, ops):
        plans = []
        for op in ops:
            assert op.process.kind == 'avg'
            p = op.process
            X = self.all_data.getitem_device(self.sidx[op.input])
            Y = self.all_data.getitem_device(self.sidx[op.output])
            shape = p.shape_out + p.shape_in[1:]
            plans.append(plan_pool2d(
                self.queue, X, Y, shape, p.pool_size, p.strides))

        return plans

    def plan_SimBCM(self, ops):
        pre = self.all_data[[self.sidx[op.pre_filtered] for op in ops]]
        post = self.all_data[[self.sidx[op.post_filtered] for op in ops]]
        theta = self.all_data[[self.sidx[op.theta] for op in ops]]
        delta = self.all_data[[self.sidx[op.delta] for op in ops]]
        alpha = self.Array([op.learning_rate * self.model.dt for op in ops])
        return [plan_bcm(self.queue, pre, post, theta, delta, alpha)]

    def plan_SimOja(self, ops):
        pre = self.all_data[[self.sidx[op.pre_filtered] for op in ops]]
        post = self.all_data[[self.sidx[op.post_filtered] for op in ops]]
        weights = self.all_data[[self.sidx[op.weights] for op in ops]]
        delta = self.all_data[[self.sidx[op.delta] for op in ops]]
        alpha = self.Array([op.learning_rate * self.model.dt for op in ops])
        beta = self.Array([op.beta for op in ops])
        return [plan_oja(self.queue, pre, post, weights, delta, alpha, beta)]

    def plan_SimVoja(self, ops):
        pre = self.all_data[[self.sidx[op.pre_decoded] for op in ops]]
        post = self.all_data[[self.sidx[op.post_filtered] for op in ops]]
        encoders = self.all_data[[self.sidx[op.scaled_encoders] for op in ops]]
        delta = self.all_data[[self.sidx[op.delta] for op in ops]]
        learning_signal = self.all_data[
            [self.sidx[op.learning_signal] for op in ops]]
        scale = self.RaggedArray([op.scale for op in ops], dtype=np.float32)
        alpha = self.Array([op.learning_rate * self.model.dt for op in ops])
        return [plan_voja(self.queue, pre, post, encoders, delta,
                          learning_signal, scale, alpha)]

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

    def print_plans(self):
        print(" Plans ".center(80, '-'))
        for plan in self._plans:
            print("%r" % plan)
            if hasattr(plan, 'description'):
                print(indent(plan.description, 4))

    def print_profiling(self, sort=None):
        """
        Parameters
        ----------
        sort : column to sort by (negative number sorts ascending)
            (0 = n_calls, 1 = runtime, 2 = q-time, 3 = subtime)
        """
        if not self.profiling:
            print("Profiling not enabled!")
            return

        # make and sort table
        table = []
        unknowns = []
        for p in self._plans:
            if isinstance(p, BasePlan):
                t = sum(p.ctimes)
                calls_per_sec = p.n_calls / t if t > 0 else np.nan
                gfps = np.nan  # gigaflops / sec
                gbps = np.nan  # gigabytes / sec
                if p.flops_per_call is not None:
                    gfps = 1e-9 * p.flops_per_call * calls_per_sec
                if p.bw_per_call is not None:
                    gbps = 1e-9 * p.bw_per_call * calls_per_sec
                table.append((p.n_calls, t, gfps, gbps, str(p)))
            else:
                unknowns.append((str(p), getattr(p, 'cumtime', '<unknown>')))

        if sort is not None:
            reverse = sort >= 0
            table.sort(key=lambda x: x[abs(sort)], reverse=reverse)

        # print table
        print(" Profiling ".center(80, '-'))
        print('%8s|%10s|%10s|%10s|' % ('n_calls', 'runtime', 'GF/s', 'GB/s'))

        for r in table:
            print('%8d|%10.3f|%10.3f|%10.3f| %s' % r)

        # totals totals
        print('-' * 80)
        col = lambda c: np.asarray(map(lambda x: x[c], table))
        times = col(1)

        def wmean(x):
            m = ~np.isnan(x)
            tm = times[m]
            return (x[m] * tm).sum() / tm.sum() if tm.size > 0 else np.nan

        print('totals:\t%2.3f\t%2.3f\t%2.3f' % (
            times.sum(), wmean(col(2)), wmean(col(3))))

        # print unknowns
        if len(unknowns) > 0:
            print('\n')
            for r in unknowns:
                print("%s %s" % r)
