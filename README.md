# Nengo_Zbrodoff
A Nengo model designed to count, learn and retrieve alphabet-arithmetic problems such as A+2=C and to respond YES or NO depending on the validity of the input problem.

## Installation
The model was designed using Nengo (version 2.8.0).
Nengo requires python3 (version 3.6.8).

To improve computing time, the OpenCL version of nengo was used (nengo-ocl version 1.4.0).
nengo-ocl requires pyopencl (version 2018.1.1). The model failed to run using newer versions of pyopencl.
NOTE: It is advised to use the nengo-ocl version included in this repository, as it contains a few additional files.

## Nengo Model
The following files contain different stages of development for the Nengo model. final_model.py contains the main model.
### final_model.py
The final version, able to calculate, memorise and retrieve problems of the form A+2=C. 

The problem can be presented in ll. 123-124.
For retrieval of a previously learned problem, 'load_from' must be active (l. 246).
To create new memory: must be simulated using shell command 'python3 final_model.py', which creates a test.npz file.
  To learn with empty memory, remove 'load_from' (l.246).

In order to change learning and retrieval in the model actions (ll.305-329):
  Learning is switched on by setting 'do_learn=GO'. Default is no learning (do_learn=CLEAR).
  retrieval is switched on by setting 'do_dm=GO'. Default is no retrieval (do_dm=CLEAR).
  Memory module receives input via 'in_layer' and 'correct' networks (see l. 324).
### count_retrieve.py
Intermediate version of the model. Is able to count and retrieve from a predefined memory, but is unable to create new memory. Yields better output than the final model.

Memory content can be predefined in ll. 112-114.
Input problem can be changed in ll. 117-1118.
### bcm_zbrodoff.py
Memory module to memorise and retrieve problems of the form A+2=C.

For recall:
Switch off ll. 127, 129, 139 and 142-145.
Make sure 'load_from' is active (l.133).
Input should have the form 'ITEM1*B + ITEM2*TWO' (ll. 71-91).
Recall is run in Nengo GUI.

For learning:
Switch on all learning parameters and connections (see recall).
Optional: switch off 'load_from' (l. 133) to use an empty memory.
Input should have the form 'ITEM1*B + ITEM2*TWO + RESULT*D' (ll. 71-91).
To store memory, simulate model using shell command 'python3 bcm_zbrodoff.py'. Duration of simulation can be set in l. 156. Must be at least 1 second long. 
### count.py
The count model before combining the different parts. Works for a limited range pf problems.
The problem can be defined in ll.58-59.
### zbrodoff_retrieve.py
Model is only able to retrieve. Has all problems containing A-F and 2-4 stored in memory.
Problem can be defined in l. 100.
### count_original.py
The count model as given to me as basis for the project. 
The problem can be defined in l. 36.
### bcm_tryout_gui.py
Memory module as given to me as basis for this project. Not designed for alphabet-arithmetic problem.
Input over time can be set in ll. 24-28. Can memorise word-pairs containing the words 'CAT', 'DOG', 'CHAIR', and 'TABLE'.
