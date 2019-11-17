# Notes 

--- 
## Algorithms 

#### On-Policy Algos 
- don't use old data, need on-policy data for updates, weaker on sample efficiency
- directly optimize objective (policy performance), better stability
- i.e. TRPO, PPO 

#### Off-Policy Algos 
- reuse old data efficiently, exploit Bellman's equations for optimality --> train Q-function with **any** environment interaction data (as long as enough high-reward areas in env)
- no guarantees on great performance, brittle and unstable 
- i.e. DDPG (learn both Q-function and policy), Q-learning, TD3, SAC

#### Code layout 
core.py 
- contain utils 

- algo.py
1. Logger setup
2. Random seed setting
3. Environment instantiation
4. Making placeholders for the computation graph
5. Building the actor-critic computation graph via the actor_critic function passed to the algorithm function as an argument
6. Instantiating the experience buffer
7. Building the computation graph for loss functions and diagnostics specific to the algorithm
8. Making training ops
9. Making the TF Session and initializing parameters
10. Setting up model saving through the logger
11. Defining functions needed for running the main loop of the algorithm (e.g. the core update function, get action function, and test agent function, depending on the algorithm)
12. Running the main loop of the algorithm:
    - Run the agent in the environment
    - Periodically update the parameters of the agent according to the main equations of the algorithm
    - Log key performance metrics and save agent

--- 
## Experiments

- Every hyperparameter in every algorithm can be controlled directly from the command line. If `kwarg` is a valid keyword arg for the function call of an algorithm, you can set values for it with the flag `--kwarg`
- Values pass through `eval()` before being used, so you can describe some functions and objects directly from the command line. 
- There’s some nice handling for kwargs that take dict values. Instead of having to provide
```bash
--key dict(v1=value_1, v2=value_2)
```
you can give
```bash
--key:v1 value_1 --key:v2 value_2
```
- launching mutiple experiments at once (sequentially) by setting a list of seeds 
- results saved to
```bash
data_dir/[outer_prefix]exp_name[suffix]/[inner_prefix]exp_name[suffix]_s[seed]
```
where `outer_prefix` is a `YY-MM-DD_` timestamp, `inner_prefix` is a `YY-MM-DD_HH-MM-SS-` timestamp, `suffix` is hyperparameters


--- 
## Utils 

--- 
#### Logging 

Logger 
- save diagnostics, hyperparameter configurations, state of training run and trained model; takes in `output_dir`, `output_fname` (.txt, tab-separated), `exp_name` (plotter can group experiments with same name, i.e. same hyperparameter with several random seeds)
- dump_tabular
    - write diagnostics to file and stdout 
- log
    - print colorized message to stdout 
- save_config 
    - save a dict of named arguments, serialize to JSON 
    - called it once at top of experiment
- save_state
    - save model, takes in `state_dict` and `itr` (None for state overwriting)
- setup_tf_saver
    - for model saving, takes in a tf session, inputs and outputs as dicts (keys to input placeholders/output nodes)
    - called once after defining computation graph but before training 


EpochLogger 
- inherit from **Logger**, easier to track average, standard deviation, min and max of any diagnostic over each epoch and across MPI workers
- get_stats 
    - get mean/std/min/max of a diagnostic
- log_tabular 
    - computes average, std, min, max of all value sin internal state, then state is wiped clean (prevent leakage to next epoch)
    - done at the end of an epoch
    - if provide `val`, the key must not have been stored before (i.e. time, epoch, total_gradient_steps)
- store 
    - save values of keys in logger internal state 

#### Typical usage 
``` python 
# instantiate logger 
logger = EpochLogger(**logger_kwargs)
logger.save_config(locals())

logger.setup_tf_saver()

for epochs:
    for steps: 
        # training operations 
        logger.store(...)
    if save_freq:
        logger.save_state(...)
    # for a bunch of key-value 
    logger.log_tabular(...)    
    logger.dump_tabular()
```

---
#### Plotter 

- use seaborn 
- optionally smooth over a windown size 
- provide log_dir, autocomplete to find all experiments of same name and plot/compare 


---
#### Runner 

ExperimentGrid
- based on `rllab VariantGenerator`, optionally takes in `name`
- add 
    - add a parameter (key) to the grid config, with potential values (vals)
- run
    - takes in `thunk`, `num_cpu`, `data_dir`, `datastamp`
    - run each variant in the grid with function `thunk`
    - use `call_experiment` to launch each experiment 
- variant_name 
    - takes in `variant` as dict, make an exp_name 
- variants 
    - makes a list of dicts, each is a valid config in the grid 

call_experiment 
- takes in `exp_name`, `thunk`, `seed`, `num_cpu`, `data_dir`, `datestamp`
- useful for running many experiments in sequence, handles splitting into multiple processes for MPI 
- function is serialized into string, then execute `run_entrypoint.py` in a subprocess with the string, it unserializes the function and executes it (prevent leaking state between experiments)
```python 
# run_entrypoint.py
import zlib
import pickle
import base64

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('encoded_thunk')
    args = parser.parse_args()
    thunk = pickle.loads(zlib.decompress(base64.b64decode(args.encoded_thunk)))
    thunk()
```