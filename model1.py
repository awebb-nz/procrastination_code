import numpy as np
import matplotlib.pyplot as plt

#%%
# initial parameters

# states of markov chain
N_intermediate_states = 0
states = np.arange(2 + N_intermediate_states)
actions = np.array([['work', 'shirk'],
                    ['completed']], dtype=object)

# changeable params
horizon = 10 # deadline
efficacy = 0.9 # self-efficacy (probability of progress on working)
discount_factor = 0.9
# reward functions
reward = 1 # unit reward
effort = -1 # unit effort
reward_pass = 1 # in unit rewards
reward_fail = -1
reward_shirk = 0.2
effort_work = 1

reward_func = np.array([[effort_work * effort, reward_shirk * reward],
                        [0]], dtype=object)
reward_func_last = np.array([reward_fail * reward, reward_pass * reward])

# transition probabilities
T_work = np.array([1-efficacy, efficacy])
T_shirk = np.array([1, 0])
T_completed = np.array([1])
T = np.array([[T_work, T_shirk],
              [T_completed]], dtype=object)
S_prime = np.array([[[0, 1], [0, 1]],
                     [1]], dtype=object)

#%%
# find optimal policy using dynamic programming

V_opt = np.full( (len(states), horizon+1), np.nan)
policy_opt =  np.full( (len(states), horizon), np.nan)

# V_opt for last time-step
for i_state, state in enumerate(states):
    
    V_opt[i_state, -1] = reward_func_last[i_state]    

# backward induction to derive optimal policy  

for i_timestep in range(horizon-1, -1, -1):
    
    for i_state, state in enumerate(states):
        
        Q = np.full( len(actions[i_state]), np.nan)
        
        for i_action, action in enumerate(actions[i_state]):
            
            Q[i_action] = reward_func[i_state][i_action] + discount_factor * (
                          T[i_state][i_action] @ V_opt[[S_prime[i_state][i_action]], i_timestep+1] )
        
        V_opt[i_state, i_timestep] = np.max(Q)
        policy_opt[i_state, i_timestep] = np.argmax(Q)
            
#%%

