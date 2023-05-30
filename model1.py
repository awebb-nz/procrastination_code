import numpy as np
import matplotlib.pyplot as plt

#%%
# find optimal policy using dynamic programming

def find_optimal_policy(states, actions, horizon, discount_factor, 
                        reward_func, reward_func_last, T):

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
                
                # q-value for each action (bellman equation)
                Q[i_action] = reward_func[i_state][i_action] + discount_factor * (
                              T[i_state][i_action] @ V_opt[states, i_timestep+1] )
            
            # find optimal action (which gives max q-value)
            V_opt[i_state, i_timestep] = np.max(Q)
            policy_opt[i_state, i_timestep] = np.argmax(Q)
            
    return V_opt, policy_opt

#%%
# initial parameters

# states of markov chain
N_intermediate_states = 0
states = np.arange(2 + N_intermediate_states) # intermediate + initial and finished states (2)
actions = np.array([['work', 'shirk'],
                    ['completed']], dtype=object) 

horizon = 10 # deadline
discount_factor = 0.9 # hyperbolic discounting factor
efficacy = 0.9 # self-efficacy (probability of progress on working)

# utilities :
reward_pass = 0.1 
reward_fail = -0.1
reward_shirk = 0.05
effort_work = -1

# reward functions
def get_reward_functions(reward_pass, reward_fail, reward_shirk, effort_work):
    
    reward_func = np.array([[effort_work, reward_shirk], 
                            [0]], dtype=object)
    reward_func_last = np.array([reward_fail, reward_pass])
    
    return reward_func, reward_func_last

# transition probabilities
def get_transition_prob(efficacy):
    
    T_work = np.array([1-efficacy, efficacy])
    T_shirk = np.array([1, 0])
    T_completed = np.array([0, 1])
    T = np.array([[T_work, T_shirk],
                  [T_completed]], dtype=object)
    
    return T
            
#%%
    

efficacys = np.linspace(0, 1, 20)
start_works = np.full( (len(efficacys), 4), np.nan ) # for 3 reward regimes (>>, >, ~>)


# utilities :
reward_pass = 1.0 
reward_fail = -1.0
reward_shirk = 0.05
effort_work = -1

for i_efficacy, efficacy in enumerate(efficacys):

    reward_func, reward_func_last = get_reward_functions(reward_pass, reward_fail, reward_shirk, effort_work)
    T = get_transition_prob(efficacy)
    V_opt, policy_opt = find_optimal_policy(states, actions, horizon, discount_factor, 
                            reward_func, reward_func_last, T)
    #print(policy_opt[0,:])
    
    # find timepoint where it becomes optimal to start working (when task not completed)
    start_work = np.where( policy_opt[0, 1:] != policy_opt[0, :-1] )[0]
    
    if len(start_work) > 0 :
        start_works[i_efficacy, 2] = start_work[0] + 1  


