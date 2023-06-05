import numpy as np
import matplotlib as mpl
mpl.rcParams['font.size'] = 14
mpl.rcParams['lines.linewidth'] = 2
import matplotlib.pyplot as plt

#%%
# find optimal policy using dynamic programming

def find_optimal_policy(states, actions, horizon, discount_factor, 
                        reward_func, reward_func_last, T):

    V_opt = np.full( (len(states), horizon+1), np.nan)
    policy_opt = np.full( (len(states), horizon), np.nan)
    Q_values = np.full( len(states), np.nan, dtype = object)
    
    for i_state, state in enumerate(states):
        
        # V_opt for last time-step 
        V_opt[i_state, -1] = reward_func_last[i_state]
        # arrays to store Q-values for each action in each state
        Q_values[i_state] = np.full( (len(actions[i_state]), horizon), np.nan)
    
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
            Q_values[i_state][:, i_timestep] = Q
            
    return V_opt, policy_opt, Q_values

#%%
# initial parameters

# states of markov chain
N_intermediate_states = 0
states = np.arange(2 + N_intermediate_states) # intermediate + initial and finished states (2)
actions = np.array([['work', 'shirk'],
                    ['completed']], dtype=object) 

horizon = 10 # deadline
discount_factor = 1.0 # hyperbolic discounting factor
efficacy = 0.9 # self-efficacy (probability of progress on working)

# utilities :
reward_pass = 2.0 
reward_fail = -2.0
reward_shirk = 0.0
effort_work = -0.0 
effort_shirk = -0 
reward_completed = reward_shirk

# reward functions
def get_reward_functions(reward_pass, reward_fail, reward_shirk, reward_completed, effort_work):
    
    reward_func = np.array([[effort_work, reward_shirk + effort_shirk], 
                            [reward_completed]], dtype=object)
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
# example run

reward_func, reward_func_last = get_reward_functions(reward_pass, reward_fail, reward_shirk, 
                                                     reward_completed, effort_work)
T = get_transition_prob(efficacy)
V_opt, policy_opt, Q_values = find_optimal_policy(states, actions, horizon, discount_factor, 
                              reward_func, reward_func_last, T)

# plots of policies and values
plt.figure( figsize = (8, 6) )
for i_state, state in enumerate(states):
    
    plt.plot(V_opt[i_state], label = 'V*%d'%i_state, marker = i_state+4, linestyle = '--')
    #plt.plot(policy_opt[i_state], label = 'policy*')
    
    for i_action, action in enumerate(actions[i_state]):
        
        plt.plot(Q_values[i_state][i_action, :], label = 'Q'+action, marker = i_state+4, linestyle = '--')
    
    #plt.title('state = %d'%state)    
    plt.legend()

#%%
    
efficacys = np.linspace(0, 1, 50)
start_works = np.full( (len(efficacys), 4), np.nan ) # for 4 reward regimes (>>, >, ~>)


# utilities :
reward_pass = 4.0 
reward_fail = -4.0
reward_shirk = 0.05
effort_work = -0.5
reward_completed = reward_shirk

for i_efficacy, efficacy in enumerate(efficacys):

    reward_func, reward_func_last = get_reward_functions(reward_pass, reward_fail, reward_shirk, 
                                                         reward_completed, effort_work)
    T = get_transition_prob(efficacy)
    V_opt, policy_opt, Q_values = find_optimal_policy(states, actions, horizon, discount_factor, 
                                  reward_func, reward_func_last, T)
    
    # find timepoints where it is optimal to work (when task not completed, state=0
    start_work = np.where( policy_opt[0, :] == 0 )[0]
    
    if len(start_work) > 0 :
        start_works[i_efficacy, 3] = start_work[0] # first time to start working 
        #print( policy_opt[0, :])

plt.figure(figsize=(8,6))
legend = ['0.5:0.05', '1:0.05', '2:0.05', '4:0.05']
for i_reward_regime, regime in enumerate(legend):
     plt.plot(efficacys, start_works[:, i_reward_regime], label = legend[i_reward_regime])
plt.xlabel('efficacy')
plt.ylabel('time to start work')
plt.legend()
plt.title('effort = %1.1f'%effort_work)

#%%

efforts = np.linspace(-8, 1, 50)
start_works = np.full( (len(efforts), 4), np.nan ) # for 4 reward regimes (>>, >, ~>)

# utilities :
efficacy = 0.9
reward_pass = 2.0 
reward_fail = -2.0
reward_shirk = 0.05
reward_completed = reward_shirk

for i_effort, effort_work in enumerate(efforts):

    reward_func, reward_func_last = get_reward_functions(reward_pass, reward_fail, reward_shirk,
                                                         reward_completed, effort_work)
    T = get_transition_prob(efficacy)
    V_opt, policy_opt, Q_values = find_optimal_policy(states, actions, horizon, discount_factor, 
                                  reward_func, reward_func_last, T)
    
    # find timepoints where it is optimal to work (when task not completed, state=0)
    start_work = np.where( policy_opt[0, :] == 0 )[0]
    
    if len(start_work) > 0 :
        start_works[i_effort, 3] = start_work[0] # first time to start working
        #print( policy_opt[0, :])

plt.figure(figsize=(8,6))
legend = ['0.5:0.05', '1:0.05', '2:0.05', '4:0.05']
for i_reward_regime, regime in enumerate(legend):
     plt.plot(efforts, start_works[:, i_reward_regime], label = regime)
plt.xlabel('effort to work')
plt.ylabel('time to start work')
plt.legend()
plt.title('efficacy = %1.1f'%efficacy)