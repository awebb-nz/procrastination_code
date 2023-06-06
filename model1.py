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
N_intermediate_states = 1
states = np.arange(2 + N_intermediate_states) # intermediate + initial and finished states (2)

# construct action matrix
actions = np.full(len(states), np.nan, dtype = object)
actions[:-1] = [ ['work', 'shirk'] for i in range( len(states)-1 )] # actions for all states but final
actions[-1] =  ['completed'] # actions for final state

horizon = 10 # deadline
discount_factor = 0.9 # hyperbolic discounting factor
efficacy = 0.9 # self-efficacy (probability of progress on working)

# utilities :
reward_pass = 2.0 
reward_fail = -2.0
reward_shirk = 0.5
effort_work = -0.4
effort_shirk = -0 
reward_completed = reward_shirk

# construct reward functions
def get_reward_functions(states, reward_pass, reward_fail, reward_shirk, reward_completed, effort_work):
    
    # reward from actions within horizon
    reward_func = np.full(len(states), np.nan, dtype = object)
    reward_func[:-1] = [ [effort_work, reward_shirk + effort_shirk] for i in range( len(states)-1 )]
    reward_func[-1] = [reward_completed]
    
    # reward from final evaluation
    reward_func_last =  np.linspace(reward_fail, reward_pass, len(states)) 
    
    return reward_func, reward_func_last

# construct transition matrix
def get_transition_prob(states, efficacy):
    
    T = np.full(len(states), np.nan, dtype = object)
    
    # for 3 states:
    T[0] = [ np.array([1-efficacy, efficacy, 0]), 
             np.array([1, 0, 0]) ] # transitions for work, shirk
    T[1] = [ np.array([0, 1-efficacy, efficacy]), 
             np.array([0, 1, 0]) ] # transitions for work, shirk
    T[2] = [ np.array([0, 0, 1]) ] # transitions for completed
    
#    # for 2 states:
#    T[0] = [ np.array([1-efficacy, efficacy]), 
#             np.array([1, 0]) ] # transitions for work, shirk
#    T[1] = [ np.array([0, 1]) ] # transitions for completed
    
    return T

#%% 
# example run

reward_func, reward_func_last = get_reward_functions(states, reward_pass, reward_fail, reward_shirk, 
                                                     reward_completed, effort_work)
T = get_transition_prob(states, efficacy)
V_opt, policy_opt, Q_values = find_optimal_policy(states, actions, horizon, discount_factor, 
                              reward_func, reward_func_last, T)

# plots of policies and values
plt.figure( figsize = (8, 6) )
for i_state, state in enumerate(states):
    
    #plt.figure( figsize = (8, 6) )
    
    plt.plot(V_opt[i_state], label = 'V*%d'%i_state, marker = i_state+4, linestyle = '--')
    #plt.plot(policy_opt[i_state], label = 'policy*')
    
    for i_action, action in enumerate(actions[i_state]):
        
        plt.plot(Q_values[i_state][i_action, :], label = 'Q'+action, marker = i_state+4, linestyle = '--')
    
    #plt.title('state = %d'%state)    
    plt.legend()

#%%
    
efficacys = np.linspace(0, 1, 50)
start_works = np.full( (len(efficacys), N_intermediate_states+1, 4), np.nan ) # for 4 reward regimes (>>, >, ~>)

horizon = 10 # deadline
discount_factor = 0.9 # hyperbolic discounting factor
# utilities :
reward_pass = 4.0 
reward_fail = -4.0
reward_shirk = 0.2
effort_work = -0.4
reward_completed = reward_shirk

for i_efficacy, efficacy in enumerate(efficacys):

    reward_func, reward_func_last = get_reward_functions(states, reward_pass, reward_fail, reward_shirk, 
                                                         reward_completed, effort_work)
    T = get_transition_prob(states, efficacy)
    V_opt, policy_opt, Q_values = find_optimal_policy(states, actions, horizon, discount_factor, 
                                  reward_func, reward_func_last, T)
    
    for i_state in range(N_intermediate_states+1):
        # find timepoints where it is optimal to work (when task not completed, state=0
        start_work = np.where( policy_opt[i_state, :] == 0 )[0]
        
        if len(start_work) > 0 :
            start_works[i_efficacy, i_state, 3] = start_work[0] # first time to start working 
            #print( policy_opt[0, :])

for i_state in range(N_intermediate_states+1):
    plt.figure(figsize=(8,6))
    legend = ['0.5:0.5', '1:0.5', '2:0.5', '4:0.5']
    for i_reward_regime, regime in enumerate(legend):
         plt.plot(efficacys, start_works[:, i_state, i_reward_regime], label = legend[i_reward_regime])
    plt.xlabel('efficacy')
    plt.ylabel('time to start work')
    plt.legend()
    plt.title('effort = %1.1f, state = %d'%(effort_work, i_state))

#%%

efforts = np.linspace(-8, 1, 50)
start_works = np.full( (len(efforts), N_intermediate_states+1, 4), np.nan ) # for 4 reward regimes (>>, >, ~>)


horizon = 10 # deadline
discount_factor = 0.9 # hyperbolic discounting factor
# utilities :
efficacy = 0.8
reward_pass = 0.5 
reward_fail = -0.5
reward_shirk = 0.5
reward_completed = reward_shirk

for i_effort, effort_work in enumerate(efforts):

    reward_func, reward_func_last = get_reward_functions(states, reward_pass, reward_fail, reward_shirk,
                                                         reward_completed, effort_work)
    T = get_transition_prob(states, efficacy)
    V_opt, policy_opt, Q_values = find_optimal_policy(states, actions, horizon, discount_factor, 
                                  reward_func, reward_func_last, T)
    
    for i_state in range(N_intermediate_states+1):
        
        # find timepoints where it is optimal to work (when task not completed, state=0)
        start_work = np.where( policy_opt[i_state, :] == 0 )[0]
        
        if len(start_work) > 0 :
            start_works[i_effort, i_state, 0] = start_work[0] # first time to start working
            #print( policy_opt[0, :])
            
for i_state in range(N_intermediate_states+1):
    plt.figure(figsize=(8,6))
    legend = ['0.5:0.5', '1:0.5', '2:0.5', '4:0.5']
    for i_reward_regime, regime in enumerate(legend):
         plt.plot(efforts, start_works[:, i_reward_regime], label = regime)
    plt.xlabel('effort to work')
    plt.ylabel('time to start work')
    plt.legend()
    plt.title('efficacy = %1.1f state = %d' %(efficacy, i_state) )