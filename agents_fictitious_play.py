# https://github.com/esalonico/seminar-learning-in-games-ss23/blob/main/fictitious-play.ipynb
# This is a fictitious play implementation of some games
# All should be repeated with mutiple agents
# zero-sum not mandatory

# In fictitious play, the agent knows the opponent's strategy
# and plays best response to it

# To see if we have converged to Nahs equilibrium,
# keep track of each agents strategies played

# GAMES:
# 1. Matching Pennies: mixed strategies, zero-sum ( + stochastic version with coin toss)
# 2. Rock-paper-Scissor: general-sum
# 3. Prisoner's Dillema: zero-sum, 1 pure Nash
# 4. Battle-of-sexes: cooperative, 2 pure Nash

import numpy as np
import matplotlib.pyplot as plt

def best_response(belief):
    strat = actions[np.argmax(belief)]
    
    return strat

# for mixed strategies
def compute_expect_utility(actions, belief):
    # compute the explicit utility of an agent
    # for each action, based on the empirical distribution
    # of the actions of its opponents
    
    '''
    dimensions n * 1, where n # actions given agent can make
    and each is the expected utility for action n
    '''
    exp_ut = []
    for action in actions:
        # (action, H), (action,T)
        exp_ut.append(payoff_matrix[action][0]*belief[0] + payoff_matrix[action][0]*belief[1])
    
    best_response(exp_ut)
    return strat_a_num
    
def update_beliefs(action, belief):
    belief[action] += 1
        
    total = sum(belief)
    return total, belief

"""
#################################
#    1. Matching Pennies        # 
#################################
            H         T
        ###################
    H   # (1, -1)  (-1, 1)#
    T   # (-1, 1)  (1, -1)#
        ###################
        
        Agent's' action goeas by row
        Opponent's by column
        
        Empirical distribution should CONVERGE to (0.5, 0.5)
        and thus to Nash equilibrium
        
        Even if payoff is given, the output is random (the penny state is random),
        no matter the agents' choice
        
        1 NE at (0.5, 0.5)
"""
# actions
actions = {0: "H", 1: "T"}
# reward
payoff_matrix = np.array([[1, -1], [-1, 1]])

'''
initialise beliefs
'''
# policy (based on probability)  holds empirical 
strategies = []

# count of H and T played by each agent
# for state 0 lets assume both have played Tails
action_count = [[0,1], [0,1]]

sums = [0, 0]
for indx, player_actions in enumerate(action_count):
    sums[indx] = sum(player_actions)
    #sum2 += action_count[1][action]

# initialise beliefs with mixed strategy of each agents
# each row is the mixed strategy/ probability of all actions of the agent
# empirical_distribution
# P(a) = w(a)/SUM(w(a')), probability of opponent playing action a and a' all actions (includiing a)
beliefs = np.array(([action_count[0][0]/sums[0], action_count[0][1]/sums[0]], [action_count[1][0]/sums[1], action_count[1][1]/sums[1]]))
#beliefs = np.array(([1.5, 2], [2, 1.5]))


#(2 players, 2 strategies (head or tail))
#empirical_distribution = {}
#total_actions = len(actions)
#for action, action_count in action_count[0].items():
#    empirical_distribution[action] = action_count / total_actions

max_iters = 100
# n * m matrix, where n iterations and m tuples represetning number of agents. 
# Each tuple m is the mixed strategy (probability of actions) of each agent m at iteration n 
strategies = beliefs
moves = []
cur_actions = []
for i in range(max_iters):
    # choose action by best response
    compute_expect_utility()
    
    best_actions = [best_response(beliefs[0]), best_response(beliefs[1])]
    
    # and update their mixed strategy/beliefs based on their best response
    total, beliefs[0] = update_beliefs(best_actions[0], beliefs[0])
    total, beliefs[1]  = update_beliefs(best_actions[1], beliefs[1])


    # keep beliefs and strategies in each iteration for computing convergence
    moves.append((actions[np.argmax(beliefs[0])],actions[np.argmin(beliefs[1])] )) #
    

