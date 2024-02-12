# https://github.com/esalonico/seminar-learning-in-games-ss23/blob/main/fictitious-play.ipynb
# This is a fictitious play implementation of some games
# All should be repeated with mutiple agents
# zero-sum not mandatory

# In fictitious play, the agent knows the opponent's strategy
# and plays best response to it

# To see if we have converged to Nahs equilibrium,
# keep track of each agents strategies played

# GAMES:
# 1. Matching Pennies: competitive, mixed strategies, zero-sum ( + stochastic version with coin toss)
# 2. Prisoner's Dillema: zero-sum, 1 pure Nash
# 3. Battle-of-sexes: cooperative, 2 pure Nash, 1 mixed
# 4. (Left-Right): pure cooperative, 2 pure Nash, 1 mixed
#
#   #################################
#   #    1. Matching Pennies        # 
#   #################################
#   competitive, mixed strategies, zero-sum ( + stochastic version with coin toss)
#
#            H         T
#        ###################
#    H   # (1, -1)  (-1, 1)#
#    T   # (-1, 1)  (1, -1)#
#        ###################
#        
#        Agent's' action goeas by row
#        Opponent's by column
#        
#        Empirical distribution should CONVERGE to (0.5, 0.5)
#        and thus to Nash equilibrium
#        
#        Even if payoff is given, the output is random (the penny state is random),
#        no matter the agents' choice
#        
#        1 NE at (0.5, 0.5)
#       
#       -> extra : Rock-paper-Scissor: general-sum
#


# Also try with 3-players

import numpy as np
import argparse
import matplotlib.pyplot as plt
import pandas as pd

def best_response(actions, exp_utilities):
    strat = np.argmax(exp_utilities)
    print('The best action is ', actions[np.argmax(exp_utilities)])
    return strat

# for mixed strategies
def compute_expect_utility(actions, belief, payoff_matrix):
    # compute the explicit utility of an agent
    # for each action, based on the empirical distribution
    # of the actions of its opponents
    
    '''
    dimensions n * 1, where n # actions given agent can make
    and each is the expected utility for action n
    '''
    exp_ut = []
    for action in range(len(actions)):
        # (action, H), (action,T)
        exp_ut.append(payoff_matrix[action][0]*belief[0] + payoff_matrix[action][1]*belief[1])
    print("util", belief, payoff_matrix)
    print("utilities are ", exp_ut)
    return best_response(actions, exp_ut)
    
def update_beliefs(actions, action_count):
    print('----update ----')
    #total = sum(action_count)
    new_belief = []
    for action in range(len(actions)):
        print(action)
        new_belief.append(action_count[action]/sum(action_count))
        print(new_belief)
    # return new belief
    new_belief = np.array(new_belief)
    print('------')
    return new_belief

"""
INITIALIZATION OF GAMES

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
def matching_pennies_init():
    # actions
    actions = {0: "H", 1: "T"}
    # reward
    payoff_matrix_agent1 = np.array([[1, -1], [-1, 1]])
    payoff_matrix_agent2 = np.array([[-1, 1], [1, -1]])
    payoff_matrices = [payoff_matrix_agent1, payoff_matrix_agent2]
    print("PAYOFF MATRICES", payoff_matrices )
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
    beliefs = np.array(([action_count[1][0]/sum(action_count[1]), action_count[1][1]/sum(action_count[1])], [action_count[0][0]/sum(action_count[0]), action_count[0][1]/sum(action_count[0])]))
    beliefs = np.array(([1.5, 2], [2, 1.5]))

    return actions, payoff_matrices, action_count, beliefs

def prisoners_dillema_init():
    # actions
    actions = {0: "H", 1: "T"}
    # reward
    payoff_matrix_agent1 = np.array([[1, -1], [-1, 1]])
    payoff_matrix_agent2 = np.array([[-1, 1], [1, -1]])
    payoff_matrices = [payoff_matrix_agent1, payoff_matrix_agent2]
    print("PAYOFF MATRICES", payoff_matrices )
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
    beliefs = np.array(([action_count[1][0]/sum(action_count[1]), action_count[1][1]/sum(action_count[1])], [action_count[0][0]/sum(action_count[0]), action_count[0][1]/sum(action_count[0])]))
    beliefs = np.array(([1.5, 2], [2, 1.5]))

    return actions, payoff_matrices, action_count, beliefs

def battle_of_sexes_init():
    # actions
    actions = {0: "H", 1: "T"}
    # reward
    payoff_matrix_agent1 = np.array([[1, -1], [-1, 1]])
    payoff_matrix_agent2 = np.array([[-1, 1], [1, -1]])
    payoff_matrices = [payoff_matrix_agent1, payoff_matrix_agent2]
    print("PAYOFF MATRICES", payoff_matrices )
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
    beliefs = np.array(([action_count[1][0]/sum(action_count[1]), action_count[1][1]/sum(action_count[1])], [action_count[0][0]/sum(action_count[0]), action_count[0][1]/sum(action_count[0])]))
    beliefs = np.array(([1.5, 2], [2, 1.5]))

    return actions, payoff_matrices, action_count, beliefs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("game", help="choose which game to play. Range [1-3]", required=False)
    args = parser.parse_args()
    game = args.game

    # check if game option given, default matching pennies
    if game is None:
        game == 1

    if game == 1:
        actions, payoff_matrices, action_count, beliefs = matching_pennies_init()
    elif game == 2:
        actions, payoff_matrices, action_count, beliefs = prisoners_dillema_init()
    elif game == 3:
        actions, payoff_matrices, action_count, beliefs = battle_of_sexes_init

    max_iters = 10
    # n * m matrix, where n iterations and m tuples represetning number of agents. 
    # Each tuple m is the mixed strategy (probability of actions) of each agent m at iteration n 
    strategies = [beliefs.tolist()]
    moves = []
    for i in range(max_iters):
        # choose action by best response
        best_actions = [compute_expect_utility(actions, beliefs[0], payoff_matrices[0]), compute_expect_utility(actions, beliefs[1], payoff_matrices[1])]

        # choose action
        action_count[0][best_actions[0]] += 1
        action_count[1][best_actions[1]] += 1

        # update their mixed strategy/beliefs based on their action
        beliefs[0] = update_beliefs(actions, action_count[1])
        beliefs[1]  = update_beliefs(actions, action_count[0])
        print(action_count)
        print("===\t\t", beliefs[0], beliefs[1], beliefs, best_actions, '===')
        print("at ", i, "best actions are ", best_actions)
        # keep beliefs and actions in each iteration for computing convergence
        strategies.append(beliefs.tolist())
        moves.append(best_actions)
        #moves.append((actions[np.argmax(beliefs[0])],actions[np.argmin(beliefs[1])] ))
        
    ### PLOT
    print("PLOTTING")
    print(moves)
    print("and")
    print(strategies)
    print(pd.DataFrame(moves, columns=["Action_agent1", "Action_agent2"]).expanding())
        

