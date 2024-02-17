# https://github.com/esalonico/seminar-learning-in-games-ss23/blob/main/fictitious-play.ipynb
# This is a fictitious play implementation of some games
# All should be repeated with mutiple agents

# In fictitious play, the agent knows the opponent's strategy
# and plays best response to it

# To see if we have converged to Nash equilibrium,
# keep track of each agents empirical frequency of strategies played

# GAMES:
# 1. Matching Pennies:  pure competitive, 1 mixed Nash, zero-sum ( + stochastic version with coin toss)
# 2. Prisoner's Dillema: zero-sum, 1 pure Nash
# 3. Battle-of-sexes: mixed cooperative-competitive, 2 pure Nash, 1 mixed
# 4. (Left-Right): pure cooperative, 2 pure Nash, 1 mixed
# Also try with 3-players

import numpy as np
import argparse
import matplotlib.pyplot as plt
import pandas as pd

def best_response(actions, exp_utilities):
    strat = np.argmax(exp_utilities)
    #print('The best action is ', actions[np.argmax(exp_utilities)])
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
        sum = 0
        for opponnent_action in range(len(actions)):
            sum += payoff_matrix[action][opponnent_action]*belief[opponnent_action]
        exp_ut.append(sum)
    return best_response(actions, exp_ut)
    
def update_beliefs(actions, action_count):
    #total = sum(action_count)
    new_belief = []
    for action in range(len(actions)):
        new_belief.append(action_count[action]/sum(action_count))
    # return new belief
    new_belief = np.array(new_belief)
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
        -> extra : Rock-paper-Scissor: general-sum
"""
def matching_pennies_init(advanced=False):
    
    # Advanced mode is Rock-Papers-Scissors
    if advanced:
        actions = {0: "R", 1: "P", 2: "S"}
        payoff_matrix_agent1 = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
        payoff_matrix_agent2 = np.array([[0, 1, -1], [-1, 0, 1], [-1, 1, 0]])
    else:
        actions = {0: "H", 1: "T"}
        payoff_matrix_agent1 = np.array([[1, -1], [-1, 1]])
        payoff_matrix_agent2 = np.array([[-1, 1], [1, -1]])
    payoff_matrices = [payoff_matrix_agent1, payoff_matrix_agent2]
    # count of H and T played by each agent
    # for state 0 lets assume both have played Tails
    if advanced:
        action_count = [[0, 1, 0], [0, 1, 0]]
    else:
        action_count = [[0,1], [0,1]]
    #action_count = [[1.5, 2], [2, 1.5]]
    
    # initialise beliefs with mixed strategy of each agents
    # each row is the mixed strategy/ probability of all actions of the agent
    # empirical_distribution
    # P(a) = w(a)/SUM(w(a')), probability of opponent playing action a and a' all actions (includiing a)
    belief1 = []
    belief2  =[]
    for action in range(len(actions)):
        belief1.append(action_count[1][action]/sum(action_count[1]))
        belief2.append(action_count[0][action]/sum(action_count[0]))
    beliefs = np.array((belief1, belief2))
    print(beliefs)
    

    return actions, payoff_matrices, action_count, beliefs

'''
#################################
#    2. Prisoners Dillema       # 
#################################
            S         B
        ###################
    S   # (-1, -1)  (-3, 0)#
    B   # (0, -3)  (-2, -2)#
        ###################

        1 pure Nash: 
            Betray-Betray
'''
def prisoners_dillema_init():
    actions = {0: "Silent", 1: "Betray"}
    payoff_matrix_agent1 = np.array([[-1, -3], [0, -2]])
    payoff_matrix_agent2 = np.array([[-1, -3], [0, -2]])
    payoff_matrices = [payoff_matrix_agent1, payoff_matrix_agent2]
    # count of H and T played by each agent
    # for state 0 lets assume both have played Tails
    action_count = [[1,0], [1,0]]

    
    beliefs = np.array(([action_count[1][0]/sum(action_count[1]), action_count[1][1]/sum(action_count[1])], [action_count[0][0]/sum(action_count[0]), action_count[0][1]/sum(action_count[0])]))

    return actions, payoff_matrices, action_count, beliefs

'''
#################################
#    3. Battle of Sexes        # 
#################################
            O         M
        ###################
    O   # (3, 2)  (0, 0)#
    M   # (0, 0)  (2, 3)#
        ###################

        2 pure Nash: 
            -> Opera-Opera
            -> Movies-Movies
        1 mixed Nash:
            -> Opera (3/5) - Opera (2/5)
'''
def battle_of_sexes_init():
    # actions
    actions = {0: "Opera", 1: "Movies"}
    # reward
    payoff_matrix_agent1 = np.array([[3, 0], [0, 2]])
    payoff_matrix_agent2 = np.array([[2, 0], [0, 3]])
    payoff_matrices = [payoff_matrix_agent1, payoff_matrix_agent2]
    # count of H and T played by each agent
    # for state 0 lets assume both have played Tails
    action_count = [[0,1], [1,0]]

    beliefs = np.array(([action_count[1][0]/sum(action_count[1]), action_count[1][1]/sum(action_count[1])], [action_count[0][0]/sum(action_count[0]), action_count[0][1]/sum(action_count[0])]))

    return actions, payoff_matrices, action_count, beliefs

def left_right():
    # actions
    actions = {0: "L", 1: "R"}
    # reward
    payoff_matrix_agent1 = np.array([[1 , 0], [0, 1]])
    payoff_matrix_agent2 = np.array([[1 , 0], [0, 1]])
    payoff_matrices = [payoff_matrix_agent1, payoff_matrix_agent2]
    # count of H and T played by each agent
    # for state 0 lets assume both have played Tails
    action_count = [[1,0], [0.5, 0.5]]

    beliefs = np.array(([action_count[1][0]/sum(action_count[1]), action_count[1][1]/sum(action_count[1])], [action_count[0][0]/sum(action_count[0]), action_count[0][1]/sum(action_count[0])]))

    return actions, payoff_matrices, action_count, beliefs

def main(game, advanced):
    print(game, advanced)
    game = int(game)
    if game == 2:
        actions, payoff_matrices, action_count, beliefs = prisoners_dillema_init()
        print('PLAYING PRISONERS DILLEMA')
    elif game == 3:
        actions, payoff_matrices, action_count, beliefs = battle_of_sexes_init()
        print('PLAYING BATTLE OF SEXES')
    elif game == 4:
        actions, payoff_matrices, action_count, beliefs = left_right()
        print('PLAYING LEFT-RIGHT')
    else:
        actions, payoff_matrices, action_count, beliefs = matching_pennies_init(advanced)
        print('PLAYING MATCHING PENNIES')


    max_iters = 100
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
        # keep beliefs and actions in each iteration for computing convergence
        strategies.append(beliefs.tolist())
        moves.append(best_actions)
        #moves.append((actions[np.argmax(beliefs[0])],actions[np.argmin(beliefs[1])] ))
        
    ### PLOT
    print("PLOTTING")
    moves = pd.DataFrame(moves, columns=["Action_agent1", "Action_agent2"])
    
    fig, (sub1, sub2) = plt.subplots(2,1)
    fig.subplots_adjust(hspace=0.5)
    sub1.set_ylim(-0.8, 1.2)
    sub2.set_ylim(-0.8, 1.2) 
    sub1.grid(True)
    sub2.grid(True)
    # for each agent plot its mixed strategy
    colors = ["g", "b", "r"]
    for i, subplot in enumerate([sub1, sub2]):
        probs = [poli[i] for poli in strategies]
        subplot.set(xlabel='Iterations', ylabel='Probability',title=f'Agent{i} empirical frequency of strategies, stages={len(probs) - 1}')
        print(len(probs[0]))
        for action in range(0, len(probs[0])):
            action_probs = [sublist[action] for sublist in probs]
            subplot.plot(list(range(0,len(probs))), action_probs, linewidth=1.5, label=f"s{action} = {actions[action]}", color=colors[action])
        subplot.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", help="choose which game to play. Range [1-3]")
    parser.add_argument("--adv", help="If game 1 chosen, seet this to 2 for rock-papers-scissors. Default macthing pennies")
    args = parser.parse_args()
    game = args.game
    advanced = args.adv
    # check if game option given, default matching pennies
    print(game, advanced)
    if game is None:
        game = 1
    if advanced is not None:
        print('ok')
        advanced = int(advanced)
        if advanced == 2:
            print('yy')
            advanced = True
        else:
            advanced = False
    else:
        advanced  = False
    main(game, advanced)