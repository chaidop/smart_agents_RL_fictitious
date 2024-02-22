import numpy as np
import random
from scipy.optimize import minimize
import argparse
import matplotlib.pyplot as plt
import pandas as pd

## --- GAMES ----------------------------
#
# 3 player:
# Player 1 wants to match the action of player 2; 
# player 2 wants to match the action of player 3; 
# player 3 wants to match the opposite of the action of player 1. 
# Each player receives a payoff of 1 if they match as desired, and −1 otherwise. 
# 1 unique Nash equilibrium, where all players uniformly randomize. 
# Jordan shows that this Nash equilibrium is locally unstable in a strong sense: for any ε > 0, 
# and for almost all initial empirical distributions that are within (Euclidean) distance ε 
# of the unique Nash equilibrium, DTFP does not converge to the NE
 
def matching_pennies_init(agents=2):
    
    actions = {0: "H", 1: "T"}
    payoff_matrix_agent1 = np.array([[1, -1], [-1, 1]])
    payoff_matrix_agent2 = np.array([[-1, 1], [1, -1]])
    payoff_matrices = [payoff_matrix_agent1, payoff_matrix_agent2]
    # count of H and T played by each agent
    # for state 0 lets assume both have played Tails
    action_count = [[0,1], [0,1]]
    #action_count = [[1.5, 2], [2, 1.5]]
    if agents == 1:
        action_count =  [[0.5, 0.5], [0.5, 0.5]]
    # initialise beliefs with mixed strategy of each agents
    # each row is the mixed strategy/ probability of all actions of the agent
    # empirical_distribution
    # P(a) = w(a)/SUM(w(a')), probability of opponent playing action a and a' all actions (includiing a)
    belief1 = []
    belief2 = []
    for action in range(len(actions)):
        belief1.append(action_count[1][action]/sum(action_count[1]))
        belief2.append(action_count[0][action]/sum(action_count[0]))
    beliefs = np.array((belief1, belief2))
    print(beliefs)
    

    return actions, payoff_matrices, action_count, beliefs

def rock_paper_scissor():
    
    actions = {0: "R", 1: "P", 2: "S"}
    payoff_matrice = np.array([[0,-1,1], [1,0,-1], [-1,1,0]])
    action_count = [[0,1,0], [0,0,1]]
    if agents == 1:
        action_count =  [[0.5, 0.5], [0.5, 0.5]]
    belief1 = []
    belief2 = []
    print(action_count[1])
    for action in range(len(actions)):
        belief1.append(action_count[1][action]/sum(action_count[1]))
        belief2.append(action_count[0][action]/sum(action_count[0]))
    beliefs = np.array((belief1, belief2))
    print(beliefs)
    

    return actions, payoff_matrice, action_count, beliefs

'''
If not zero-sum, then due to Shapley fictitious play not converge
At least 1 loss is -1 reward. 1 win is 0.5 and both wins is +1
Tie is 0
1 Nash (1/3, 1/3, 1/3)

P1=Rock
                         P3

                Rock    Paper   Scissors
             ----------------------------
    Rock     |   0   |   -1   |    0.5  |
             |--------------------------|
P2  Paper    |  -1   |   -1   |    0    |
             |--------------------------|
    Scissors |  0.5  |    0   |    2    |
             ----------------------------

P1=Paper
                         P3

                Rock    Paper   Scissors
             ----------------------------
    Rock     |   2   |   0.5  |    0    |
             |--------------------------|
P2  Paper    |  0.5  |    0   |   -1    |
             |--------------------------|
    Scissors |   0   |   -1   |   -1    |
             ----------------------------

P1=Scissors
                         P3

                Rock    Paper   Scissors
             ----------------------------
    Rock     |  -1   |    0   |   -1    |
             |--------------------------|
P2  Paper    |   0   |    2   |   0.5   |
             |--------------------------|
    Scissors |  -1   |   0.5  |    0    |
             ----------------------------
'''
def rock_paper_scissor_3_player():
    actions = {0: "R", 1: "P", 2: "S"}
    # payoff matrices are the same for each player
    payoff_matrix_agentR = np.array([[0, -1, 0.5], [-1, -1, 0], [0.5, 0, 2]])
    payoff_matrix_agentP = np.array([[2, 0.5, 0], [0.5, 0, -1], [0, -1, -1]])
    payoff_matrix_agentS = np.array([[-1, 0, -1], [0, 2, 0.5], [-1, 0.5, 0]])
    payoff_matrices = np.array((payoff_matrix_agentR, payoff_matrix_agentP, payoff_matrix_agentS))
    action_count = [[1000, 0, 0], [0, 1, 0], [0, 0, 1]]
    
    # each palyer keeps beliefs for every other opponent
    # 2 x 3 matrix
    belief1 = np.ones((2, 3))
    belief2 = np.ones((2, 3))
    belief3 = np.ones((2, 3))
    # belief for opponent number 1
    for action in range(len(actions)):
        belief1[0][action] = action_count[1][action]/sum(action_count[0])
        belief2[0][action] = action_count[0][action]/sum(action_count[0])
        belief3[0][action] = action_count[0][action]/sum(action_count[0])
    # belief for opponent numb 2
    for action in range(len(actions)):
            belief1[1][action] = action_count[2][action]/sum(action_count[1])
            belief2[1][action] = action_count[2][action]/sum(action_count[1])
            belief3[1][action] = action_count[1][action]/sum(action_count[1])
    beliefs = np.array((belief1, belief2, belief3))
    print(beliefs)
    

    return actions, payoff_matrices, payoff_matrix_agentR, payoff_matrix_agentP, payoff_matrix_agentS, action_count, beliefs, belief1, belief2, belief3
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
def prisoners_dillema_init(agents=2):
    actions = {0: "Silent", 1: "Betray"}
    payoff_matrix_agent1 = np.array([[-1, -3], [0, -2]])
    payoff_matrix_agent2 = np.array([[-1, -3], [0, -2]])
    payoff_matrices = [payoff_matrix_agent1, payoff_matrix_agent2]
    # count of H and T played by each agent
    # for state 0 lets assume both have played Tails
    action_count = [[1,0], [1,0]]
    ### if agents =2, (FP vs RL), take into consideration the 0.5 0.5 policy of the RL agent
    if agents == 1:
        action_count =  [[0.5, 0.5], [0.5, 0.5]]

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
def battle_of_sexes_init(agents=2):
    # actions
    actions = {0: "Opera", 1: "Movies"}
    # reward
    payoff_matrix_agent1 = np.array([[3, 0], [0, 2]])
    payoff_matrix_agent2 = np.array([[2, 0], [0, 3]])
    payoff_matrices = [payoff_matrix_agent1, payoff_matrix_agent2]
    # count of H and T played by each agent
    # for state 0 lets assume both have played Tails
    action_count = [[0,1], [1,0]]
    if agents == 1:
        action_count =  [[0.5, 0.5], [0.5, 0.5]]
    beliefs = np.array(([action_count[1][0]/sum(action_count[1]), action_count[1][1]/sum(action_count[1])], [action_count[0][0]/sum(action_count[0]), action_count[0][1]/sum(action_count[0])]))

    return actions, payoff_matrices, action_count, beliefs

def left_right(agents=2):
    # actions
    actions = {0: "L", 1: "R"}
    # reward
    payoff_matrix_agent1 = np.array([[1 , 0], [0, 1]])
    payoff_matrix_agent2 = np.array([[1 , 0], [0, 1]])
    payoff_matrices = [payoff_matrix_agent1, payoff_matrix_agent2]
    # count of H and T played by each agent
    # for state 0 lets assume both have played Tails
    action_count = [[1,0], [0.5, 0.5]]
    if agents == 1:
        action_count =  [[0.5, 0.5], [0.5, 0.5]]
    beliefs = np.array(([action_count[1][0]/sum(action_count[1]), action_count[1][1]/sum(action_count[1])], [action_count[0][0]/sum(action_count[0]), action_count[0][1]/sum(action_count[0])]))

    return actions, payoff_matrices, action_count, beliefs




#--------------------MINIMAX Q---------------------------------



payoffs = [[1, -1],
           [-1, 1]]

epsilon = 0.3
learning_rate = 1
gamma = 0.9

#Minimax Qlearning agent
class MinimaxAgent():
    def __init__(self, epsilon, gamma, player_id, payoffs, actions, learning_rate=1. ):
        self.epsilon = epsilon
        self.gamma = gamma
        self.player_id = player_id
        self.learning_rate = learning_rate
        # P <- 1/|A|
        self.P = np.array([1/len(actions), 1/len(actions)])
        # Q(s,a,o) <- 1
        self.Q = np.ones((2, 2))
        # V(s) <- 1
        self.V = 1

        self.actions = actions
        self.payoffs = np.array(payoffs) 
        

    def take_action(self):
        ## if probability epsilon, then take at random, else take action with policy distribution
        
        #if np.random.rand() < self.epsilon:  # Exploration: choose random action
        #    return np.random.choice(self.n_actions)
        #elif self.policy is not None:  # Use external policy if available
        #    action_probs = self.policy[state]
        #    return np.random.choice(np.arange(self.n_actions), p=action_probs)
        if random.choices([0, 1], weights=(self.epsilon * 100, (1 - self.epsilon) * 100))[0] == 0:
            return random.choice([0, 1])
        else:
            return random.choices([0, 1], weights=(self.P[0] * 100, self.P[1] * 100))[0]

    def learn(self, action, opponent):
        return self.payoffs[action][opponent]


    def update_Q(self, action, opponent, reward):
        #print(self.Q)
        #print(self.Q[action][opponent])
        #print((1 - self.learning_rate) * self.Q[action][opponent])
        #print(self.learning_rate * (reward + self.gamma * self.V))
        #print((1 - self.learning_rate) * self.Q[action][opponent] + self.learning_rate * (reward + self.gamma * self.V))
        self.Q[action][opponent] = (1 - self.learning_rate) * self.Q[action][opponent] + self.learning_rate * (reward + self.gamma * self.V)

    def update_P(self, opponent):
        bnds = ((0., 1.), (0., 1.))
        cons = ({'type': 'eq', 'fun': lambda x: 1.0 - np.sum(x)})


        if self.player_id == 1:
            f = lambda  x: min(np.matmul(x.T,self.Q))
        else:
            f = lambda  x: min(np.matmul(x.T,self.Q.T))

        self.P = minimize(fun=lambda x: -f(x), x0=np.array([0., 0.]), constraints=cons, bounds=bnds).x

    def update_V(self):

        if self.player_id == 1:
            f = lambda  x: min(np.matmul(x.T,self.Q))
        else:
            f = lambda  x: min(np.matmul(x.T,self.Q.T))

        self.V = f(self.P)

def update_prob():
    ## the probability of playing an action in each stage
    ## is calculated as 
    # (Qvalue of action)/(sum of Qvalues of all actions in cur state)
    return

#------------------FP-------------------------------------
        






def best_response(actions, exp_utilities):
    strat = np.argmax(exp_utilities)
    #print('The best action is ', actions[np.argmax(exp_utilities)])
    return strat

# for mixed strategies
def compute_expect_utility(actions, belief, payoff_matrix, advanced = False):
    # compute the explicit utility of an agent
    # for each action, based on the empirical distribution
    # of the actions of its opponents
    
    # if 3 player game, payoff_matrix is n matrices for the n actions 
    # the agent plays in contrast to the other 2 players
    if (advanced == True):
        exp_ut = []
        for action_matrix in payoff_matrix:
            sum = 0
            for row in range(len(actions)):
                for col in range(len(actions)):
                    sum += action_matrix[row][col]*belief[0][row]*belief[1][col]
            exp_ut.append(sum)
    else:
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
    
def update_beliefs(actions, action_count, advanced=False):
    #total = sum(action_count)
    new_belief = []
    
    if advanced == True:
        new_belief = np.ones((len(action_count), len(actions)))
        for i, belief_on_opponent in enumerate(action_count):
            for action in range(len(actions)):
                new_belief[i][action] = action_count[i][action]/sum(action_count[i])
    else:
        for action in range(len(actions)):
            new_belief.append(action_count[action]/sum(action_count))
    # return new belief
    new_belief = np.array(new_belief)
    return new_belief


### ----- MAIN FUNCTION


def main(game, agents):
    print(game, agents)
    game = int(game)
    if game == 2:
        actions, payoff_matrices, action_count, beliefs = prisoners_dillema_init(agents)
        print('PLAYING PRISONERS DILLEMA')
    elif game == 3:
        actions, payoff_matrices, action_count, beliefs = battle_of_sexes_init(agents)
        print('PLAYING BATTLE OF SEXES')
    elif game == 4:
        actions, payoff_matrices, action_count, beliefs = left_right(agents)
        print('PLAYING LEFT-RIGHT')
    elif game == 5:
        actions, payoff_matrices, payoff_matrix_agentR, payoff_matrix_agentP, payoff_matrix_agentS, action_count, beliefs, belief1, belief2, belief3 = rock_paper_scissor_3_player()
    elif game == 6:
        actions, payoff_matrices, action_count, beliefs =  rock_paper_scissor()
    else:
        actions, payoff_matrices, action_count, beliefs = matching_pennies_init(agents)
        print('PLAYING MATCHING PENNIES')


    max_iters = 1000
    
    agents = int(agents)
    if agents == 1:
        moves, strategies = test_FPvRL(actions, payoff_matrices, action_count, beliefs, max_iters)
        agent_type = ['RL Agent 1', 'FP Agent 2']
    elif agents == 2:
        if game == 5:
            moves, strategies = test_FPvFP_3_player(actions, payoff_matrices, action_count, beliefs, max_iters, three=True)
            agent_type = ['FP Agent 1', 'FP Agent 2', 'FP Agent 3']
        elif game == 6:
            print('hi')
            moves, strategies = test_FPvFP_3_player(actions, payoff_matrices, action_count, beliefs, max_iters)
            agent_type = ['FP Agent 1', 'FP Agent 2']
        else:
            moves, strategies = test_FPvFP(actions, payoff_matrices, action_count, beliefs, max_iters)
            agent_type = ['FP Agent 1', 'FP Agent 2']
    else:
        moves, strategies = test_RLvRL(actions, payoff_matrices, action_count, beliefs, max_iters)
        agent_type = ['RL Agent 1', 'RL Agent 2']
        
    
        
    ### PLOT
    print("PLOTTING")
    # you can prinnt below to see the moves played by each agents for each stage game
    if game == 5:
        moves = pd.DataFrame(moves, columns=["Action_agent1", "Action_agent2", "Action_agent3"])
    moves = pd.DataFrame(moves, columns=["Action_agent1", "Action_agent2"])
    if game == 5:
        fig, (sub1, sub2, sub3) = plt.subplots(3,1)
        sub3.set_ylim(-0.8, 1.2) 
        sub3.grid(True)
        subgraph_list = [sub1, sub2, sub3]
    else:
        fig, (sub1, sub2) = plt.subplots(2,1)
        subgraph_list = [sub1, sub2]
    fig.subplots_adjust(hspace=0.5)
    sub1.set_ylim(-0.8, 1.2)
    sub2.set_ylim(-0.8, 1.2) 
    sub1.grid(True)
    sub2.grid(True)
    # for each agent plot its mixed strategy
    colors = ["g", "b", "r"]
    for i, subplot in enumerate(subgraph_list):
        probs = [poli[i] for poli in strategies]
        subplot.set(xlabel='Iterations', ylabel='Probability',title=f'{agent_type[i]} empirical frequency of strategies, stages={len(probs) - 1}')
        print(len(probs[0]))
        for action in range(0, len(probs[0])):
            action_probs = [sublist[action] for sublist in probs]
            subplot.plot(list(range(0,len(probs))), action_probs, linewidth=1.5, label=f"s{action} = {actions[action]}", color=colors[action])
        subplot.legend()

    plt.tight_layout()
    plt.show()

### -------------------- DIFFERENT AGENTS TESTS
def test_FPvRL(actions, payoff_matrices, action_count, belief, max_iters):
    agent1 = MinimaxAgent(epsilon=0.3, learning_rate=1.0, gamma=0.9, player_id=1, payoffs = payoff_matrices[0], actions=actions)
    #make fp believe rl played 0.5 both actions
    strategies = [[agent1.P.tolist(), belief[0].tolist()]]
    curr_episode = 0
    moves = []
    for curr_episode in range(max_iters):
        # take action
        action1 = agent1.take_action()
        action2 = compute_expect_utility(actions, belief[1], payoff_matrices[1])

        # update
        rew1 = agent1.learn(action=action1, opponent=action2)
        action_count[0][action1] += 1
        action_count[1][action2] += 1
        # update FP agent's belief for the RL agent and save its own probabilities distribution
        belief[1]  = update_beliefs(actions, action_count[0])
        fp_probabilities  = update_beliefs(actions, action_count[1])

        agent1.update_Q(action=action1, opponent=action2, reward=rew1)
        agent1.update_P(opponent=action2)
        agent1.update_V()

        strategies.append([agent1.P,fp_probabilities.tolist()])
        moves.append([action1, action2])

        if curr_episode % 100 == 0:
            agent1.learning_rate *= 0.8

        curr_episode += 1

    return moves, strategies
# this func is for RPS game with FP vs FP. It is the only one supporting 3 players. If three =False, RPS is played with 2 players
def test_FPvFP_3_player(actions, payoff_matrices, action_count, beliefs, max_iters, three = False):
    print(beliefs[0])
    strategies = []
    if three == True:
        strategies.append([np.array(beliefs[1][0]), np.array(beliefs[0][0]), np.array(beliefs[0][1])])
    else:
        strategies.append([np.array(beliefs[1]), np.array(beliefs[0])])
    moves = []
    for i in range(max_iters):
        if three == True:
            # choose action by best response
            best_actions = [compute_expect_utility(actions, beliefs[0], payoff_matrices, advanced=True), compute_expect_utility(actions, beliefs[1], payoff_matrices, advanced=True), compute_expect_utility(actions, beliefs[2], payoff_matrices, advanced=True)]

            # choose action
            action_count[0][best_actions[0]] += 1
            action_count[1][best_actions[1]] += 1
            action_count[2][best_actions[2]] += 1

            # update their mixed strategy/beliefs based on their action
            beliefs[0] = update_beliefs(actions, [action_count[1], action_count[2]], advanced=True)
            beliefs[1]  = update_beliefs(actions, [action_count[0], action_count[2]], advanced=True)
            beliefs[2]  = update_beliefs(actions, [action_count[0], action_count[1]], advanced=True)
            # keep beliefs and actions in each iteration for computing convergence
            strategies.append([update_beliefs(actions, action_count[0]), update_beliefs(actions, action_count[1]), update_beliefs(actions, action_count[2])])
        else:
            best_actions = [compute_expect_utility(actions, beliefs[0], payoff_matrices), compute_expect_utility(actions, beliefs[1], payoff_matrices)]

            action_count[0][best_actions[0]] += 1
            action_count[1][best_actions[1]] += 1
            beliefs[0] = update_beliefs(actions, action_count[1])
            beliefs[1]  = update_beliefs(actions, action_count[0])
            strategies.append([update_beliefs(actions, action_count[0]), update_beliefs(actions, action_count[1])])

        moves.append(best_actions)
    '''
    import pandas as pd
    empirical_freq_means = pd.DataFrame(moves, columns=["A", "B", "C"]).expanding().mean()
    print(empirical_freq_means)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    empirical_freq_means.plot(ax=ax)
    import matplotlib.ticker as ticker
    import pandas as pd
    from tqdm import tqdm
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.axhline(0.5, alpha=0.5, color="red", linewidth=1)
    ax.legend(labels=["A's empirical frequency", "B's empirical frequency", "C's empirical frequency"])

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Empirical frequency")
    '''
    return moves, strategies
def test_FPvFP(actions, payoff_matrices, action_count, beliefs, max_iters):
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
    return moves, strategies

def test_RLvRL(actions, payoff_matrices, action_count, beliefs, max_iter):    
    agent1 = MinimaxAgent(epsilon=0.3, learning_rate=1.0, gamma=0.9, player_id=1, payoffs = payoff_matrices[0], actions=actions)
    agent2 = MinimaxAgent(epsilon=0.3, learning_rate=1.0, gamma=0.9, player_id=2, payoffs = payoff_matrices[1], actions=actions)

    strategies = [[agent1.P,agent2.P]]
    curr_episode = 0
    moves = []
    for curr_episode in range(max_iter):
        action1 = agent1.take_action()
        action2 = agent2.take_action()

        rew1 = agent1.learn(action=action1, opponent=action2)
        rew2 = agent2.learn(action=action2, opponent=action1)

        agent1.update_Q(action=action1, opponent=action2, reward=rew1)
        agent1.update_P(opponent=action2)
        agent1.update_V()

        agent2.update_Q(action=action2, opponent=action1, reward=rew2)
        agent2.update_P(opponent=action1)
        agent2.update_V()

        strategies.append([agent1.P,agent2.P])
        moves.append([action1, action2])

        if curr_episode % 50 == 0:
            agent1.learning_rate *= 0.8
            agent2.learning_rate *= 0.8

        curr_episode += 1

    ### OUTCOME
    print("Agent's 1 Policy:")
    print(agent1.P)

    print("Agent's 2 Policy:")
    print(agent2.P)

    print("V:")
    print(agent1.V*0.1, agent2.V*0.1)
    print(agent2.V*0.1 + agent1.V*0.1)

    return moves, strategies

### --------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", help="choose which game to play. Range [1-3]")
    parser.add_argument("--adv", help="If game 1 chosen, seet this to 2 for rock-papers-scissors. Default macthing pennies")
    parser.add_argument("--agents", help="Choose what agents to test. 1: FP vs RL, 2: FP vs FP 3: RL vs RL")
    args = parser.parse_args()
    game = args.game
    agents = args.agents
    # check if game option given, default matching pennies
    print(game)
    if game is None:
        game = 1
    if agents is None:
        agents = 1
    ## for debugging
    #game = 6
    #agents = 2
    main(game, agents)