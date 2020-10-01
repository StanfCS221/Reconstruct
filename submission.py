from util import manhattanDistance
from game import Directions
import random, util
from timeit import default_timer as timer
from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """
    def __init__(self):
        self.lastPositions = []
        self.dc = None


    def getAction(self, gameState):
        """
        getAction chooses among the best options according to the evaluation function.

        getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
        ------------------------------------------------------------------------------
        Description of GameState and helper functions:

        A GameState specifies the full game state, including the food, capsules,
        agent configurations and score changes. In this function, the |gameState| argument
        is an object of GameState class. Following are a few of the helper methods that you
        can use to query a GameState object to gather information about the present state
        of Pac-Man, the ghosts and the maze.

        gameState.getLegalActions():
            Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

        gameState.generateSuccessor(agentIndex, action):
            Returns the successor state after the specified agent takes the action.
            Pac-Man is always agent 0.

        gameState.getPacmanState():
            Returns an AgentState object for pacman (in game.py)
            state.configuration.pos gives the current position
            state.direction gives the travel vector

        gameState.getGhostStates():
            Returns list of AgentState objects for the ghosts

        gameState.getNumAgents():
            Returns the total number of agents in the game

        gameState.getScore():
            Returns the score corresponding to the current state of the game


        The GameState class is defined in pacman.py and you might want to look into that for
        other helper methods, though you don't need to.
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best


        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (oldFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

######################################################################################
# Problem 1b: implementing minimax
#file = open('Tree.txt', 'w')
class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (problem 1)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction. Terminal states can be found by one of the following:
          pacman won, pacman lost or there are no legal moves.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          Directions.STOP:
            The stop direction, which is always legal

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.getScore():
            Returns the score corresponding to the current state of the game

          gameState.isWin():
            Returns True if it's a winning state

          gameState.isLose():
            Returns True if it's a losing state

          self.depth:
            The depth to which search should continue

        """
        #start = timer()
        # BEGIN_YOUR_CODE (our solution is 20 lines of code, but don't worry if you deviate from this)
        def recurse(state, depth, turn):
            if depth == 1:
                return self.evaluationFunction(state), Directions.STOP
            if state.isWin():
                return float('inf'), Directions.STOP
            if state.isLose():
                return -1 * float('inf'), Directions.STOP

            if turn == 0:
                # file.write('PLAYER VALUE\n')
                choices = [(recurse(state.generateSuccessor(0, action), depth - 1, state.getNumAgents() - 1)[0], action)
                           for action in state.getLegalActions(0)]
                return max(choices)

            else:
                # file.write('GHOST VALUE\n')
                choices = [(recurse(state.generateSuccessor(turn, action), depth - 1, turn - 1)[0], action) for action
                           in state.getLegalActions(turn)]
                return min(choices)

        value, action = recurse(gameState, self.depth * gameState.getNumAgents(), 0)
        # file.write('One Iteration FInished\n\n\n')
        # print(value)

        #end = timer()
        #print('Time needed for move = ' + str(end - start))
        return action
        # END_YOUR_CODE

######################################################################################
# Problem 2a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (problem 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        #start = timer()
        # BEGIN_YOUR_CODE (our solution is 36 lines of code, but don't worry if you deviate from this)
        def recurse(state, depth, turn, alpha, beta):
            if depth == 1:
                print('Vraceno iz d=1 za turn' + str(turn))
                return self.evaluationFunction(state), Directions.STOP
            if state.isWin():
                print('Vraceno iz win za turn' + str(turn))
                #return float('inf'), Directions.STOP
                return 999999, Directions.STOP
            if state.isLose():
                print('Vraceno iz lose za turn' + str(turn))
                #return -1 * float('inf'), Directions.STOP
                return -999999, Directions.STOP

            if turn == 0:
                changed = False
                bestValue = -float('inf')
                bestAction = Directions.STOP
                print(len(state.getLegalActions(0)))
                for action in state.getLegalActions(0):
                    value, _ = recurse(state.generateSuccessor(0,action), depth-1, state.getNumAgents()-1, alpha, beta)
                    print(value)
                    if bestValue < value:
                        changed = True
                        bestValue = value
                        bestAction = action
                    alpha = max(alpha, bestValue)
                    if beta < alpha:
                        break
                print('Best value = ' + str(bestValue))
                print('Broj akcija je: ' + str(len(state.getLegalActions(0))))
                print('Changed: ' + str(changed))
                if bestAction == Directions.STOP:
                    print('Vraceno iz turn0')
                return bestValue, bestAction

            else:
                bestValue = float('inf')
                bestAction = Directions.STOP
                for action in state.getLegalActions(turn):
                    value, _ = recurse(state.generateSuccessor(turn,action), depth-1, turn-1, alpha, beta)
                    if bestValue > value:
                        bestValue = value
                        bestAction = action
                    beta = min(beta, bestValue)
                    if beta < alpha:
                        break
                if bestAction == Directions.STOP:
                    print('Vraceno iz ghost turn' + str(turn))
                return bestValue, bestAction

        value, action = recurse(gameState, self.depth*gameState.getNumAgents(), 0, -1*float('inf'), float('inf'))
        #print(value)
        #end = timer()
        #print('Time needed for move = ' + str(end-start))
        return action
        # END_YOUR_CODE

######################################################################################
# Problem 3b: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (problem 3)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """

        # BEGIN_YOUR_CODE (our solution is 20 lines of code, but don't worry if you deviate from this)
        #start = timer()
        def recurse(state, depth, turn):
            if depth == 1:
                return self.evaluationFunction(state), Directions.STOP
            if state.isWin():
                return float('inf'), Directions.STOP
            if state.isLose():
                return -1 * float('inf'), Directions.STOP

            if turn == 0:
                choices = [(recurse(state.generateSuccessor(0, action), depth - 1, state.getNumAgents() - 1)[0], action)
                           for action in state.getLegalActions(0)]
                return max(choices)

            else:
                actions = state.getLegalActions(turn)
                if len(actions) > 0:
                    probability = 1 / len(actions)
                else:
                    return 0, Directions.STOP
                sumV = probability * sum(
                    recurse(state.generateSuccessor(turn, action), depth - 1, turn - 1)[0] for action in actions)
                return sumV, random.choice(actions)

        value, action = recurse(gameState, self.depth * gameState.getNumAgents(), 0)
        # file.write('One Iteration FInished\n\n\n')
        # print(value)

        #end = timer()
        #print('Time needed for move = ' + str(end - start))
        return action
        # END_YOUR_CODE

######################################################################################
# Problem 4a (extra credit): creating a better evaluation function

def betterEvaluationFunction(currentGameState):
    """
      Your extreme, unstoppable evaluation function (problem 4).

      DESCRIPTION: <write something here so we know what you did>
    """

    # BEGIN_YOUR_CODE (our solution is 13 lines of code, but don't worry if you deviate from this)
    score = 0
    score += currentGameState.getScore()

    # END_YOUR_CODE

# Abbreviation
better = betterEvaluationFunction
