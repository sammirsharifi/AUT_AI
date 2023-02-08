# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentgameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        successorgameState = currentgameState.generatePacmanSuccessor(action)
        newPos = successorgameState.getPacmanPosition()
        newFood = successorgameState.getFood()
        newGhostStates = successorgameState.getGhostStates()
        foodNum = currentgameState.getFood().count()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        point=0
        if len(newFood.asList()) == foodNum:
            point = 10000000
        for food in newFood.asList():
                point=min(manhattanDistance(food , newPos) , point)
        for ghost in newGhostStates:
            if manhattanDistance(ghost.configuration.pos,newPos) <= point :
                if manhattanDistance(ghost.configuration.pos,newPos)>=ghost.scaredTimer:
                    point += 10 ** (2 - manhattanDistance(ghost.configuration.pos, newPos))
        return -point    

        
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
    def value(self,agent, depth, gameState,alpha=None,beta=None,isExpectimax=False):
        
        if alpha is None  and  beta is None:
            if gameState.isLose() or gameState.isWin() or depth == self.depth:
                return scoreEvaluationFunction(gameState)
            if agent == 0: 
                if isExpectimax: return max(self.value(1, depth, gameState.generateSuccessor(agent, newState),isExpectimax=True) for newState in gameState.getLegalActions(agent))
                return max(self.value(1, depth, gameState.generateSuccessor(agent, newState)) for newState in gameState.getLegalActions(agent))
            else:  
                nextAgent = (agent + 1) %(gameState.getNumAgents()) 
                if nextAgent == 0:depth+=1
                if isExpectimax: return sum(self.value(nextAgent, depth, gameState.generateSuccessor(agent, newState),isExpectimax=True) for newState in gameState.getLegalActions(agent))
                return min(self.value(nextAgent, depth, gameState.generateSuccessor(agent, newState)) for newState in gameState.getLegalActions(agent))
        else:
            if gameState.isLose() or gameState.isWin() or depth == self.depth:
                return scoreEvaluationFunction(gameState)
            if agent==0:
                maxval=-100000
                for action in gameState.getLegalActions(agent):
                    maxval=max(maxval,self.value(1, depth, gameState.generateSuccessor(agent, action),alpha,beta))
                    if maxval > beta : return maxval
                    alpha=max(maxval,alpha)
                return maxval
            else:
                minval=100000
                nextAgent = (agent + 1) %(gameState.getNumAgents()) 
                if nextAgent == 0:depth+=1
                for newState in gameState.getLegalActions(agent):
                    minval=min(minval,self.value(nextAgent, depth, gameState.generateSuccessor(agent, newState),alpha,beta))
                    if minval<alpha: return minval
                    beta=min(beta,minval)
                return minval      

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        maximum_val = -1000000
        optimal_action = Directions.STOP
        for agentState in gameState.getLegalActions(0):
            value = self.value(1, 0, gameState.generateSuccessor(0, agentState))
            if value > maximum_val :
                maximum_val = value
                optimal_action = agentState

        return optimal_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        maxval = -100000
        action = Directions.STOP
        alpha = -100000
        beta = 100000
        for agentState in gameState.getLegalActions(0):
            ghostValue = self.value(1, 0, gameState.generateSuccessor(0, agentState), alpha, beta)
            if ghostValue > maxval:
                maxval = ghostValue
                action = agentState
            if maxval > beta:
                return maxval
            alpha = max(alpha, maxval)

        return action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        maximum_val = -100000
        optimal_action = Directions.STOP
        for agentState in gameState.getLegalActions(0):
            value = self.value(1, 0, gameState.generateSuccessor(0, agentState),isExpectimax=True)
            if value > maximum_val :
                maximum_val = value
                optimal_action = agentState

        return optimal_action

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***getCapsules()"
    """Calculating distance to the closest food pellet"""
    
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    foodNum = currentGameState.getFood().count()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    point=0
    if len(newFood.asList()) == foodNum:
            point = 10000000
    for food in newFood.asList():
                point=min(manhattanDistance(food , newPos) , point)
    for ghost in newGhostStates:
            if manhattanDistance(ghost.configuration.pos,newPos) <= point :
                if manhattanDistance(ghost.configuration.pos,newPos)>=ghost.scaredTimer:
                    point += 10 ** (2 - manhattanDistance(ghost.configuration.pos, newPos))
    return -point    
better = betterEvaluationFunction

