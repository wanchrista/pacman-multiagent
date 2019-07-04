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

    def evaluationFunction(self, currentGameState, action):
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

        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        # Ghost score helper
        def ghostScore(newPos, ghostPos):
            if manhattanDistance(newPos, ghostPos) > 0:
                # Assume Pacman would want to move away from the ghost - Make it weigh more
                return 10.0 / manhattanDistance(newPos, ghostPos)
            return 0

        # Food score helper
        def foodScore(newPos, newFood):
            if newFood.asList():
                distance = list()
                for food in newFood.asList():
                    distance.append(manhattanDistance(newPos, food))
                # Assume Pacman wants to move close to food, but make it weigh less in comparison if there is a ghost
                return 10.0 / min(distance)
            return 0

        return successorGameState.getScore() - ghostScore(newPos, newGhostStates[0].getPosition()) + foodScore(newPos, newFood)

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

        def minimax(gameState, depth, agent):
            if agent >= gameState.getNumAgents():
                agent = 0
                depth += 1

            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            if agent == self.index:
                n = ("unknown", -float("inf"))

                if not gameState.getLegalActions(agent):
                    return self.evaluationFunction(gameState)

                for move in gameState.getLegalActions(agent):
                    # Nothing to do here
                    if move == "Stop":
                        continue

                    maxi = minimax(gameState.generateSuccessor(agent, move), depth, agent + 1)
                    if type(maxi) is not float:
                        maxi = maxi[1]

                    if maxi > n[1]:
                        n = (move, maxi)

                return n
            else:
                n = ("unknown", float("inf"))

                if not gameState.getLegalActions(agent):
                    return self.evaluationFunction(gameState)

                for move in gameState.getLegalActions(agent):
                    # Nothing to do here
                    if move == "Stop":
                        continue

                    mini = minimax(gameState.generateSuccessor(agent, move), depth, agent + 1)
                    if type(mini) is not float:
                        mini = mini[1]

                    if mini < n[1]:
                        n = (move, mini)

                return n

        return minimax(gameState, 0, 0)[0]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def alphaBeta(gameState, depth, agent, alpha, beta):
            if agent >= gameState.getNumAgents():
                agent = 0
                depth += 1

            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            if agent == 0:
                n = ("unknown", -float("inf"))

                # If there are no more actions for PacMan, the game is over
                if not gameState.getLegalActions(agent):
                    return self.evaluationFunction(gameState)

                for move in gameState.getLegalActions(agent):
                    maxi = alphaBeta(gameState.generateSuccessor(agent, move), depth, agent + 1, alpha, beta)

                    if type(maxi) is not float:
                        maxi = maxi[1]

                    # Return immediately
                    if maxi >= beta:
                        return move, maxi
                    elif maxi > n[1]:
                        n = (move, maxi)
                        alpha = max(alpha, maxi)

                return n
            else:
                n = ("unknown", float("inf"))

                # If there are no more actions for ghost, the game is over
                if not gameState.getLegalActions(agent):
                    return self.evaluationFunction(gameState)

                for move in gameState.getLegalActions(agent):
                    mini = alphaBeta(gameState.generateSuccessor(agent, move), depth, agent + 1, alpha, beta)

                    if type(mini) is not float:
                        mini = mini[1]

                    # Return immediately
                    if mini <= alpha:
                        return move, mini
                    elif mini < n[1]:
                        n = [move, mini]
                        beta = min(beta, mini)

                return n

        return alphaBeta(gameState, 0, 0, -float("inf"), float("inf"))[0]


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

        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        def expected(gameState, agent, depth):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)

            actions = gameState.getLegalActions(agent)
            cost = 0

            for action in actions:
                successor = gameState.generateSuccessor(agent, action)
                if agent == gameState.getNumAgents() - 1:
                    cost += maxActions(successor, depth - 1)
                else:
                    cost += expected(successor, agent + 1, depth)

            return cost / len(actions)

        def maxActions(gameState, depth):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)

            actions = gameState.getLegalActions(0)
            cost = list()

            for action in actions:
                successor = gameState.generateSuccessor(0, action)
                cost.append(expected(successor, 1, depth))

            return max(cost)

        # Default action to do nothing
        maxAction = Directions.STOP
        maxCost = -(float("inf"))
        actions = gameState.getLegalActions(0)

        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            if expected(successor, 1, self.depth) > maxCost:
                maxCost = expected(successor, 1, self.depth)
                maxAction = action

        return maxAction

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: Weigh PacMan's actions based on the amount of score that he recieves when he completes the action.
      Give running towards ghosts that are near Pacman a negative score so that Pacman is less inclined to move towards
      the ghost (and therefore less likely to die). Edible ghosts give Pacman the most score, so give ghosts that
      Pacman is not scared of the most weight when evaluating.
    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()

    # Ghost score helper
    def ghostScore(newPos, ghostStates):
        ghostTotal = 0
        for ghost in ghostStates:
            distance = manhattanDistance(newPos, newGhostStates[0].getPosition())
            if distance > 0:
                if ghost.scaredTimer > 0:
                    ghostTotal += 100.0 / distance
                else:
                    ghostTotal -= 10.0 / distance
        return ghostTotal

    # Food score helper
    def foodScore(newPos, newFood):
        if newFood.asList():
            distance = list()
            for food in newFood.asList():
                distance.append(manhattanDistance(newPos, food))
            # Assume Pacman wants to move close to food, but make it weigh less in comparison if there is a ghost
            return 10.0 / min(distance)
        return 0

    return currentGameState.getScore() + ghostScore(newPos, newGhostStates) + foodScore(newPos, newFood)

# Abbreviation
better = betterEvaluationFunction

