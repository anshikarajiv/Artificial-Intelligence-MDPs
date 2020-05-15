import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        for _ in range(0, self.iterations):
            tem = util.Counter()
            for state in self.mdp.getStates():
                if not self.mdp.isTerminal(state):
                    y = []
                    for a in self.mdp.getPossibleActions(state):
                        y.append(self.getQValue(state, a))
                    maxi = max(y)
                    tem[state] = maxi
            self.values = tem

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        q = 0
        if not self.mdp.isTerminal(state):
            for t in self.mdp.getTransitionStatesAndProbs(state, action):
                r = self.mdp.getReward(state, action, t[0])
                q += t[1] * (r + (self.discount * self.values[t[0]]))
        return q

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if not self.mdp.isTerminal(state):
            q, Action = None, None
            for a in self.mdp.getPossibleActions(state):
                q_t = self.getQValue(state, a)
                if q == None or q_t > q:
                    Action = a
                    q = q_t
            return Action
        return None

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
