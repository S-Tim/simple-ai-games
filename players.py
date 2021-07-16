from abc import ABC, abstractmethod
from random import randint

from numpy import argmax


class Player(ABC):
    def __init__(self, name=''):
        self.name = name

    @abstractmethod
    def next_action(self, state):
        ...

    def __str__(self):
        return self.name


class IndecisivePlayer(Player):
    def __init__(self, name=''):
        super().__init__(name)
        self.name = name

    def next_action(self, state):
        moves = self.next_actions(state)
        return moves[randint(0, len(moves) - 1)]

    @abstractmethod
    def next_actions(self, state):
        ...


class RandomPlayer(IndecisivePlayer):
    def next_actions(self, state):
        return state.actions()


class ConsolePlayer(Player):

    def next_action(self, state):
        actions = state.actions()
        while True:
            try:
                print(state)
                action = input(f'Action [{actions}]: ')
                action = int(action)
                if action in actions:
                    return action
            except Exception:
                pass


class MiniMaxPlayer(IndecisivePlayer):
    def __init__(self, lookahead):
        super().__init__('MiniMax')
        assert lookahead > 0
        self.lookahead = lookahead

    def next_actions(self, state):
        values = [(action, self.minimax(state.move(action), self.lookahead)) for action in state.actions()]

        behavior = max if state.player() == 1 else min
        best_value = behavior(values, key=lambda a: a[1])[1]
        best_moves = filter(lambda x: x[1] == best_value, values)

        return list(map(lambda x: x[0], best_moves))

    def minimax(self, state, lookahead):
        if state.gameover() or lookahead == 0:
            # return 1 or -1 if someone won, else 0
            return state.winner()

        # maximizing
        # state.player() is the next player to move but here we evaluate the possible states when we would move
        if state.player() == 1:
            max_eval = -1
            evaluations = [self.minimax(state.move(action), lookahead - 1) for action in state.actions()]
            max_eval = max(max_eval, *evaluations)
            return max_eval
        else:
            # minimizing
            min_eval = 1
            evaluations = [self.minimax(state.move(action), lookahead - 1) for action in state.actions()]
            min_eval = min(min_eval, *evaluations)
            return min_eval


class NNPlayer(Player):
    def __init__(self, model, name=''):
        super().__init__(name)
        self.model = model

    def next_action(self, state):
        actions = state.actions()
        current_player = state.player()
        states = [state.move(action).cells for action in actions]
        probs = self.model.predict(states)
        player_probs = [p[current_player] for p in probs]
        return actions[argmax(player_probs)]


class ReinforcementPlayer(Player):
    def __init__(self, model, name=''):
        super().__init__(name)
        self.model = model

    def next_action(self, state):
        actions = state.actions()
        states = [state.move(action).cells for action in actions]
        rewards = self.model.predict(states)
        behavior = max if state.player() == 1 else min
        moves = zip(actions, rewards)
        best_move = behavior(moves, key=lambda move: move[1])
        return best_move[0]
