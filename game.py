from types import SimpleNamespace
from players import RandomPlayer


def play(state, player1=None, player2=None):
    players = {1: player1 or RandomPlayer(),
               -1: player2 or RandomPlayer()}
    states = []
    while not state.gameover():
        player = players[state.player()]
        action = player.next_action(state)
        state = state.move(action)
        states.append(SimpleNamespace(action=action, state=state))
    return states, state.winner()


def simulate(state, play_count, player1=None, player2=None):
    import click
    plays = []
    label = f'Simulating {play_count} games...'
    with click.progressbar(label=label, length=play_count) as bar:
        for _ in range(play_count):
            states, winner = play(state, player1, player2)
            plays.append((states, winner))
            bar.update(1)
    return plays
