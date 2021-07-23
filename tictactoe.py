import click

from game import play, simulate
from players import ConsolePlayer, MiniMaxPlayer, NNPlayer, ReinforcementPlayer, RandomPlayer, MiniMaxPlayerNN, \
    MiniMaxPlayerNN2
from tictactoe_state import TicTacToeState


@click.command('connect4')
@click.option('--simulations', '-s',
              default=5000,
              help='How many plays to simulate for training.')
@click.option('--mode', '-m',
              default='reinforcement',
              type=click.Choice(['console', 'window', 'simulation', 'reinforcement']),
              help='Starts game in a terminal or a window.')
@click.option('--ai-player', '-p',
              default='cnn2',
              type=click.Choice(['nn', 'minimax', 'cnn', 'cnn2']),
              help='Type of ai player')
@click.option('--epochs', '-e',
              default=3,
              help='Number of epochs to train. (only for ai-player=nn)')
@click.option('--lookahead', '-l',
              default=1,
              help='Lookahead depth for the minimax algorithm.'
                   ' (only for ai-player=minimax)')
def tictactoe(simulations, mode, ai_player, epochs, lookahead):
    state = TicTacToeState()

    if ai_player == 'nn':
        from tictactoe_model import TicTacToeModel
        model = TicTacToeModel()
        plays = simulate(state, simulations)
        model.train(plays, epochs=epochs)
        autoplayer = NNPlayer(model, 'DNN')
    elif ai_player == 'cnn':
        from tictactoe_model_cnn import TicTacToeModelCnn
        model = TicTacToeModelCnn()
        plays = simulate(state, simulations)
        model.train(plays, epochs=epochs)
        autoplayer = NNPlayer(model, 'CNN')
    elif ai_player == 'cnn2':
        from tictactoe_model_cnn2 import TicTacToeModelCnn2
        model = TicTacToeModelCnn2()
        plays = simulate(state, simulations, player1=RandomPlayer(), player2=RandomPlayer())
        model.train(plays, epochs=epochs)
        autoplayer = ReinforcementPlayer(model, 'CNN2')
    else:
        autoplayer = MiniMaxPlayer(lookahead=lookahead)

    if mode == 'console':
        states, _ = play(state, ConsolePlayer(), autoplayer)
        print(states[-1].state)
    elif mode == 'simulation':
        player1 = autoplayer

        # from tictactoe_model_cnn import TicTacToeModelCnn
        # cnn_model = TicTacToeModelCnn()
        # cnn_plays = simulate(state, simulations)
        # cnn_model.train(cnn_plays, epochs=epochs)
        # player2 = NNPlayer(cnn_model, 'CNN')
        player2 = RandomPlayer()

        plays = simulate(state, 50, player1=player1, player2=player2)
        player1_wins, player2_wins, draws = game_statistics(plays)

        plays = simulate(state, 50, player1=player2, player2=player1)
        p2, p1, d = game_statistics(plays)
        player1_wins += p1
        player2_wins += p2
        draws += d

        print(f'{player1} vs. {player2}')
        print(f'{player1} wins: {player1_wins}')
        print(f'{player2} wins: {player2_wins}')
        print(f'Draws: {draws}')
    elif mode == 'reinforcement':
        # initial training with random plays (not really necessary)
        from tictactoe_model import TicTacToeModel
        model = TicTacToeModel()
        plays = simulate(state, simulations, player1=RandomPlayer(), player2=RandomPlayer())
        model.train(plays, epochs=epochs)
        nnplayer = NNPlayer(model, 'DNN')
        autoplayer = MiniMaxPlayer(1)

        player1 = autoplayer
        # player2 = RandomPlayer()

        for _ in range(10):
            # training through self-play
            # plays = simulate(state, 100, player1=player1, player2=player1)
            # print_game(plays[0])
            # print_game(plays[1])
            # model.train(plays, epochs=epochs)

            # benchmark vs random
            duel_random_tictactoe(player1)

    else:
        from tictactoe_window import TicTacToeWindow
        state = TicTacToeState()
        state = state.move(autoplayer.next_action(state))
        TicTacToeWindow(autoplayer=autoplayer, state=state).show()


def duel_random_tictactoe(player1):
    player2 = MiniMaxPlayer(2)
    state = TicTacToeState()
    duel(player1, player2, state)


def duel(player1, player2, state):
    plays = simulate(state, 50, player1=player1, player2=player2)
    print_game(plays[0])

    player1_wins, player2_wins, draws = game_statistics(plays)
    plays = simulate(state, 50, player1=player2, player2=player1)
    print_game(plays[0])

    p2, p1, d = game_statistics(plays)
    player1_wins += p1
    player2_wins += p2
    draws += d
    print(f'{player1} vs. {player2}')
    print(f'{player1} wins: {player1_wins}')
    print(f'{player2} wins: {player2_wins}')
    print(f'Draws: {draws}')


def game_statistics(plays):
    player1_wins = [p[1] for p in plays].count(1)
    player2_wins = [p[1] for p in plays].count(-1)
    draws = [p[1] for p in plays].count(0)

    return player1_wins, player2_wins, draws


def print_game(play):
    for state in play[0]:
        print(state, end=', ')
    print()


if __name__ == '__main__':
    tictactoe()
