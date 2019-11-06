"""
This is the only file you should change in your submission!
"""
from basicplayer import basic_evaluate, minimax, get_all_next_moves, is_terminal
from util import memoize, run_search_function, INFINITY, NEG_INFINITY

STUDENT_ID = 20536826
AGENT_NAME = "ayylmao"
COMPETE = True


def focused_evaluate(board):
    """
    Given a board, return a numeric rating of how good
    that board is for the current player.
    A return value >= 1000 means that the current player has won;
    a return value <= -1000 means that the current player has lost
    """
    if board.is_win() == board.get_current_player_id():
        return 1100 - board.num_tokens_on_board()
    elif board.is_win() == board.get_other_player_id():
        return -1100 + board.num_tokens_on_board()

    score = 0
    score += board.longest_chain(board.get_current_player_id()) * 10
    # Prefer having your pieces in the center of the board.
    for row in range(6):
        for col in range(7):
            if board.get_cell(row, col) == board.get_current_player_id():
                score -= abs(3 - col)
            elif board.get_cell(row, col) == board.get_other_player_id():
                score += abs(3 - col)
    return score


# Create a "player" function that uses the focused_evaluate function
# You can test this player by choosing 'quick' in the main program.
quick_to_win_player = lambda board: minimax(board, depth=4, eval_fn=focused_evaluate)


def alpha_beta_search_find_board_value(
    board,
    depth,
    eval_fn,
    alpha,
    beta,
    get_next_moves_fn=get_all_next_moves,
    is_terminal_fn=is_terminal,
):
    """
    Alpha-beta helper function: Return the minimax value of a particular board,
    given a particular depth to estimate to, pruning if available.
    For the MIN levels, in order to minimize the score, we maximize the negative,
    which is why it flips the sign if its a MIN node. 
    """
    if is_terminal_fn(depth, board):
        # Rating: eval_fn(board)
        # Move: None
        return eval_fn(board), None

    # Rating: -INF
    # Move: None
    best_val = NEG_INFINITY, None

    for move, new_board in get_next_moves_fn(board):
        # Note: beta = -alpha => beta = INF
        #       alpha = -beta => alpha = -INF
        val = (
            -1
            * alpha_beta_search_find_board_value(
                new_board,
                depth - 1,
                eval_fn,
                -beta,
                -alpha,
                get_next_moves_fn,
                is_terminal_fn,
            )[0]
        )
        # Attempt to maximize children
        if val > best_val[0]:
            best_val = val, move
        # Attempt to maximize alpha value
        alpha = max(alpha, val)
        # Prune if alpha >= beta
        if alpha >= beta:
            return best_val

    return best_val


def alpha_beta_search(
    board,
    depth,
    eval_fn,
    get_next_moves_fn=get_all_next_moves,
    is_terminal_fn=is_terminal,
):
    """
    Do a alpha-beta search to the specified depth on the specified board.

    board -- the ConnectFourBoard instance to evaluate
    depth -- the depth of the search tree (measured in maximum distance from a leaf to the root)
    eval_fn -- the evaluation function to use to give a value to a leaf of the tree

    Returns an integer, the column number of the column that the search determines you should add a token to
    """
    _, move = alpha_beta_search_find_board_value(
        board, depth, eval_fn, NEG_INFINITY, INFINITY, get_next_moves_fn, is_terminal_fn
    )
    return move


def alpha_beta_player(board):
    return alpha_beta_search(board, depth=8, eval_fn=focused_evaluate)


def ab_iterative_player(board):
    return run_search_function(
        board, search_fn=alpha_beta_search, eval_fn=focused_evaluate, timeout=5
    )


def better_evaluate(board):
    """
    Given a board, return a numeric rating of how good
    that board is for the current player.
    A return value >= 1000 means that the current player has won;
    a return value <= -1000 means that the current player has lost
    
    Idea: Prefer having contiguous pieces in the center of the board.
          If both players have equivalent amount of chains
          in the middle of the board, we count this as being favorable
          towards the opponent as we want to play defensively
          and prevent the other player from creating longer chains
          or placing pieces in the center of the board.
    """
    current_player = board.get_current_player_id()
    other_player = board.get_other_player_id()

    if board.is_win() == current_player:
        return 1100 - board.num_tokens_on_board()
    elif board.is_win() == other_player:
        return -1100 + board.num_tokens_on_board()
    elif board.is_tie():
        return 0
    score = 0

    for chain in board.chain_cells(other_player):
        # Prefer having pieces in contiguous chains
        score -= len(chain) ** 2
        for cell in chain:
            # Prefer having pieces in the center of the board.
            score += 5 * abs(3 - cell[1])

    for chain in board.chain_cells(current_player):
        # Prefer having pieces in contiguous chains
        score += len(chain) ** 1.5
        for cell in chain:
            # Prefer having pieces in the center of the board.
            score -= 3 * abs(3 - cell[1])
    return score


better_evaluate = memoize(better_evaluate)


# A player that uses alpha-beta and better_evaluate:
def my_player(board):
    return run_search_function(
        board, search_fn=alpha_beta_search, eval_fn=better_evaluate, timeout=5
    )


# my_player = lambda board: alpha_beta_search(board, depth=4, eval_fn=better_evaluate)
