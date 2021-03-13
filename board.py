from typing import List
import numpy as np

class game():
  layer_map = {
    0: 'P', 2: 'R', 4: 'N', 6: 'B', 8: 'Q', 10: 'K', # Even = white
    1: 'p', 3: 'r', 5: 'n', 7: 'b', 9: 'q', 11: 'k', # Odd = black
  }
  inverse_layer_map = { v: k for k, v in layer_map.items() }
  default_fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'

  def __init__(self, fen: str = None):
    self.is_game_over = False
    self._white_to_move = True
    self._board_state = np.zeros((8, 8, 12), dtype=np.uint8)

    # If a player is in single-check, moving a piece to any square with 1 will stop the check.
    self._check_line = np.zeros((8, 8), dtype=np.uint8)

    self._board_state = np.zeros((8, 8, 12), dtype=np.uint8)

    self._is_in_check_white = False
    self._is_in_double_check_white = False
    self._is_in_check_black = False
    self._is_in_double_check_black = False

    # FEN-related parameters
    self._can_castle_short_white = True
    self._can_castle_long_white = True
    self._can_castle_short_black = True
    self._can_castle_long_black = True
    self._en_passant_target = '-'
    self._halfmove_clock = 0

    self._move_history: List[str] = list()

    self.reset_game(fen)

  @property
  def move_history(self) -> List[str]:
    return self._move_history

  @property
  def fen(self) -> str:
    fen_rows: List[str] = []
    board = self._get_layer_representation()
    for row in range(8):
      square_values: List[str] = []
      empty_count = 0
      for col in range(8):
        if board[row, col] == -1:
          empty_count += 1
        else:
          if empty_count > 0:
            square_values.append(str(empty_count))
            empty_count = 0
          square_values.append(self.layer_map.get(board[row, col], ''))
      if empty_count > 0:
        square_values.append(str(empty_count))

      fen_rows.append(''.join(square_values))
    position = '/'.join(fen_rows)

    active_color = 'w' if self._white_to_move else 'b'

    castling = 'K' if self._can_castle_short_white else ''
    castling += 'Q' if self._can_castle_long_white else ''
    castling += 'k' if self._can_castle_short_black else ''
    castling += 'q' if self._can_castle_long_black else ''
    if len(castling) == 0:
      castling = '-'

    en_passant_target = self._en_passant_target
    halfmove_clock = self._halfmove_clock
    fullmove_clock = len(self._move_history) // 2 + 1

    return f'{position} {active_color} {castling} {en_passant_target} {halfmove_clock} {fullmove_clock}'

  def reset_game(self, fen: str = None) -> None:
    self._set_board(fen)
    self._set_flags(fen)

    self._move_history = list()

  def get_legal_moves(self):
    legal_moves = list()
    board = self._get_layer_representation()

    # Rooks
    layer = 2 + (0 if self._white_to_move else 1)
    ones = np.where(self._board_state[:, :, layer] == 1)
    rook_indices = [[row, col] for row, col in zip(ones[0], ones[1])]

    for rook in rook_indices:
      legal_moves += self._get_lateral_moves(rook[0], rook[1], board)

    # Bishops
    layer = 6 + (0 if self._white_to_move else 1)
    ones = np.where(self._board_state[:, :, layer] == 1)
    bishop_indices = [[row, col] for row, col in zip(ones[0], ones[1])]

    for bishop in bishop_indices:
      legal_moves += self._get_diagonal_moves(bishop[0], bishop[1], board)

    # Queens
    layer = 8 + (0 if self._white_to_move else 1)
    ones = np.where(self._board_state[:, :, layer] == 1)
    queen_indices = [[row, col] for row, col in zip(ones[0], ones[1])]

    for queen in queen_indices:
      legal_moves += self._get_lateral_moves(queen[0], queen[1], board)
      legal_moves += self._get_diagonal_moves(queen[0], queen[1], board)

    return legal_moves

  @staticmethod
  def _check_move(r, dr, c, dc, b, par):
    """ Checks whether a move is allowed.

        Args:
          r: Row of the piece in question
          dr: How many rows over to check
          c: Column of the piece in question
          dc: How many columns over to check
          b: 2D representation of the board. -1 for empty square, and check the layer_map property
             for piece-to-int conversion
          par: Parity of the piece's color. 0 if the piece is white, 1 if black

        Returns:
          blocked: Boolean describing if the remainder of the line in question is blocked by a piece
                   or board edge
          move: List representing the start square and end square of a legal move. Returns None if
                no legal move is possible
          is_check: Boolean describing whether this move creates check on the opposing player
    """
    if (0 <= r + dr <= 7) and (0 <= c + dc <= 7):
      # Square is vacant
      if b[r + dr, c + dc] == -1:
        return (False, [[r, c, int(b[r, c])], [r + dr, c + dc, int(b[r, c])]], False)

      # Square is not vacant, but occupied by an opposing piece
      elif b[r + dr, c + dc] % 2 != par:
        is_king = (b[r + dr, c + dc] == 10 + par)
        return (True, [[r, c, int(b[r, c])], [r + dr, c + dc, int(b[r, c])]], is_king)

      # Square is not vacant, and occupied by a same-color piece
      else:
        return (True, None, False)
    else:
      return (True, None, False)

  def _get_diagonal_moves(self, row, col, board) -> List[List[List[int]]]:
    moves: List[List[List[int]]] = list()
    parity = board[row, col] % 2 # parity is 0 if the piece we're moving is white, and 1 if black.

    # If the player is in double-check, the king has to move, so there's no point in checking moves
    if not (self._is_in_double_check_white if parity == 0 else self._is_in_double_check_black):
      return moves

    directions = {
      'pp': {'blocked': False, 'r':  1, 'c':  1},
      'pm': {'blocked': False, 'r':  1, 'c': -1},
      'mp': {'blocked': False, 'r': -1, 'c':  1},
      'mm': {'blocked': False, 'r': -1, 'c': -1},
    }

    for n in range(1, 8):
      for direction, values in directions.items():
        if not values['blocked']:
          move_check = self._check_move(row, n * values['r'], col, n * values['c'], board, parity)
          directions[direction]['blocked'], move, check = move_check

          if move is not None:
            moves.append(move)

            if check:
              if self._white_to_move:
                self._is_in_check_black = True
              else:
                self._is_in_check_white = True

    return moves

  def _get_lateral_moves(self, row, col, board) -> List[List[List[int]]]:
    moves: List[List[List[int]]] = list()
    parity = board[row, col] % 2 # parity is 0 if the piece we're moving is white, and 1 if black.

    # If the player is in double-check, the king has to move, so there's no point in checking moves
    if not (self._is_in_double_check_white if parity == 0 else self._is_in_double_check_black):
      return moves

    directions = {
      'up':    {'blocked': False, 'r': -1, 'c':  0},
      'down':  {'blocked': False, 'r':  1, 'c':  0},
      'left':  {'blocked': False, 'r':  0, 'c': -1},
      'right': {'blocked': False, 'r':  0, 'c':  1},
    }

    for n in range(1, 8):
      for direction, values in directions.items():
        if not values['blocked']:
          move_check = self._check_move(row, n * values['r'], col, n * values['c'], board, parity)
          directions[direction]['blocked'], move, check = move_check

          if move is not None:
            moves.append(move)

            if check:
              if self._white_to_move:
                self._is_in_check_black = True
              else:
                self._is_in_check_white = True

    return moves

  def _set_board(self, fen: str = None) -> None:
    fen = fen if fen is not None else self.default_fen
    position, *_ = fen.split(' ')

    # Wipe the board - Shape: (rows, columns, layers)
    self._board_state = np.zeros((8, 8, 12), dtype=np.uint8)

    for i, row in enumerate(position.split('/')):
      column = 0
      for square_value in row:
        if square_value.isnumeric():
          column += int(square_value)
        else:
          if square_value not in self.inverse_layer_map.keys():
            raise ValueError(f'Invalid FEN: {square_value} not a valid piece.')
          layer = self.inverse_layer_map[square_value]
          self._board_state[i, column, layer] = 1
          column += 1

  def _set_flags(self, fen: str = None) -> None:
    self.is_game_over = False       # TODO
    self._is_in_check_white = False # TODO
    self._is_in_check_black = False # TODO

    self._is_in_double_check_white = False # TODO
    self._is_in_double_check_black = False # TODO

    fen = fen if fen is not None else self.default_fen
    _, to_move, castling, en_passant, halfmove, _ = fen.split(' ')

    self._white_to_move = (to_move == 'w')
    self._can_castle_short_white = ('K' in castling)
    self._can_castle_long_white = ('Q' in castling)
    self._can_castle_short_black = ('k' in castling)
    self._can_castle_long_black = ('q' in castling)
    self._en_passant_target = en_passant
    self._halfmove_clock = int(halfmove)

  def _get_layer_representation(self) -> np.ndarray:
    board = np.ones((8, 8))
    for layer in range(12):
      board += self._board_state[:, :, layer] * (layer + 1)
    return board - 2

  def __str__(self) -> str:
    def piece(row, column):
      index_array = np.where(self._board_state[row, column, :] == 1)[0]
      if len(index_array) > 0:
        return self.layer_map.get(index_array[0])
      return ' '

    return f'''
       ---------------
      |{'|'.join([piece(0, col) for col in range(8)])}|
      |{'|'.join([piece(1, col) for col in range(8)])}|
      |{'|'.join([piece(2, col) for col in range(8)])}|
      |{'|'.join([piece(3, col) for col in range(8)])}|
      |{'|'.join([piece(4, col) for col in range(8)])}|
      |{'|'.join([piece(5, col) for col in range(8)])}|
      |{'|'.join([piece(6, col) for col in range(8)])}|
      |{'|'.join([piece(7, col) for col in range(8)])}|
       ---------------
    '''

  def __hash__(self) -> int:
    return hash(self.fen)


if __name__ == '__main__':
  new_game = game('rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3')
  print(new_game)
  print(new_game.fen)
  print(new_game.get_legal_moves())
  print(new_game._is_in_check_white)
