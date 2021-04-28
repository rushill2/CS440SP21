import math
import chess.lib
from chess.lib.utils import encode, decode
from chess.lib.heuristics import evaluate
from chess.lib.core import makeMove

###########################################################################################
# Utility function: Determine all the legal moves available for the side.
# This is modified from chess.lib.core.legalMoves:
#  each move has a third element specifying whether the move ends in pawn promotion
def generateMoves(side, board, flags):
    for piece in board[side]:
        fro = piece[:2]
        for to in chess.lib.availableMoves(side, board, piece, flags):
            promote = chess.lib.getPromote(None, side, board, fro, to, single=True)
            yield [fro, to, promote]
            
###########################################################################################
# Example of a move-generating function:
# Randomly choose a move.
def random(side, board, flags, chooser):
    '''
    Return a random move, resulting board, and value of the resulting board.
    Return: (value, moveList, boardList)
      value (int or float): value of the board after making the chosen move
      moveList (list): list with one element, the chosen move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    moves = [ move for move in generateMoves(side, board, flags) ]
    if len(moves) > 0:
        move = chooser(moves)
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
        value = evaluate(newboard)
        return (value, [ move ], { encode(*move): {} })
    else:
        return (evaluate(board), [], {})

###########################################################################################
# Stuff you need to write:
# Move-generating functions using minimax, alphabeta, and stochastic search.
def minimax(side, board, flags, depth):
    '''
    Return a minimax-optimal move sequence, tree of all boards evaluated, and value of best path.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''
    moves = [move for move in generateMoves(side, board, flags)]
    moveTree = {}
    moveList = []
    # Base Case
    if depth==0 or len(moves)==0:
      return evaluate(board), [], {}

    # For min
    if side is True:
      value = float('inf')
      moveListT = []
      for x in moves:
        move = encode(*x)
        newside, newboard, newflags = makeMove(True, board, x[0], x[1], flags, x[2])
        minimaxval, moveListT, moveTree[move] = minimax(newside, newboard, newflags, depth-1)
        if minimaxval < value:
          moveListT.append(x)
          value = minimaxval
          moveList = moveListT
      moveList.reverse()
      return value, moveList, moveTree

    # For max
    elif side is False:
      moveListF = []
      value = float('-inf')
      for x in moves:
        move = encode(*x)
        newside, newboard, newflags = makeMove(False, board, x[0], x[1], flags, x[2])
        minimaxval, moveListF, moveTree[move] = minimax(newside, newboard, newflags, depth-1)
        if minimaxval > value:
          moveListF.append(x)
          value = minimaxval
          moveList = moveListF
      moveList.reverse()
      return value, moveList, moveTree
    raise NotImplementedError("you need to write this!")



def alphabeta(side, board, flags, depth, alpha=-math.inf, beta=math.inf):
    '''
    Return minimax-optimal move sequence, and a tree that exhibits alphabeta pruning.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''
    moves = [move for move in generateMoves(side, board, flags)]
    moveTree = {}
    moveList = []
    # Base Case
    if depth==0 or len(moves)==0:
      return evaluate(board), [], {}

    # For min
    if side is True:
      value = float('inf')
      moveListT = []
      for x in moves:
        move = encode(*x)
        newside, newboard, newflags = makeMove(True, board, x[0], x[1], flags, x[2])
        minimaxval, moveListT, moveTree[move] = alphabeta(newside, newboard, newflags, depth-1, alpha, beta)
        if minimaxval < value:
          moveListT.insert(0,x)
          value = minimaxval
          moveList = moveListT
        beta = min(value, beta)
        if alpha >= beta:
          break
      return value, moveList, moveTree

    # For max
    elif side is False:
      moveListF = []
      value = float('-inf')
      for x in moves:
        move = encode(*x)
        newside, newboard, newflags = makeMove(False, board, x[0], x[1], flags, x[2])
        minimaxval, moveListF, moveTree[move] = alphabeta(newside, newboard, newflags, depth-1, alpha, beta)
        if minimaxval > value:
          moveListF.insert(0,x)
          value = minimaxval
          moveList = moveListF
        alpha = max(value, alpha)
        if beta <= alpha:
          break
      return value, moveList, moveTree
    print(moveTree)

    raise NotImplementedError("you need to write this!")

# depthflag = 0
valset = []

def stochastichelp(side, board, flags, depth, breadth, chooser, moves, mark, value):
  # value = 0
  avg = 0
  moveList = []
  moveTree = {}
  avgval=0
  #for depth=depth-1
  for i in range(breadth):
    if depth==0:
      return evaluate(board), [], {}
    else:
      pick = chooser(moves)
        # print(pick)
      newside, newboard, newflags = makeMove(side, board, pick[0], pick[1], flags, pick[2])
      value, moveList, moveTree = stochastichelp(newside, newboard, newflags, depth-1, breadth, chooser, moves, 0, value)
      avg += value/breadth
  return avg, moveList, moveTree


def stochastic(side, board, flags, depth, breadth, chooser):
    '''
    Choose the best move based on breadth randomly chosen paths per move, of length depth-1.
    Return: (value, moveList, moveTree)
      value (float): average board value of the paths for the best-scoring move
      moveLists (list): any sequence of moves, of length depth, starting with the best move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
      breadth: number of different paths 
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''

    moves = [move for move in generateMoves(side, board, flags)]

    moveTree = {}
    moveList= []
    # valset = []
    value = 0
    for move in moves:
      newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
    # Edge case
    if depth==0 or len(moves)==0:
      return evaluate(board), [], {}
    
    # depthflag = depth
    for move in generateMoves(side, board, flags):
      newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
      value, moveList, moveTree[encode(*move)] = stochastichelp(newside, newboard, newflags, depth-1, breadth, chooser, moves, 0, value)
      valset.append(value)
      
    # taking min/max avg values for all moves.
    if side:
      value = min(valset)
    else:
      value = max(valset)

    return value, moveList, moveTree


   
