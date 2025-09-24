"""
Student Agent Implementation for River and Stones Game

This file contains the essential utilities and template for implementing your AI agent.
Your task is to complete the StudentAgent class with intelligent move selection.

Game Rules:
- Goal: Get 4 of your stones into the opponent's scoring area
- Pieces can be stones or rivers (horizontal/vertical orientation)  
- Actions: move, push, flip (stone↔river), rotate (river orientation)
- Rivers enable flow-based movement across the board

Your Task:
Implement the choose() method in the StudentAgent class to select optimal moves.
You may add any helper methods and modify the evaluation function as needed.
"""

from collections import deque
import random
import copy
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import json


class Piece:
    def __init__(self, owner:str, side:str="stone", orientation:Optional[str]=None):
        self.owner = owner
        self.side = side
        self.orientation = orientation if orientation else "horizontal"
    def copy(self): return Piece(self.owner, self.side, self.orientation)
    def to_dict(self): return {"owner":self.owner,"side":self.side,"orientation":self.orientation}
    @staticmethod
    def from_dict(d:Optional[Dict[str,Any]]):
        if d is None: return None
        return Piece(d["owner"], d.get("side","stone"), d.get("orientation","horizontal"))


# ==================== GAME UTILITIES ====================
# Essential utility functions for game state analysis

def in_bounds(x: int, y: int, rows: int, cols: int) -> bool:
    """Check if coordinates are within board boundaries."""
    return 0 <= x < cols and 0 <= y < rows

def count_piece_in_goal(board,player,rows,cols,score_cols):
    if player == "circle":
        goal_row = top_score_row()
    else:  
        goal_row = bottom_score_row(rows)
        
    cnt = 0
    for x in score_cols:
        if board[goal_row][x] is not None:
            cnt+=1
    return cnt
      
def game_over(board, rows, score_cols):
    row1 = top_score_row()
    row2 = bottom_score_row(rows)
    cnt1 = 0
    cnt2 = 0
    for col in score_cols:
        piece1 = board[row1][col]
        if piece1 is not None and piece1.side == "stone":
            cnt1 += 1

        piece2 = board[row2][col]
        if piece2 is not None and piece2.side == "stone":
            cnt2 += 1

    if cnt1 == 4 or cnt2 == 4:
        return True

    return False

# ==================== GAME UTILITIES ====================
# Essential utility functions for game state analysis
def eval_hardcode(board : List[List[Optional["Piece"]]], player : str, rows: int, cols : int, score_cols: List[int]) -> int: 
    score = 0
    if player == "circle":
        target1 = [(rows - 3, 3), (rows - 3, 8)]
        target2 = [(rows - 2, 3), (rows - 2, 8)]
    else:
        target1 = [(1, 3), (1, 8)]
        target2 = [(2, 3), (2, 8)]

    for x, y in target1:
        piece = board[x][y]
        if piece is None:
            score += 0
        elif piece.owner != player:
            score -= 1
        elif piece.side == "river" and piece.orientation == "vertical":
            score += 2
        else:
            # for stone piece of current player
            score += 1

    for x, y in target2:
        piece = board[x][y]
        if piece is None:
            score += 0
        elif piece.owner != player:
            score -= 1
        elif piece.side == "stone":
            score += 2
        else:
            # for river piece of current player
            score += 1

    return score


#  some heuristic functions for move oedering/pruning
def distance_to_goal(x, y, player, rows, cols, score_cols):
    """
    Compute minimum distance from (x, y) to player's scoring area.
    score_cols[player] is assumed to give scoring columns for this player.
    """
    # For example: if player scores on the top row
    if player == "circle":
        goal_y = 2
    else:
        goal_y = rows - 3

    # Manhattan distance to nearest scoring cell
    return min(abs(y - goal_y) + abs(x - col) for col in score_cols)

def get_goal_side_rows(player,rows,cols,score_cols):
    # 5 rows
    if player== "circle":
        return range(0,4)
    else:
        return range(rows-5,rows-1)
    
    
def in_goal_side_rows(
    fx,
    fy,
    player,
    rows,
    cols,
    score_cols,
):
    """
    Check if exit (fx, fy) is in the 'goal side rows' rows/cols near the score area.
    """
    
    # goal side 5 rows
    if player == "circle":  # suppose Player 1 scores at right
        return fx <= 4
    else:  # Player 2 scores at left
        return fx >= rows - 5 


def has_adjacent_river(board, x, y, rows, cols)->bool:
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if in_bounds(nx, ny, rows, cols):
            neighbor = board[ny][nx]
            if neighbor and neighbor.side == "river":
                return True

    return False


def score_move(move, board, player, rows, cols, score_cols):
    fx, fy = move["to"]
    sx, sy = move["from"]
    score = 0

    if move["action"] == "move":
        
        if not in_goal_side_rows(sx,sy,player,rows,cols,score_cols) and  in_goal_side_rows(fx,fy,player,rows,cols,score_cols):
            score += 10 
            
        if sx not in score_cols and fx in score_cols:
            score += 1 
        sx,sy = move["from"]
        if not is_own_score_cell(sx,sy,player,rows,cols,score_cols) and is_own_score_cell(fx, fy, player, rows, cols, score_cols):
            score += 10
        
    # 4. For pushes: increase opponent distance from goal
    opponent = get_opponent(player)
    if move["action"] == "push":
        fx,fy = move["to"]
        pushed_piece_owner = board[fy][fx].owner
        
        dest_x, dest_y = move["pushed_to"]
        if pushed_piece_owner == player:
            score -= distance_to_goal(dest_x,dest_y,player,rows,cols,score_cols) * 0.1 
        else:
            score += distance_to_goal(dest_x, dest_y, opponent, rows, cols, score_cols) * 0.1
        # Prefer pushes to cells with no adjacent rivers
    return score


def score_flow_move(move, board, player, rows, cols, score_cols):
    x, y = move["from"]
    piece = board[y][x]
    score = 0
    
    # Save original state
    original_side = piece.side
    original_orientation = getattr(piece, "orientation", None)
    
    
    # Temporarily apply the move
    if move["action"] == "flip":
        if is_own_score_cell(x,y,player,rows,cols,score_cols) and piece.side =="river" :
            score +=100
        # elif is_own_score_cell(x,y,player,rows,cols,score_cols) and piece.side =="stone":
        #     score +=1
        piece.side = "river" if piece.side == "stone" else "stone"
        
    elif move["action"] == "rotate":
        piece.orientation = "vertical" if original_orientation == "horizontal" else "horizontal"

    # Determine if this move is a river push scenario
    river_push = move["action"] == "push" and getattr(piece, "side", "stone") == "river"

    # Compute flow destinations
    flow_cells = get_river_flow_destinations(
        board,
        x, y,               # starting river cell
        x, y,               # source position (self)
        player,
        rows,
        cols,
        score_cols,
        river_push=river_push
    )

    # Only consider safe cells (not in opponent score)
    safe_flow = [
        (fx, fy)
        for fx, fy in flow_cells
        if not is_opponent_score_cell(fx, fy, player, rows, cols, score_cols)
    ]
    
    goal_side_flow = [
        (fx,fy)
        for fx,fy in safe_flow
        if in_goal_side_rows(fx,fy,player,rows,cols,score_cols)
    ]
    
    in_goal_flow = [
        (fx,fy)
        for fx,fy in safe_flow
        if is_own_score_cell(fx,fy,player,rows,cols,score_cols)
    ]

    # Base score = number of safe flow cells
    # score += len(safe_flow)
    score += len(goal_side_flow) # bonus for goal side flow
    score += len(in_goal_flow) *5 # bonus for flow in goal

    # Bonus for connecting to existing rivers
    # if move["action"] in ("flip", "rotate") and piece.side == "river" and has_adjacent_river(board, x, y, rows, cols):
    #     score += 2

    # Revert piece to original state
    piece.side = original_side
    piece.orientation = original_orientation

    return score




def score_cols_for(cols: int) -> List[int]:
    """Get the column indices for scoring areas."""
    w = 4
    start = max(0, (cols - w) // 2)
    return list(range(start, start + w))


def top_score_row() -> int:
    """Get the row index for Circle's scoring area."""
    return 2


def bottom_score_row(rows: int) -> int:
    """Get the row index for Square's scoring area."""
    return rows - 3


def is_opponent_score_cell(
    x: int, y: int, player: str, rows: int, cols: int, score_cols: List[int]
) -> bool:
    """Check if a cell is in the opponent's scoring area."""
    if player == "circle":
        return (y == bottom_score_row(rows)) and (x in score_cols)
    else:
        return (y == top_score_row()) and (x in score_cols)


def is_own_score_cell(
    x: int, y: int, player: str, rows: int, cols: int, score_cols: List[int]
) -> bool:
    """Check if a cell is in the player's own scoring area."""
    if player == "circle":
        return (y == top_score_row()) and (x in score_cols)
    else:
        return (y == bottom_score_row(rows)) and (x in score_cols)


def get_opponent(player: str) -> str:
    """Get the opponent player identifier."""
    return "square" if player == "circle" else "circle"


# ==================== generate moves ======================



def get_river_flow_destinations(
    board: List[List[Optional["Piece"]]],
    rx: int,
    ry: int,
    sx: int,
    sy: int,
    player: str,
    rows: int,
    cols: int,
    score_cols: List[int],
    river_push: bool = False
) -> List[Tuple[int, int]]:
    """
    Compute all possible destinations for a piece entering a river.
    Correctly handles normal river flow and river pushes.

    Args:
        board: Current board state
        rx, ry: River entry coordinates
        sx, sy: Source piece coordinates (ignored as obstacle)
        player: Current player
        rows, cols: Board dimensions
        score_cols: List of scoring columns
        river_push: True if a river piece is pushing a stone

    Returns:
        List of unique (x, y) coordinates where the piece can end up
    """

    destinations = set()
    visited = set()
    queue = deque([(rx, ry)])

    while queue:
        x, y = queue.popleft()
        if (x, y) in visited or not in_bounds(x, y, rows, cols):
            continue
        visited.add((x, y))

        cell = board[y][x]

        # In river-push, allow BFS even if starting cell is a stone
        if river_push and (x, y) == (rx, ry):
            is_river = True
        else:
            is_river = cell is not None and getattr(cell, "side", None) == "river"

        if not is_river:
            # Empty cell: valid destination unless it's opponent's scoring area
            if cell is None and not is_opponent_score_cell(x, y, player, rows, cols, score_cols):
                destinations.add((x, y))
            continue

        # Determine flow directions based on orientation
        orientation = getattr(cell, "orientation", "horizontal")
        dirs = [(1, 0), (-1, 0)] if orientation == "horizontal" else [(0, 1), (0, -1)]

        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            while in_bounds(nx, ny, rows, cols):
                # Block opponent scoring area
                if is_opponent_score_cell(nx, ny, player, rows, cols, score_cols):
                    break

                # Skip source cell (for push) so stone can flow
                if (nx, ny) == (sx, sy):
                    nx += dx
                    ny += dy
                    continue

                next_cell = board[ny][nx]

                if next_cell is None:
                    destinations.add((nx, ny))
                    nx += dx
                    ny += dy
                    continue

                if getattr(next_cell, "side", None) == "river":
                    if (nx, ny) not in visited:
                        queue.append((nx, ny))
                    break  # stop current direction; BFS will continue from river
                break  # blocked by stone

    return list(destinations)

def generate_all_moves(
    board: List[List[Optional["Piece"]]],
    player: str,
    rows: int,
    cols: int,
    score_cols: List[int]
) -> List[Dict[str, Any]]:
    """
    Generate all valid moves for the current player on the board.
    Uses compute_valid_targets to ensure only legal moves/pushes.
    Supports move, push, flip, and rotate actions.
    """
    moves = []

    for y in range(rows):
        for x in range(cols):
            piece = board[y][x]
            if not piece or piece.owner != player:
                continue

            # ------------------- Compute valid moves & pushes -------------------
            valid_info = compute_valid_targets(board, x, y, player, rows, cols, score_cols)

            # Add normal moves
            for to in valid_info['moves']:
                moves.append({
                    "action": "move",
                    "from": [x, y],
                    "to": list(to)
                })

            # Add valid pushes (with opponent-score restriction)
            for from_cell, pushed_cell in valid_info['pushes']:
                px, py = pushed_cell
                target_piece = board[from_cell[1]][from_cell[0]] if in_bounds(from_cell[0], from_cell[1], rows, cols) else None

                # Only skip if the pushed piece would land in *opponent's* scoring area
                if target_piece and is_opponent_score_cell(px, py, target_piece.owner, rows, cols, score_cols):
                    continue

                moves.append({
                    "action": "push",
                    "from": [x, y],
                    "to": list(from_cell),
                    "pushed_to": list(pushed_cell)
                })

            # ------------------- Flips -------------------
            if piece.side == "stone":
                # Flip stone -> river
                for ori in ("horizontal", "vertical"):
                    moves.append({
                        "action": "flip",
                        "from": [x, y],
                        "orientation": ori
                    })
            else:
                # Flip river -> stone
                moves.append({
                    "action": "flip",
                    "from": [x, y]
                })

                # Rotate river piece
                new_ori = "vertical" if piece.orientation == "horizontal" else "horizontal"
                moves.append({
                    "action": "rotate",
                    "from": [x, y],
                    "orientation": new_ori
                })

    # ------------------- Deduplicate moves -------------------
    seen = set()
    unique_moves = []

    for m in moves:
        key = (
            m["from"][0],
            m["from"][1],
            tuple(m["to"]) if m.get("to") else None,
            tuple(m["pushed_to"]) if m.get("pushed_to") else None,
            m.get("orientation", None),
            m["action"]
        )
        if key not in seen:
            seen.add(key)
            unique_moves.append(m)

    return unique_moves


def generate_heuristic_moves(board, player, rows, cols, score_cols):
    no_of_piece_in_goal = count_piece_in_goal(board,player,rows,cols,score_cols)
    moves = generate_all_moves(board, player, rows, cols, score_cols)
    
    filtered_moves = []
    if no_of_piece_in_goal == 4:
        for move in moves:
            if move['action'] == "flip" :  
                sx,sy = move["from"]
                piece = board[sy][sx]
                if piece.side == "river" and is_own_score_cell(sx,sy,player,rows,cols,score_cols):
                    filtered_moves.append(move)             
        
    else:
        river_moves = [m for m in moves if m["action"] in ("flip", "rotate")]
        river_moves = sorted(river_moves, key=lambda m: score_flow_move(m, board, player, rows, cols, score_cols), reverse=True)
        river_moves = river_moves[:10]
        weights1 = [score_flow_move(m, board, player, rows, cols, score_cols) for m in river_moves]  

        moves_with_to = [m for m in moves if m["action"] in ("push","move")]
        moves_with_to = sorted(moves_with_to,key=lambda m: score_move(m, board, player, rows, cols, score_cols),reverse=True)
        moves_with_to = moves_with_to[:20]
        weights2 = [score_move(m, board, player, rows, cols, score_cols) for m in moves_with_to] 

        filtered_moves = river_moves + moves_with_to
        weights1.extend(weights2)
        
        if random.random() < 0.1 and  filtered_moves:
           idx = random.randrange(len(filtered_moves))
        #    random_move = random.choices(filtered_moves,weights=weights1,k=1)
           filtered_moves.pop(idx)
        else:
            random.shuffle(filtered_moves)
    return filtered_moves
    
def count_all_moves(board, player, rows, cols, score_cols):
    moves = generate_all_moves(board, player, rows, cols, score_cols)
    return len(moves)



# ==================== BOARD EVALUATION ====================


def eval_stone_in_goal(board: List[List[Any]], player, row, col, score_cols: List[int]):
    # function to check if we are at our goal
    if player == "circle":
        scoreRow = top_score_row()
    else:
        scoreRow = bottom_score_row(row)
    score = 0

    for y in score_cols:
        piece = board[scoreRow][y]
        if piece and piece.owner == player and piece.side == "stone":
            score += 1  # fixed bonus
    return score


def eval_river_in_goal(board: List[List[Any]], player, row, col, score_cols: List[int]):
    # function to check if we are at our goal
    if player == "circle":
        scoreRow = top_score_row()
    else:
        scoreRow = bottom_score_row(row)
    score = 0

    for y in score_cols:
        piece = board[scoreRow][y]
        if piece and piece.owner == player and piece.side == "river":
            score += 1  # fixed bonus
    return score

def eval_pieces_in_goal(board,player,rows,cols,score_cols):
    cnt = count_piece_in_goal(board,player,rows,cols,score_cols)
    score = 0
    if player == "circle":
        goal_row = top_score_row()
    else:
        goal_row = bottom_score_row(rows)
        
    if cnt == 4:
        score +=1
    return score
    
    

def eval_manhattan_dist(
    board: List[List[Any]], player, rows, cols, score_cols: List[Any]
):
    # Manhattan distance between my pieces and the nearest EMPTY goal cell
    score = 0
    if player == "circle":
        scoreRow = top_score_row()
    else:
        scoreRow = bottom_score_row(rows)

    emptyGoals = [(scoreRow, gx) for gx in score_cols if board[scoreRow][gx] is None]

    for i in range(rows):
        for j in range(cols):
            piece = board[i][j]
            if piece and piece.owner == player and emptyGoals:
                # Manhattan distance to the closest EMPTY goal
                dists = [abs(i - gr) + abs(j - gc) for gr, gc in emptyGoals]
                score += min(dists)
    # print("manhatt:", score)
    return score


def eval_dist_to_goal_row(board,player,rows,cols,score_cols):
    if player == "circle":
        goal_row = top_score_row()
    else:
        goal_row = bottom_score_row(rows)    
    score = 0
    for y in range(rows):
        for x in range(cols):
            piece = board[y][x]
            if piece is not None:
                score += abs(y-goal_row)
    return score

def eval_dist_to_goal_cols(board,player,rows,cols,score_cols):
    goal_side_rows = get_goal_side_rows(player,rows,cols,score_cols)
    if player=="circle" :
        goal_row = top_score_row() 
    else :
        goal_row = bottom_score_row(rows)
    score = 0
    for y in range(rows):
        for x in range(cols):
            piece = board[y][x]
            if piece is not None and y in goal_side_rows:
                minS = 1000
                for col in score_cols:
                    if board[goal_row][col] is not None:
                        minS = min(minS,abs(x-col))
                if minS != 1000:
                    score+=minS  
    return score

def eval_piece_near_river(board: List[List[Any]], player, row, col):
    # Number of my pieces adjacent to a river
    score = 0
    for i in range(row):
        for j in range(col):
            piece = board[i][j]
            if piece and piece.owner == player:
                neighbour = False
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nj, ni = j + dx, i + dy
                    if in_bounds(nj, ni, row, col):
                    # if 0<=ni < row and 0<= nj < col:
                        neighbor = board[ni][nj]
                        if neighbor and neighbor.side == "river":
                            neighbour = True
                            break
                if neighbour:
                    score += 1
    return score


def eval_no_of_moves(board, player, rows, cols, score_cols):
    moves = count_all_moves(board, player, rows, cols, score_cols)
    # print("move_cnt:", len(moves))
    return moves


def eval_river_cnt(board, player, rows, cols, score_cols):
    score = 0
    for i in range(rows):
        for j in range(cols):
            piece = board[i][j]
            if piece and piece.owner == player and piece.side == "river":
                score += 1
    return score

def eval_moves_dist_to_goal_side_row(board,player,rows,cols,score_cols,moves):
    goal_side_rows = get_goal_side_rows(player,rows,cols,score_cols)
    score = 0
    for move in moves:
        if move["action"] == "move":
            dx,dy = move["to"]
            score+= min([abs(dy-row) for row in goal_side_rows])
    return score

def eval_moves_dist_to_goal_side_col(board,player,rows,cols,score_cols,moves):
    goal_side_rows = get_goal_side_rows(player,rows,cols,score_cols)
    score = 0
    for move in moves:
        if move["action"] == "move":
            dx,dy = move["to"]
            if dy in goal_side_rows:
                score+=min([abs(dx-col) for col in score_cols])
    return score



def evaluate_simple(board, player, rows, cols, score_cols):
    """
    Combine all simple evaluation functions with given weights.
    """
    weights = load_weights()
    opponent = get_opponent(player)
    
    
    all_piece_in_goal=eval_pieces_in_goal(board,player,rows,cols,score_cols)
    
    stone_goal = eval_stone_in_goal(board, player, rows, cols, score_cols)
    opp_stone_goal = eval_stone_in_goal(board, opponent, rows, cols, score_cols)

    river_goal = eval_river_in_goal(board, player, rows, cols, score_cols)
    opp_river_goal = eval_river_in_goal(board, opponent, rows, cols, score_cols)

    # manh_dist = eval_manhattan_dist(board, player, rows, cols, score_cols)
    # opp_manh_dist = eval_manhattan_dist(board, opponent, rows, cols, score_cols)
    
    row_dist = eval_dist_to_goal_row(board, player, rows, cols, score_cols)
    opp_row_dist = eval_dist_to_goal_row(board, opponent, rows, cols, score_cols)
    
    col_dist = eval_dist_to_goal_cols(board, player, rows, cols, score_cols)
    opp_col_dist = eval_dist_to_goal_cols(board, opponent, rows, cols, score_cols)
    
    near_river = eval_piece_near_river(board, player, rows, cols)
    opp_near_river = eval_piece_near_river(board, opponent, rows, cols)

    moves = generate_all_moves(board, player, rows, cols, score_cols)
    opp_moves = generate_all_moves(board, opponent, rows, cols, score_cols)
    
    # move_count = len(moves)
    # opp_move_count = len(opp_moves)
    
    goal_side_row_dist_moves = eval_moves_dist_to_goal_side_row(board,player,rows,cols,score_cols,moves) 
    opp_goal_side_row_dist_moves = eval_moves_dist_to_goal_side_row(board,opponent,rows,cols,score_cols,opp_moves) 

    goal_side_col_dist_moves = eval_moves_dist_to_goal_side_col(board,player,rows,cols,score_cols,moves) 
    opp_goal_side_col_dist_moves = eval_moves_dist_to_goal_side_col(board,opponent,rows,cols,score_cols,opp_moves) 



    features ={
        "all_piece_in_goal":all_piece_in_goal,
        "stone_goal": stone_goal,
        "opp_stone_goal": opp_stone_goal,
        "river_goal": river_goal,
        "opp_river_goal": opp_river_goal,
        # "manh_dist": manh_dist,
        # "opp_manh_dist": opp_manh_dist,
        
        "row_dist":row_dist,
        "opp_row_dist":opp_row_dist,
        "col_dist":col_dist,
        "opp_col_dist":opp_col_dist,
        
        
        
        "near_river": near_river,
        "opp_near_river": opp_near_river,
        # "move_count": move_count,
        # "opp_move_count": opp_move_count,
        
        "goal_side_dist_row_moves":goal_side_row_dist_moves,
        "opp_goal_side_dist_row_moves":opp_goal_side_row_dist_moves,
        "goal_side_dist_col_moves":goal_side_col_dist_moves,
        "opp_goal_side_dist_col_moves":opp_goal_side_col_dist_moves
    }

    # player_river_cnt = eval_river_cnt(board,player,rows,cols,score_cols)
    # player_stone_cnt = 12-player_river_cnt

    # opp_river_cnt = eval_river_cnt(board,opponent,rows,cols,score_cols)
    # opp_stone_cnt = 12- opp_river_cnt

   
    total = sum(weights[k] * features[k] for k in features)
    return total


# ===================== simmulate for minimax


def compute_valid_targets(
    board: List[List[Optional[Piece]]],
    sx: int,
    sy: int,
    player: str,
    rows: int,
    cols: int,
    score_cols: List[int],
) -> Dict[str, Any]:
    if not in_bounds(sx, sy, rows, cols):
        return {"moves": set(), "pushes": []}

    p = board[sy][sx]
    if p is None or p.owner != player:
        return {"moves": set(), "pushes": []}

    moves = set()
    pushes = []
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for dx, dy in dirs:
        tx, ty = sx + dx, sy + dy
        if not in_bounds(tx, ty, rows, cols):
            continue

        # block entering opponent score cell
        if is_opponent_score_cell(tx, ty, player, rows, cols, score_cols):
            continue

        target_piece = board[ty][tx]

        if target_piece is None:
            # Normal move
            moves.add((tx, ty))

        elif target_piece.side == "river":
            # Cannot push rivers; compute river flow for moves
            flow = get_river_flow_destinations(
                board, tx, ty, sx, sy, player, rows, cols, score_cols
            )
            for fx, fy in flow:
                moves.add((fx, fy))

        elif target_piece.side == "stone":
            # Handle push logic
            if p.side == "stone":
                # Stone pushes stone: next cell in same direction must be empty
                px, py = tx + dx, ty + dy
                if (
                    in_bounds(px, py, rows, cols)
                    and board[py][px] is None
                    and not is_opponent_score_cell(px, py, target_piece.owner, rows, cols, score_cols)
                ):
                    pushes.append(((tx, ty), (px, py)))

            elif p.side == "river":
                # River pushes stone: simulate river replacing stone
                original_piece = board[ty][tx]
                board[ty][tx] = p  # temporarily place river
                flow = get_river_flow_destinations(
                    board,
                    tx, ty,
                    sx, sy,  # original river position
                    original_piece.owner,
                    rows, cols,
                    score_cols,
                    river_push=True
                )
                board[ty][tx] = original_piece  # restore stone

                for fx, fy in flow:
                    # Cannot move back to original river position
                    if (fx, fy) == (sx, sy):
                        continue
                    if not is_opponent_score_cell(fx, fy, original_piece.owner, rows, cols, score_cols):
                        pushes.append(((tx, ty), (fx, fy)))

    return {"moves": moves, "pushes": pushes}


def validate_and_apply_move(
    board: List[List[Optional["Piece"]]],
    move: Dict[str, Any],
    player: str,
    rows: int,
    cols: int,
    score_cols: List[int]
) -> Tuple[bool, str]:
    """
    Validate and apply a move for the stones & river game.
    Supports actions: move, push, flip, rotate.

    Returns:
        Tuple[bool, str]: (success flag, message)
    """
    # Validate input type
    if not isinstance(move, dict):
        return False, "move must be a dict"

    action = move.get("action")

    # -------------------
    # MOVE ACTION
    # -------------------
    if action == "move":
        fr = move.get("from")
        to = move.get("to")
        if not fr or not to:
            return False, "move needs 'from' and 'to'"
        fx, fy = int(fr[0]), int(fr[1])
        tx, ty = int(to[0]), int(to[1])

        if not (in_bounds(fx, fy, rows, cols) and in_bounds(tx, ty, rows, cols)):
            return False, "out of bounds"
        if is_opponent_score_cell(tx, ty, player, rows, cols, score_cols):
            return False, "cannot move into opponent score cell"

        piece = board[fy][fx]
        if piece is None or piece.owner != player:
            return False, "invalid piece"

        # Normal move into empty cell
        if board[ty][tx] is None:
            board[ty][tx] = piece
            board[fy][fx] = None
            return True, "moved"

        # Move with push
        pushed = move.get("pushed_to")
        if not pushed:
            return False, "destination occupied; 'pushed_to' required"
        ptx, pty = int(pushed[0]), int(pushed[1])
        dx, dy = tx - fx, ty - fy

        if (ptx, pty) != (tx + dx, ty + dy):
            return False, "'pushed_to' must be in line with move direction"
        if not in_bounds(ptx, pty, rows, cols):
            return False, "pushed_to out of bounds"
        if is_opponent_score_cell(ptx, pty, player, rows, cols, score_cols):
            return False, "cannot push into opponent score cell"
        if board[pty][ptx] is not None:
            return False, "'pushed_to' not empty"

        # Apply move + push
        board[pty][ptx] = board[ty][tx]  # pushed piece
        board[ty][tx] = piece            # mover occupies destination
        board[fy][fx] = None             # origin cleared
        return True, "move + push applied"

    # -------------------
    # PUSH ACTION
    # -------------------
    elif action == "push":
        fr = move.get("from")
        to = move.get("to")
        pushed = move.get("pushed_to")
        if not fr or not to or not pushed:
            return False, "push needs 'from', 'to', and 'pushed_to'"

        fx, fy = int(fr[0]), int(fr[1])
        tx, ty = int(to[0]), int(to[1])
        px, py = int(pushed[0]), int(pushed[1])

        if not all(in_bounds(x, y, rows, cols) for x, y in [(fx, fy), (tx, ty), (px, py)]):
            return False, "out of bounds"

        piece = board[fy][fx]
        target = board[ty][tx]
        if piece is None or piece.owner != player:
            return False, "invalid piece"
        if target is None:
            return False, "'to' must be occupied"
        if board[py][px] is not None:
            return False, "'pushed_to' not empty"
        if piece.side == "river" and target.side == "river":
            return False, "rivers cannot push rivers"

        pushed_player = target.owner
        if is_opponent_score_cell(tx, ty, player, rows, cols, score_cols) or \
           is_opponent_score_cell(px, py, pushed_player, rows, cols, score_cols):
            return False, "push would enter opponent score cell"

        # Validate against computed valid pushes
        valid_info = compute_valid_targets(board, fx, fy, player, rows, cols, score_cols)
        if ((tx, ty), (px, py)) not in valid_info['pushes']:
            return False, "push pair invalid"

        # Apply push
        board[py][px] = board[ty][tx]  # pushed piece
        board[ty][tx] = piece          # mover occupies target
        board[fy][fx] = None           # origin cleared

        # If mover was river, convert back to stone
        if board[ty][tx].side == "river":
            board[ty][tx].side = "stone"
            board[ty][tx].orientation = None

        return True, "push applied"

    # -------------------
    # FLIP ACTION
    # -------------------
    elif action == "flip":
        fr = move.get("from")
        if not fr:
            return False, "flip needs 'from'"
        fx, fy = int(fr[0]), int(fr[1])
        if not in_bounds(fx, fy, rows, cols):
            return False, "out of bounds"
        piece = board[fy][fx]
        if piece is None or piece.owner != player:
            return False, "invalid piece"

        # Stone -> River
        if piece.side == "stone":
            ori = move.get("orientation")
            if ori not in ("horizontal", "vertical"):
                return False, "stone -> river needs valid orientation"

            # Check that resulting flow won't reach opponent score
            piece.side = "river"
            piece.orientation = ori
            flow = get_river_flow_destinations(board, fx, fy, fx, fy, player, rows, cols, score_cols)
            piece.side = "stone"  # revert temporarily
            piece.orientation = None
            for x, y in flow:
                if is_opponent_score_cell(x, y, player, rows, cols, score_cols):
                    return False, "flip would allow flow into opponent score"

            # Commit flip
            piece.side = "river"
            piece.orientation = ori
            return True, "flipped to river"

        # River -> Stone
        piece.side = "stone"
        piece.orientation = None
        return True, "flipped to stone"

    # -------------------
    # ROTATE ACTION
    # -------------------
    elif action == "rotate":
        fr = move.get("from")
        if not fr:
            return False, "rotate needs 'from'"
        fx, fy = int(fr[0]), int(fr[1])
        if not in_bounds(fx, fy, rows, cols):
            return False, "out of bounds"
        piece = board[fy][fx]
        if piece is None or piece.owner != player:
            return False, "invalid piece"
        if piece.side != "river":
            return False, "rotate only on river"

        # Toggle orientation
        piece.orientation = "horizontal" if piece.orientation == "vertical" else "vertical"

        # Check resulting river flow
        flow = get_river_flow_destinations(board, fx, fy, fx, fy, player, rows, cols, score_cols)
        for x, y in flow:
            if is_opponent_score_cell(x, y, player, rows, cols, score_cols):
                # Revert orientation
                piece.orientation = "horizontal" if piece.orientation == "vertical" else "vertical"
                return False, "rotation allows flow into opponent score"

        return True, "rotated"

    # -------------------
    # UNKNOWN ACTION
    # -------------------
    return False, "unknown action"

def simulate_move(
    board: List[List[Any]],
    move: Dict[str, Any],
    player: str,
    rows: int,
    cols: int,
    score_cols: List[int],
) -> Tuple[bool, Any]:
    """
    Simulate a move on a copy of the board.

    Returns:
        (success: bool, new_board or error_message)
    """

    board_copy = copy.deepcopy(board)
    success, message = validate_and_apply_move(
        board_copy, move, player, rows, cols, score_cols
    )

    if success:
        return True, board_copy
    else:
        return False, message

def load_weights(file_path="weights.json"):
    with open(file_path, "r") as f:
        weights = json.load(f)
    return weights


def minimax_move(
    board: List[List[Any]],
    player: str,
    rows: int,
    cols: int,
    score_cols: List[int],
) -> Dict[str, Any]:
    maxVal = -float("inf")
    maxMove = None
    opponent = get_opponent(player)

    moves = generate_heuristic_moves(board, player, rows, cols, score_cols)
    print(f"Minimax for {player}, evaluating {len(moves)} moves")
    for move in moves:
        success, result = simulate_move(board, move, player, rows, cols, score_cols)
        if success:
            child_board = result
            childVal = maxValue(
                child_board,
                player,  # Next player (minimizing)
                player,  # Original player (for evaluation)
                rows,
                cols,
                score_cols,
                -float("inf"),
                float("inf"),
                1,
            )
            # print(f"Move {move} -> Score: {childVal}")

            if childVal > maxVal:
                maxVal = childVal
                maxMove = move
                # print(f"New best: {move} with score {maxVal}")
        else:
            print(f"Move failed: {move}, Error: {result}")

    # print(f"Selected: {maxMove} with score {maxVal}")
    return maxMove


def minValue(
    board: List[List[Any]],
    current_player: str,  # Player whose turn it is (minimizing)
    original_player: str,  # Player we're optimizing for
    rows: int,
    cols: int,
    score_cols: List[int],
    alpha: float,
    beta: float,
    ply: int,
) -> float:
    if ply == 0 or game_over(board,rows,score_cols):
        return evaluate_simple(board, original_player, rows, cols, score_cols)

    # GENERATE moves for the CURRENT player (minimizing)
    moves = generate_heuristic_moves(board, current_player, rows, cols, score_cols)

    if not moves:  # No moves available
        return evaluate_simple(board, original_player, rows, cols, score_cols)

    minVal = float("inf")
    opponent = get_opponent(current_player)

    for move in moves:
        success, child_board = simulate_move(
            board, move, current_player, rows, cols, score_cols
        )
        if success:
            childVal = maxValue(
                child_board,
                opponent,  # Next player (maximizing)
                original_player,  # Keep original player
                rows,
                cols,
                score_cols,
                alpha,
                beta,
                ply,
            )
            minVal = min(minVal, childVal)
            beta = min(beta, childVal)
            if beta <= alpha:
                break
        else:
            print(f"Move failed in minValue: {move}")

    return minVal


def maxValue(
    board: List[List[Any]],
    current_player: str,  # Player whose turn it is (maximizing)
    original_player: str,  # Player we're optimizing for
    rows: int,
    cols: int,
    score_cols: List[int],
    alpha: float,
    beta: float,
    ply: int,
) -> float:
    if ply == 0 or game_over(board,rows,score_cols):
        return evaluate_simple(board, original_player, rows, cols, score_cols)

    # GENERATE moves for the CURRENT player (maximizing)
    moves = generate_heuristic_moves(board, current_player, rows, cols, score_cols)

    if not moves:  # No moves available
        return evaluate_simple(board, original_player, rows, cols, score_cols)

    maxVal = -float("inf")
    opponent = get_opponent(current_player)

    for move in moves:
        success, child_board = simulate_move(
            board, move, current_player, rows, cols, score_cols
        )
        if success:
            childVal = maxValue(
                child_board,
                current_player,  # Next player (minimizing)
                original_player,  # Keep original player
                rows,
                cols,
                score_cols,
                alpha,
                beta,
                ply-1,
            )
            maxVal = max(maxVal, childVal)
            alpha = max(alpha, childVal)
            if alpha >= beta:
                break
        else:
            print(f"Move failed in maxValue: {move}")

    return maxVal

# ==================== BASE AGENT CLASS ====================


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    """

    def __init__(self, player: str):
        """Initialize agent with player identifier."""
        self.player = player
        self.opponent = get_opponent(player)

    @abstractmethod
    def choose(
        self, board: List[List[Any]], rows: int, cols: int, score_cols: List[int]
    ) -> Optional[Dict[str, Any]]:
        """
        Choose the best move for the current board state.

        Args:
            board: 2D list representing the game board
            rows, cols: Board dimensions
            score_cols: List of column indices for scoring areas

        Returns:
            Dictionary representing the chosen move, or None if no moves available
        """
        pass

# ==================== STUDENT AGENT IMPLEMENTATION ====================

class StudentAgent(BaseAgent):
    """
    Student Agent Implementation
    
    TODO: Implement your AI agent for the River and Stones game.
    The goal is to get 4 of your stones into the opponent's scoring area.
    
    You have access to these utility functions:
    - generate_all_moves(): Get all legal moves for current player
    - basic_evaluate_board(): Basic position evaluation 
    - simulate_move(): Test moves on board copy
    - count_stones_in_scoring_area(): Count stones in scoring positions
    """
    
    def __init__(self, player: str):
        super().__init__(player)
        # TODO: Add any initialization you need
    
    def choose(self, board: List[List[Any]], rows: int, cols: int, score_cols: List[int], current_player_time: float, opponent_time: float) -> Optional[Dict[str, Any]]:
        """
        Choose the best move for the current board state.
        
        Args:
            board: 2D list representing the game board
            rows, cols: Board dimensions  
            score_cols: Column indices for scoring areas
            
        Returns:
            Dictionary representing your chosen move
        """
        my_move = minimax_move(board, self.player, rows, cols, score_cols)
        print("my_move",my_move)

        if not my_move:
            return None
        # moves = generate_all_moves(board,self.player,rows,cols,score_cols)
        # print("move_cnt", len(moves))
        # for move in moves:
        #     print(move)

        # TODO: Replace random selection with your AI algorithm
        return my_move

# ==================== TESTING HELPERS ====================

def test_student_agent():
    """
    Basic test to verify the student agent can be created and make moves.
    """
    print("Testing StudentAgent...")
    
    try:
        from gameEngine import default_start_board, DEFAULT_ROWS, DEFAULT_COLS
        
        rows, cols = DEFAULT_ROWS, DEFAULT_COLS
        score_cols = score_cols_for(cols)
        board = default_start_board(rows, cols)
        
        agent = StudentAgent("circle")
        move = agent.choose(board, rows, cols, score_cols,1.0,1.0)
        
        if move:
            print("✓ Agent successfully generated a move")
        else:
            print("✗ Agent returned no move")
    
    except ImportError:
        agent = StudentAgent("circle")
        print("✓ StudentAgent created successfully")

if __name__ == "__main__":
    # Run basic test when file is executed directly
    test_student_agent()

