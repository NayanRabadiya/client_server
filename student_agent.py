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

import random
import copy
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod


# ==================== GAME UTILITIES ====================
# Essential utility functions for game state analysis
def eval_hardcode(board, player, rows, cols, score_cols):
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
    band = how wide the band is (2 means last 2 columns).
    """
    offset = 2
    if player == "circle":  # suppose Player 1 scores at right
        return fx <= offset + 2
    else:  # Player 2 scores at left
        return fx >= rows - 3 - offset


def has_adjacent_river(board, x, y, rows, cols):
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
    score = 0

    if move['action']=="move":
        
        score -= distance_to_goal(fx, fy, player, rows, cols, score_cols)
        if is_own_score_cell(fx, fy, player, rows, cols, score_cols):
            score += 50


    # 4. For pushes: increase opponent distance from goal
    opponent = get_opponent(player)
    if move['action']=="push":
        dest_x, dest_y = move['pushed_to']
        score += distance_to_goal(dest_x, dest_y, opponent, rows, cols, score_cols)
        # Prefer pushes to cells with no adjacent rivers
        if not has_adjacent_river(board, dest_x, dest_y, rows, cols):
            score += 10

    return score

def score_flow_move(move, board, player, rows, cols, score_cols):
    x, y = move['from']
    piece = board[y][x]

    # Save original state
    original_side = piece.side
    original_orientation = getattr(piece, "orientation", None)

    # Temporarily apply the move
    if move["action"] == "flip":
        # Flip stone → river or river → stone
        piece.side = "river" if piece.side == "stone" else "stone"
    elif move["action"] == "rotate":
        piece.orientation = "vertical" if original_orientation == "horizontal" else "horizontal"

    # Calculate flow from this piece
    flow_cells = agent_river_flow(board, x, y, x, y, player, rows, cols, score_cols)

    # Only consider safe cells (not in opponent score)
    safe_flow = [(fx, fy) for fx, fy in flow_cells if not is_opponent_score_cell(fx, fy, player, rows, cols, score_cols)]

    # Score = number of safe flow cells
    score = len(safe_flow)

    # Optional bonuses
    if any(is_own_score_cell(fx, fy, player, rows, cols, score_cols) for fx, fy in safe_flow):
        score += 10  # directly reaches goal
    if move["action"] == "flip" and piece.side == "river" and has_adjacent_river(board, x, y, rows, cols):
        score += 2  # connects to existing rivers

    # Revert the piece to original state
    piece.side = original_side
    piece.orientation = original_orientation

    return score

def in_bounds(x: int, y: int, rows: int, cols: int) -> bool:
    """Check if coordinates are within board boundaries."""
    return 0 <= x < cols and 0 <= y < rows


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
def agent_river_flow(
    board,
    rx: int,
    ry: int,
    sx: int,
    sy: int,
    player: str,
    rows: int,
    cols: int,
    score_cols: List[int],
) -> List[Tuple[int, int]]:
    """
    Simulate river flow from a given entry point (rx, ry). Correctly handles
    moving along rivers and switching directions at river intersections.

    Args:
        board: Current board state
        rx, ry: River entry point
        sx, sy: Source position (the piece that is moving; its own cell is ignored as an obstacle)
        player: Current player
        rows, cols: Board dimensions
        score_cols: Scoring column indices

    Returns:
        List of unique (x, y) coordinates where the piece can end up via river flow.
    """
    destinations = []
    visited = set()
    queue = [(rx, ry)]  # Old version: Use a list

    while queue:
        x, y = queue.pop(0)  # Old version: pop from the front (slow for large lists)
        if (x, y) in visited or not in_bounds(x, y, rows, cols):
            continue
        visited.add((x, y))

        cell = board[y][x]
        if cell is None:
            if not is_opponent_score_cell(x, y, player, rows, cols, score_cols):
                destinations.append((x, y))
            continue

        if cell.side != "river":
            continue

        # River flow directions
        dirs = (
            [(1, 0), (-1, 0)] if cell.orientation == "horizontal" else [(0, 1), (0, -1)]
        )

        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            while in_bounds(nx, ny, rows, cols):
                if is_opponent_score_cell(nx, ny, player, rows, cols, score_cols):
                    break

                next_cell = board[ny][nx]
                if next_cell is None:
                    destinations.append((nx, ny))
                    nx += dx
                    ny += dy
                    continue

                if nx == sx and ny == sy:
                    nx += dx
                    ny += dy
                    continue

                if next_cell.side == "river":
                    queue.append((nx, ny))
                    break
                break

    # Remove duplicates
    unique_destinations = []
    seen = set()
    for d in destinations:
        if d not in seen:
            seen.add(d)
            unique_destinations.append(d)

    return unique_destinations


def generate_1_step_moves(board, player, rows, cols, score_cols):
    """
    Generate simple 1-step moves for player's stones.
    """
    moves = []
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for y in range(rows):
        for x in range(cols):
            piece = board[y][x]
            if not piece or piece.owner != player:
                continue

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if not in_bounds(nx, ny, rows, cols):
                    continue
                if is_opponent_score_cell(nx, ny, player, rows, cols, score_cols):
                    continue
                if board[ny][nx] is None:
                    moves.append({"action": "move", "from": [x, y], "to": [nx, ny]})

    return moves


def generate_stone_pushes(board, player, rows, cols, score_cols):
    """
    Generate push moves where a stone pushes an adjacent opponent piece.
    """
    moves = []
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for y in range(rows):
        for x in range(cols):
            piece = board[y][x]
            if not piece or piece.owner != player or piece.side != "stone":
                continue

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if not in_bounds(nx, ny, rows, cols):
                    continue
                target = board[ny][nx]
                if not target or target.owner == player:
                    continue

                px, py = nx + dx, ny + dy
                if not in_bounds(px, py, rows, cols):
                    continue
                if board[py][px] is not None:
                    continue
                if is_opponent_score_cell(nx, ny, player, rows, cols, score_cols):
                    continue
                if is_own_score_cell(px, py, player, rows, cols, score_cols):
                    continue

                moves.append(
                    {
                        "action": "push",
                        "from": [x, y],
                        "to": [nx, ny],
                        "pushed_to": [px, py],
                    }
                )

    return moves


def generate_stone_flips(board, player, rows, cols, score_cols):
    """
    Generate flips of stones into rivers.
    """
    moves = []
    for y in range(rows):
        for x in range(cols):
            piece = board[y][x]
            if not piece or piece.owner != player or piece.side != "stone":
                continue

            # Save the original state of the piece
            original_side = piece.side
            original_orientation = getattr(piece, "orientation", None)

            for orientation in ("horizontal", "vertical"):
                # Temporarily change the piece to a river
                piece.side = "river"
                piece.orientation = orientation

                # Use the efficient flow calculator
                flow = agent_river_flow(
                    board, x, y, x, y, player, rows, cols, score_cols
                )

                # Check if the flow is safe (doesn't go into opponent score)
                if not any(
                    is_opponent_score_cell(fx, fy, player, rows, cols, score_cols)
                    for fx, fy in flow
                ):
                    moves.append(
                        {"action": "flip", "from": [x, y], "orientation": orientation}
                    )

            # CRITICAL: Revert the piece back to its original state immediately after the loop
            piece.side = original_side
            piece.orientation = original_orientation

    return moves


def generate_river_flips(board, player, rows, cols, score_cols):
    """
    Generate flips of river back to stone.
    """
    moves = []
    for y in range(rows):
        for x in range(cols):
            piece = board[y][x]
            if not piece or piece.owner != player or piece.side != "river":
                continue
            moves.append({"action": "flip", "from": [x, y]})
    return moves


def generate_river_rotates(board, player, rows, cols, score_cols):
    """
    Generate rotations of river orientation.
    """
    moves = []
    for y in range(rows):
        for x in range(cols):
            piece = board[y][x]
            if not piece or piece.owner != player or piece.side != "river":
                continue

            # Save the original orientation
            original_orientation = piece.orientation

            # Calculate the new orientation
            new_orientation = (
                "vertical" if original_orientation == "horizontal" else "horizontal"
            )

            # Temporarily change the piece's orientation
            piece.orientation = new_orientation

            # Use the efficient flow calculator
            flow = agent_river_flow(board, x, y, x, y, player, rows, cols, score_cols)

            # Check if the new flow is safe
            if not any(
                is_opponent_score_cell(dx, dy, player, rows, cols, score_cols)
                for dx, dy in flow
            ):
                moves.append({"action": "rotate", "from": [x, y]})

            # CRITICAL: Revert the piece back to its original orientation
            piece.orientation = original_orientation

    return moves


def generate_stone_in_river_flows(board, player, rows, cols, score_cols):
    """
    Generate moves where a stone moves onto a river and then flows.
    """
    moves = []
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # 1-step directions

    for y in range(rows):
        for x in range(cols):
            piece = board[y][x]
            if not piece or piece.owner != player or piece.side != "stone":
                continue

            # First, check all 4 adjacent cells for a river
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if not in_bounds(nx, ny, rows, cols):
                    continue

                # The adjacent cell must contain a river to step onto
                adj_cell = board[ny][nx]
                if not adj_cell or adj_cell.side != "river":
                    continue  # Skip if not a river

                # Now, calculate flow starting FROM THE RIVER CELL (nx, ny)
                # The source piece is at (x, y)
                flow = agent_river_flow(
                    board, nx, ny, x, y, player, rows, cols, score_cols
                )

                for fx, fy in flow:
                    in_goal_band = in_goal_side_rows(
                        fx, fy, player, rows, cols, score_cols
                    )
                    if not is_opponent_score_cell(
                        fx, fy, player, rows, cols, score_cols
                    ):
                        if in_goal_band:
                            moves.append(
                                {"action": "move", "from": [x, y], "to": [fx, fy]}
                            )

    # sort moves based on some function
    return moves


def generate_river_moves(board, player, rows, cols, score_cols):
    """
    Generate moves where a river piece moves onto a river and then flows.
    """
    moves = []
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # 1-step directions

    for y in range(rows):
        for x in range(cols):
            piece = board[y][x]
            if not piece or piece.owner != player or piece.side != "river":
                continue

            # First, check all 4 adjacent cells for a river (any player's river)
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if not in_bounds(nx, ny, rows, cols):
                    continue

                # The adjacent cell must contain a river to step onto
                adj_cell = board[ny][nx]
                if not adj_cell or adj_cell.side != "river":
                    continue  # Skip if not a river

                # Now, calculate flow starting FROM THE ADJACENT RIVER CELL (nx, ny)
                # The source piece is at (x, y)
                flow = agent_river_flow(
                    board, nx, ny, x, y, player, rows, cols, score_cols
                )

                for fx, fy in flow:
                    # Check if the flow destination is valid (not in opponent score)
                    in_goal_band = in_goal_side_rows(
                        fx, fy, player, rows, cols, score_cols
                    )
                    if not is_opponent_score_cell(
                        fx, fy, player, rows, cols, score_cols
                    ):
                        if in_goal_band:
                            moves.append(
                                {"action": "move", "from": [x, y], "to": [fx, fy]}
                            )
    return moves


def generate_river_pushes(board, player, rows, cols, score_cols):
    """
    Generate moves where a river piece pushes an adjacent opponent stone along the river flow.
    The river piece stays in place and flips to stone after pushing.
    """
    moves = []
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # Check all 4 adjacent cells

    for y in range(rows):
        for x in range(cols):
            river_piece = board[y][x]
            # Check if current cell has one of our river pieces
            if (
                not river_piece
                or river_piece.owner != player
                or river_piece.side != "river"
            ):
                continue

            # Check all four adjacent cells for opponent stones to push
            for dx, dy in directions:
                stone_x, stone_y = x + dx, y + dy  # Coordinates of adjacent cell

                if not in_bounds(stone_x, stone_y, rows, cols):
                    continue

                stone = board[stone_y][stone_x]
                # Verify adjacent cell contains opponent's stone
                if not stone or stone.owner == player or stone.side != "stone":
                    continue

                # Calculate all possible flow destinations for the STONE starting from its position
                # The stone's own position (stone_x, stone_y) is ignored as obstacle
                flow_destinations = agent_river_flow(
                    board,
                    stone_x,
                    stone_y,
                    stone_x,
                    stone_y,
                    player,
                    rows,
                    cols,
                    score_cols,
                )

                # Check each possible destination for validity
                for dest_x, dest_y in flow_destinations:
                    # Destination must be empty and not in any score area
                    if (
                        board[dest_y][dest_x] is None
                        and not is_opponent_score_cell(
                            dest_x, dest_y, player, rows, cols, score_cols
                        )
                        and not is_own_score_cell(
                            dest_x, dest_y, player, rows, cols, score_cols
                        )
                        and not has_adjacent_river(board, dest_x, dest_y, rows, cols)
                    ):

                        moves.append(
                            {
                                "action": "push",
                                "from": [x, y],  # River piece position
                                "to": [stone_x, stone_y],  # Opponent stone position
                                "pushed_to": [dest_x, dest_y],  # Where stone lands
                            }
                        )

    return moves


def generate_all_moves(board, player, rows, cols, score_cols):
    moves = []
    moves.extend(generate_river_moves(board, player, rows, cols, score_cols))
    moves.extend(generate_stone_in_river_flows(board, player, rows, cols, score_cols))
    moves.extend(generate_stone_flips(board, player, rows, cols, score_cols))
    moves.extend(generate_river_flips(board, player, rows, cols, score_cols))
    moves.extend(generate_river_rotates(board, player, rows, cols, score_cols))
    moves.extend(generate_river_pushes(board, player, rows, cols, score_cols))
    moves.extend(generate_stone_pushes(board, player, rows, cols, score_cols))
    moves.extend(generate_1_step_moves(board, player, rows, cols, score_cols))
    
    river_moves = [m for m in moves if m["action"] in ("flip", "rotate")]
    river_moves = sorted(river_moves, key=lambda m: score_flow_move(m, board, player, rows, cols, score_cols), reverse=True)
    river_moves = river_moves[:3]
    
    moves_with_to = [m for m in moves if "to" in m]
    moves_with_to = sorted(moves_with_to,key=lambda m: score_move(m, board, player, rows, cols, score_cols),reverse=True)
    moves_with_to = moves_with_to[:3]
    
    moves = river_moves + moves_with_to
    print("moves",len(moves))
    return moves


# ==================== BOARD EVALUATION ====================


def eval_stone_progress(board, player, rows, cols):
    """
    Reward stones that are closer to the opponent's goal row.
    Simple positional scoring.
    """
    score = 0
    for y in range(rows):
        for x in range(cols):
            piece = board[y][x]
            if piece and piece.owner == player and piece.side == "stone":
                # Basic positional scoring
                if player == "circle":
                    score += y  # circle moves downward
                else:
                    score += rows - y  # cross moves upward
    return score


def eval_central_positions(board, player, cols):
    score = 0
    mid_col = cols // 2
    for row in board:
        for x in range(cols):
            piece = row[x]
            if piece and piece.owner == player and piece.side == "stone":
                score += 1 * (mid_col - abs(mid_col - x))  # closer to center = higher
    return score


def eval_stones_in_goal(board, player, score_cols):
    """
    Bonus for stones already in goal columns.
    """
    score = 0
    for y in range(len(board)):
        for x in score_cols:
            piece = board[y][x]
            if piece and piece.owner == player and piece.side == "stone":
                score += 1  # fixed bonus
    return score


def eval_stones_on_river(board, player):
    """
    Bonus for stones currently on river cells.
    """
    score = 0
    for row in board:
        for piece in row:
            if piece and piece.owner == player and piece.side == "river":
                score += 1  # small bonus
    return score


def eval_instant_score_potential(board, player, rows, cols, score_cols):
    """
    Reward stones that can reach goal columns in 1 river flow move.
    """
    score = 0
    for y in range(rows):
        for x in range(cols):
            piece = board[y][x]
            if piece and piece.owner == player and piece.side == "stone":
                # Check all river flows from adjacent rivers
                directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if in_bounds(nx, ny, rows, cols):
                        neighbor = board[ny][nx]
                        if (
                            neighbor
                            and neighbor.owner == player
                            and neighbor.side == "river"
                        ):
                            flow = agent_river_flow(
                                board, nx, ny, x, y, player, rows, cols, score_cols
                            )
                            if any(d[1] in score_cols for d in flow):
                                score += 1  # big bonus for potential instant scoring
    return score


def evaluate_simple(board, player, rows, cols, score_cols):
    """
    Combine all simple evaluation functions with given weights.
    """
    weights = {
        "progress": 50,  # Reward moving stones toward opponent's goal
        "goal": 100,  # Strong bonus for stones already in goal columns
        "river": 5,  # Small bonus for stones on river
        "central": 20,  # Encourage central positioning
        "instant_score": 30,  # High reward for stones that can score via river in 1 move
    }
    total = 0
    total += weights["progress"] * eval_stone_progress(board, player, rows, cols)
    total += weights["goal"] * eval_stones_in_goal(board, player, score_cols)
    total += weights["river"] * eval_stones_on_river(board, player)
    total += weights["central"] * eval_central_positions(board, player, cols)
    total += weights["instant_score"] * eval_instant_score_potential(
        board, player, rows, cols, score_cols
    )
    return total


def count_stones_in_scoring_area(
    board: List[List[Any]], player: str, rows: int, cols: int, score_cols: List[int]
) -> int:
    """Count how many stones a player has in their scoring area."""
    count = 0

    if player == "circle":
        score_row = top_score_row()
    else:
        score_row = bottom_score_row(rows)

    for x in score_cols:
        if in_bounds(x, score_row, rows, cols):
            piece = board[score_row][x]
            if piece and piece.owner == player and piece.side == "stone":
                count += 1

    return count


def basic_evaluate_board(
    board: List[List[Any]], player: str, rows: int, cols: int, score_cols: List[int]
) -> float:
    """
    Basic board evaluation function.

    Returns a score where higher values are better for the given player.
    Students can use this as a starting point and improve it.
    """
    score = 0.0
    opponent = get_opponent(player)

    # Count stones in scoring areas
    player_scoring_stones = count_stones_in_scoring_area(
        board, player, rows, cols, score_cols
    )
    opponent_scoring_stones = count_stones_in_scoring_area(
        board, opponent, rows, cols, score_cols
    )

    score += player_scoring_stones * 100
    score -= opponent_scoring_stones * 100

    # Count total pieces and positional factors
    for y in range(rows):
        for x in range(cols):
            piece = board[y][x]
            if piece and piece.owner == player and piece.side == "stone":
                # Basic positional scoring
                if player == "circle":
                    score += (rows - y) * 0.1
                else:
                    score += y * 0.1

    return score


def agent_apply_move(
    board,
    move: Dict[str, Any],
    player: str,
    rows: int,
    cols: int,
    score_cols: List[int],
) -> Tuple[bool, str]:
    """
    Apply a move to a board copy for simulation purposes.

    Args:
        board: Board state to modify
        move: Move dictionary
        player: Current player
        rows, cols: Board dimensions
        score_cols: Scoring column indices

    Returns:
        (success: bool, message: str)
    """
    action = move.get("action")

    if action == "move":
        return _apply_move_action(board, move, player, rows, cols, score_cols)
    elif action == "push":
        return _apply_push_action(board, move, player, rows, cols, score_cols)
    elif action == "flip":
        return _apply_flip_action(board, move, player, rows, cols, score_cols)
    elif action == "rotate":
        return _apply_rotate_action(board, move, player, rows, cols, score_cols)

    return False, "unknown action"


def _apply_move_action(board, move, player, rows, cols, score_cols):
    """Apply a move action."""
    fr = move.get("from")
    to = move.get("to")
    if not fr or not to:
        return False, "bad move format"

    fx, fy = int(fr[0]), int(fr[1])
    tx, ty = int(to[0]), int(to[1])

    if not in_bounds(fx, fy, rows, cols) or not in_bounds(tx, ty, rows, cols):
        return False, "out of bounds"

    if is_opponent_score_cell(tx, ty, player, rows, cols, score_cols):
        return False, "cannot move into opponent score cell"

    piece = board[fy][fx]
    if piece is None or piece.owner != player:
        # if piece is None:
        #     return False, "None piece"
        # else:
        #     return False, "owner diff"
        return False, "invalid piece"

    if board[ty][tx] is None:
        # Simple move
        board[ty][tx] = piece
        board[fy][fx] = None
        return True, "moved"

    # Move with push
    pushed_to = move.get("pushed_to")
    if not pushed_to:
        return False, "destination occupied; pushed_to required"

    ptx, pty = int(pushed_to[0]), int(pushed_to[1])
    dx, dy = tx - fx, ty - fy

    if (ptx, pty) != (tx + dx, ty + dy):
        return False, "invalid pushed_to"

    if not in_bounds(ptx, pty, rows, cols):
        return False, "pushed_to out of bounds"

    if is_opponent_score_cell(ptx, pty, player, rows, cols, score_cols):
        return False, "cannot push into opponent score"

    if board[pty][ptx] is not None:
        return False, "pushed_to not empty"

    board[pty][ptx] = board[ty][tx]
    board[ty][tx] = piece
    board[fy][fx] = None
    return True, "moved with push"


def _apply_push_action(board, move, player, rows, cols, score_cols):
    """Apply a push action."""
    fr = move.get("from")
    to = move.get("to")
    pushed_to = move.get("pushed_to")

    if not fr or not to or not pushed_to:
        return False, "bad push format"

    fx, fy = int(fr[0]), int(fr[1])
    tx, ty = int(to[0]), int(to[1])
    px, py = int(pushed_to[0]), int(pushed_to[1])

    if is_opponent_score_cell(
        tx, ty, player, rows, cols, score_cols
    ) or is_opponent_score_cell(px, py, player, rows, cols, score_cols):
        return False, "push would move into opponent score cell"

    if not (
        in_bounds(fx, fy, rows, cols)
        and in_bounds(tx, ty, rows, cols)
        and in_bounds(px, py, rows, cols)
    ):
        return False, "out of bounds"

    piece = board[fy][fx]
    if piece is None or piece.owner != player:
        return False, "invalid piece"

    if board[ty][tx] is None:
        return False, "'to' must be occupied"

    if board[py][px] is not None:
        return False, "pushed_to not empty"

    board[py][px] = board[ty][tx]
    board[ty][tx] = board[fy][fx]
    board[fy][fx] = None
    return True, "pushed"


def _apply_flip_action(board, move, player, rows, cols, score_cols):
    """Apply a flip action."""
    fr = move.get("from")
    if not fr:
        return False, "bad flip format"

    fx, fy = int(fr[0]), int(fr[1])
    piece = board[fy][fx]

    if piece is None or piece.owner != player:
        return False, "invalid piece"

    if piece.side == "stone":
        # Stone to river
        orientation = move.get("orientation")
        if orientation not in ("horizontal", "vertical"):
            return False, "stone->river needs orientation"

        # Check if new river would allow flow into opponent score
        board[fy][fx].side = "river"
        board[fy][fx].orientation = orientation
        flow = agent_river_flow(board, fx, fy, fx, fy, player, rows, cols, score_cols)

        # Revert for safety check
        board[fy][fx].side = "stone"
        board[fy][fx].orientation = None

        for dx, dy in flow:
            if is_opponent_score_cell(dx, dy, player, rows, cols, score_cols):
                return False, "flip would allow flow into opponent score cell"

        # Apply flip
        board[fy][fx].side = "river"
        board[fy][fx].orientation = orientation
        return True, "flipped to river"
    else:
        # River to stone
        board[fy][fx].side = "stone"
        board[fy][fx].orientation = None
        return True, "flipped to stone"


def _apply_rotate_action(board, move, player, rows, cols, score_cols):
    """Apply a rotate action."""
    fr = move.get("from")
    if not fr:
        return False, "bad rotate format"

    fx, fy = int(fr[0]), int(fr[1])
    piece = board[fy][fx]

    if piece is None or piece.owner != player or piece.side != "river":
        return False, "invalid rotate"

    # Try rotation
    old_orientation = piece.orientation
    piece.orientation = "horizontal" if piece.orientation == "vertical" else "vertical"

    # Check flow safety after rotation
    flow = agent_river_flow(board, fx, fy, fx, fy, player, rows, cols, score_cols)

    for dx, dy in flow:
        if is_opponent_score_cell(dx, dy, player, rows, cols, score_cols):
            # Revert rotation
            piece.orientation = old_orientation
            return False, "rotate would allow flow into opponent score cell"

    return True, "rotated"


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
    success, message = agent_apply_move(
        board_copy, move, player, rows, cols, score_cols
    )

    if success:
        return True, board_copy
    else:
        return False, message


def evaluate(
    board: List[List[Any]], player: str, rows: int, cols: int, score_cols: List[int]
):
    """
    Basic board evaluation function.

    Returns a score where higher values are better for the given player.
    Students can use this as a starting point and improve it.
    """
    score = 0.0
    opponent = get_opponent(player)

    # Count stones in scoring areas
    player_scoring_stones = count_stones_in_scoring_area(
        board, player, rows, cols, score_cols
    )
    opponent_scoring_stones = count_stones_in_scoring_area(
        board, opponent, rows, cols, score_cols
    )

    score += player_scoring_stones * 100
    score -= opponent_scoring_stones * 100

    # Count total pieces and positional factors
    for y in range(rows):
        for x in range(cols):
            piece = board[y][x]
            if piece and piece.owner == player and piece.side == "stone":
                # Basic positional scoring
                if player == "circle":
                    score += y * 1
                else:
                    score += rows - y

    return score


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

    moves = generate_all_moves(board, player, rows, cols, score_cols)
    print(f"Minimax for {player}, evaluating {len(moves)} moves")
    for move in moves:
        success, result = simulate_move(board, move, player, rows, cols, score_cols)
        if success:
            child_board = result
            childVal = minValue(
                child_board,
                opponent,  # Next player (minimizing)
                player,  # Original player (for evaluation)
                rows,
                cols,
                score_cols,
                -float("inf"),
                float("inf"),
                2,
            )
            print(f"Move {move} -> Score: {childVal}")

            if childVal > maxVal:
                maxVal = childVal
                maxMove = move
                print(f"New best: {move} with score {maxVal}")
        else:
            print(f"Move failed: {move}, Error: {result}")

    print(f"Selected: {maxMove} with score {maxVal}")
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
    if ply == 0:
        return evaluate_simple(board, original_player, rows, cols, score_cols)

    # GENERATE moves for the CURRENT player (minimizing)
    moves = generate_all_moves(board, current_player, rows, cols, score_cols)

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
                ply-1,
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
    if ply == 0:
        return evaluate_simple(board, original_player, rows, cols, score_cols)

    # GENERATE moves for the CURRENT player (maximizing)
    moves = generate_all_moves(board, current_player, rows, cols, score_cols)

    if not moves:  # No moves available
        return evaluate_simple(board, original_player, rows, cols, score_cols)

    maxVal = -float("inf")
    opponent = get_opponent(current_player)

    for move in moves:
        success, child_board = simulate_move(
            board, move, current_player, rows, cols, score_cols
        )
        if success:
            childVal = minValue(
                child_board,
                opponent,  # Next player (minimizing)
                original_player,  # Keep original player
                rows,
                cols,
                score_cols,
                alpha,
                beta,
                ply,
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

    def choose(
        self, board: List[List[Any]], rows: int, cols: int, score_cols: List[int]
    ) -> Optional[Dict[str, Any]]:
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

        if not my_move:
            return None

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
        move = agent.choose(board, rows, cols, score_cols)

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
