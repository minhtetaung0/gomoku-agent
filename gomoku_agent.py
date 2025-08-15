import random
from typing import Tuple, List, Optional
import re
import json

# Import the game framework components
from gomoku.agents.base import Agent
from gomoku.core.models import GameState, Player
from gomoku.llm.openai_client import OpenAIGomokuClient


class GomokuAgent(Agent):
    """A Gomoku LLM agent that uses a language model to make strategic moves."""

    def _setup(self):
        """
        Initialize the agent by setting up the language model client.
        This method is called once when the agent is created.
        """
        # self.system_prompt = self._create_system_prompt()
        self.llm = OpenAIGomokuClient(model="qwen/qwen3-8b", temperature=0)

    # ----------------------------
    # Minimal smart fallback tools
    # ----------------------------
    def _get_board(self, game_state: GameState) -> Optional[List[List[str]]]:
        """Safely access the internal board if available (list[list[str]])."""
        return getattr(game_state, "board", None)

    def _five_after_move(self, board: List[List[str]], r: int, c: int, sym: str) -> bool:
        """Check if placing sym at (r,c) yields five in a row."""
        n = len(board)
        if board[r][c] != '.':
            return False
        board[r][c] = sym
        try:
            # count consecutive in 4 directions (pairs): (dr,dc)
            dirs = [(1,0), (0,1), (1,1), (1,-1)]
            for dr, dc in dirs:
                cnt = 1
                # forward
                i, j = r + dr, c + dc
                while 0 <= i < n and 0 <= j < n and board[i][j] == sym:
                    cnt += 1
                    i += dr; j += dc
                # backward
                i, j = r - dr, c - dc
                while 0 <= i < n and 0 <= j < n and board[i][j] == sym:
                    cnt += 1
                    i -= dr; j -= dc
                if cnt >= 5:
                    return True
            return False
        finally:
            board[r][c] = '.'  # undo

    def _longest_line_after(self, board: List[List[str]], r: int, c: int, sym: str) -> int:
        """Return the longest contiguous line length achieved at (r,c) after placing sym."""
        n = len(board)
        board[r][c] = sym
        best = 1
        dirs = [(1,0), (0,1), (1,1), (1,-1)]
        for dr, dc in dirs:
            cnt = 1
            i, j = r + dr, c + dc
            while 0 <= i < n and 0 <= j < n and board[i][j] == sym:
                cnt += 1
                i += dr; j += dc
            i, j = r - dr, c - dc
            while 0 <= i < n and 0 <= j < n and board[i][j] == sym:
                cnt += 1
                i -= dr; j -= dc
            if cnt > best:
                best = cnt
        board[r][c] = '.'
        return best

    def _nearest_my_stone_dist(self, board: List[List[str]], r: int, c: int, sym: str) -> int:
        """Chebyshev distance to the nearest of my stones (lower is better)."""
        n = len(board)
        best = 10**9
        for i in range(n):
            for j in range(n):
                if board[i][j] == sym:
                    d = max(abs(i - r), abs(j - c))
                    if d < best:
                        best = d
        # If no stones yet, return a large number (will be balanced by center bias)
        return best if best != 10**9 else 10**6

    def _center_bias(self, n: int, r: int, c: int) -> float:
        """Negative distance to center to prefer central lanes on empty/early boards."""
        center = (n - 1) / 2.0
        # Use Chebyshev distance for grid symmetry
        d = max(abs(r - center), abs(c - center))
        return -d

    def _smart_fallback(self, game_state: GameState, me: str, opp: str) -> Tuple[int, int]:
        """
        Smarter-than-random fallback:
        1) Win now
        2) Block opponent's immediate win
        3) Heuristic score: longest line, adjacency, center bias
        """
        legal = game_state.get_legal_moves()
        n = game_state.board_size

        board = self._get_board(game_state)

        # If we cannot access the board grid, choose a deterministic center-oriented move.
        if board is None:
            cx, cy = (n - 1) / 2.0, (n - 1) / 2.0
            legal.sort(key=lambda rc: (abs(rc[0] - cx) + abs(rc[1] - cy), rc[0], rc[1]))
            return legal[0]

        # 1) Win immediately if possible
        for r, c in legal:
            if self._five_after_move(board, r, c, me):
                return (r, c)

        # 2) Block opponent's immediate win
        opp_win_cells = []
        for r, c in legal:
            if self._five_after_move(board, r, c, opp):
                opp_win_cells.append((r, c))
        if opp_win_cells:
            opp_win_cells.sort()  # determinism: smallest (row, col)
            return opp_win_cells[0]

        # 3) Heuristic scoring
        #    Score = 100*longest_line + 2*(-nearest_my_dist) + 1*center_bias
        #    (weights are small and simple to keep code minimal)
        best_score = -10**9
        best_moves = []

        for r, c in legal:
            # prefer adjacent-to-my-stones and extending my lines
            ll = self._longest_line_after(board, r, c, me)
            nd = self._nearest_my_stone_dist(board, r, c, me)
            cb = self._center_bias(n, r, c)
            score = 100 * ll + 2 * (-nd) + cb

            if score > best_score:
                best_score = score
                best_moves = [(r, c)]
            elif score == best_score:
                best_moves.append((r, c))

        best_moves.sort()  # determinism tie-break
        return best_moves[0] if best_moves else random.choice(legal)

    async def get_move(self, game_state: GameState) -> Tuple[int, int]:
        """
        Generate the next move for the current game state using an LLM.

        Args:
            game_state: Current state of the Gomoku game board

        Returns:
            tuple: (row, col) coordinates of the chosen move
        """
        # Get the current player's symbol (e.g., 'X' or 'O')
        player = self.player.value

        # Determine the opponent's symbol by checking which player we are
        rival = (Player.WHITE if self.player == Player.BLACK else Player.BLACK).value

        # Convert the game board to a human-readable string format
        board_str = game_state.format_board("standard")
        board_size = game_state.board_size

        # Prepare the conversation messages for the language model
        messages = [
            {
                "role": "system",
                "content": f"""
You are a ruthless Gomoku tactician. Board: 8x8. Win condition: 5 in a row (rows/cols/diagonals).
You play as {player}. Opponent is {rival}. You must output ONE legal move only.

## Hard Priorities (apply in order)
1) WIN NOW: If you can make exactly five-in-a-row this move, play that cell.
2) BLOCK LOSS: If opponent can win next move with one placement, block that cell.
3) CREATE FORKS: Prefer moves that create two simultaneous threats (e.g., two open fours or four+broken-four).
4) POWER FOURS/THREES:
   • Extend your open four or broken four.
   • Block opponent open four or double-three/double-threat.
   • Extend your open three to open four.
5) SHAPE/PLACEMENT HEURISTICS (when no immediate tactics above apply):
   • Prefer extending your longest continuous line(s).
   • Prefer moves adjacent (Chebyshev distance ≤ 1) to your stones over isolated ones.
   • Prefer center/central lanes early if options are equal.
   • Avoid plays that are two or more cells away from all stones unless they create a new high-threat line.

## Legality & Determinism
- Move must be empty and inside the board (0–{board_size-1} for row and col).
- If multiple moves are tied by the same highest priority, choose the one with the smallest (row, col) lexicographically to be deterministic.

## Output Format (STRICT)
Return ONLY a single compact JSON object on one line:
{{"row": <int>, "col": <int>}}
No explanations. No analysis. No Markdown. No code fences. No 'think' tags. No extra keys.
Here is the current board:

{board_str}
"""
            }
        ]

        # Send the messages to the language model and get the response
        content = await self.llm.complete(messages)

        # Parse the LLM response to extract move coordinates
        try:
            # Use regex to find JSON-like content in the response
            if m := re.search(r"{[^}]+}", content, re.DOTALL):
                # Parse the JSON to extract row and column
                move = json.loads(m.group(0))
                row, col = (move["row"], "col" in move and move["col"] or move.get("column"))
                # normalize if different key names are ever used
                if isinstance(col, bool):  # guard against weird parsing
                    col = move["col"]
                # Validate that the proposed move is legal
                if game_state.is_valid_move(row, col):
                    return (row, col)
        except Exception:
            pass

        # Smarter fallback (win/block/heuristic) instead of pure random
        # Assumes '.' is empty on the internal board (common in Gomoku engines).
        # If internal board not available, center-biased deterministic choice is used.
        me = player
        opp = rival
        return self._smart_fallback(game_state, me, opp)
