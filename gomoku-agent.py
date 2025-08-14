import random
from typing import Tuple
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

        # # Define system prompt for the agent
        # self.system_prompt = self._create_system_prompt()

        # Create the LLM client using OpenAIGomokuClient with the specified model
        self.llm = OpenAIGomokuClient(model="qwen/qwen3-8b")

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
                row, col = (move["row"], move["col"])

                # Validate that the proposed move is legal
                if game_state.is_valid_move(row, col):
                    return (row, col)
        except json.JSONDecodeError as e:
            # If JSON parsing fails, continue to fallback strategy
            pass

        # Fallback: if LLM response is invalid, choose a random legal move
        legal_moves = game_state.get_legal_moves()
        return random.choice(legal_moves)

    # def _create_system_prompt(self) -> str:
    #     """Create the system prompt to set the context for the agent."""
    #     return (
    #         "You are a highly skilled Gomoku AI agent playing on an 8x8 board. "
    #         "The goal of the game is to get five consecutive stones in a row, "
    #         "either horizontally, vertically, or diagonally. Your moves should always aim to "
    #         "either block the opponent from winning or advance towards winning yourself. "
    #         "In the event that there is no immediate winning or blocking move, select the best strategic move. "
    #         "You should only provide your move as row and column coordinates, formatted as {'row': <row_number>, 'col': <col_number>}. "
    #         "Never explain your move in text—only provide the coordinates of your move."
    #     )