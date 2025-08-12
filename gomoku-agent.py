import re
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from gomoku import Agent
from gomoku.core.models import Player

class GomokuAgent(Agent):
    """
    A Gomoku AI agent that uses a language model to make strategic moves.
    Inherits from the base Agent class provided by the Gomoku framework.
    """

    def _setup(self):
        """
        Initialize the agent by setting up the language model client.
        This method is called once when the agent is created.
        """
        # Load the tokenizer and model from Hugging Face using the HF_TOKEN for authentication
        model_name = "deepseek/deepseek-r1-0528-qwen3-8b"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=os.getenv("HF_TOKEN"))
        self.model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=os.getenv("HF_TOKEN"))

    async def get_move(self, game_state):
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

        # Enhanced Prompt Engineering for better decision-making
        messages = [
            {
                "role": "system",
                "content": f"""You are an advanced Gomoku AI agent skilled at strategic decision-making. You are playing on an 8x8 board, where the goal is to align five stones (either 'X' or 'O') in a row, vertically, horizontally, or diagonally.
You are currently playing as {player}, and your opponent is {rival}.
Analyze the current board configuration and consider the following:
- Block the opponent if they are about to win.
- Make moves that help you get five-in-a-row or set up your next moves.
- You can respond with only one valid move.

Your task is to provide the best move to make based on the current state of the board.

Here is the current board:

{board_str}

Your move should follow this format (without explanation): 
{{ "row": <row_number>, "col": <col_number> }}"""
            },
            {
                "role": "user",
                "content": f"Based on the board above, please provide your best move to win the game or block the opponent. Remember, winning requires five consecutive stones in a row, either vertically, horizontally, or diagonally."
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

        # Fallback: if LLM response is invalid, choose the first available legal move
        return game_state.get_legal_moves()[0]
