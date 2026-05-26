import math
import random
import time
import numpy as np
from connect4.policy import Policy


class MCTSNode:
    def __init__(self, board: np.ndarray, current_player: int, parent=None, action=None):
        self.board = board
        self.current_player = current_player
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = [c for c in range(7) if board[0, c] == 0]

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def is_terminal(self):
        return len(self.untried_actions) == 0 and len(self.children) == 0

    def ucb1(self, c_param=1.41):
        if self.visits == 0:
            return float('inf')
        return (self.value / self.visits) + c_param * math.sqrt(math.log(self.parent.visits) / self.visits)


class AhaMCTS_Cautious(Policy):
    """
    Variación 2: MCTS con defensa de dos movimientos.
    Antes de entrar al árbol MCTS, analiza si el oponente tiene una amenaza
    doble (puede ganar por dos columnas distintas en su próximo turno).
    Si la detecta, prioriza el bloqueo de esas columnas en lugar de explorar
    libremente, evitando trampas que el MCTS estándar puede ignorar.
    """

    def __init__(self):
        self.time_limit = 0.15

    def mount(self, timeout: float = 30.0) -> None:
        pass

    def act(self, s: np.ndarray) -> int:
        available = [c for c in range(7) if s[0, c] == 0]
        if not available:
            return 0
        if len(available) == 1:
            return available[0]

        pieces = np.count_nonzero(s)
        my_player = 1 if pieces % 2 == 0 else -1

        # Victoria inmediata
        for col in available:
            board_copy = s.copy()
            row = self._drop(board_copy, col, my_player)
            if row >= 0 and self._check_win(board_copy, row, col, my_player):
                return col

        # Bloqueo inmediato
        for col in available:
            board_copy = s.copy()
            row = self._drop(board_copy, col, -my_player)
            if row >= 0 and self._check_win(board_copy, row, col, -my_player):
                return col

        # Defensa de dos movimientos: detectar amenaza doble del oponente
        forced = self._find_double_threat(s, available, my_player)
        if forced is not None:
            return forced

        root = MCTSNode(board=s.copy(), current_player=my_player)
        end_time = time.time() + self.time_limit

        while time.time() < end_time:
            node = root
            while node.is_fully_expanded() and not node.is_terminal():
                node = max(node.children, key=lambda c: c.ucb1())

            if not node.is_fully_expanded():
                action = random.choice(node.untried_actions)
                node.untried_actions.remove(action)
                next_board = node.board.copy()
                self._drop(next_board, action, node.current_player)
                child = MCTSNode(
                    board=next_board,
                    current_player=-node.current_player,
                    parent=node,
                    action=action
                )
                node.children.append(child)
                node = child

            reward = self._simulate(node.board, node.current_player, my_player, node.action)

            curr = node
            while curr is not None:
                curr.visits += 1
                curr.value += reward
                curr = curr.parent

        if not root.children:
            return random.choice(available)
        return max(root.children, key=lambda c: c.visits).action

    def _find_double_threat(self, board: np.ndarray, available: list, my_player: int):
        """
        Detecta si el oponente tiene amenaza doble: tras cualquier jugada nuestra,
        el oponente puede ganar por dos columnas distintas.
        Si encontramos una jugada nuestra que evita eso, la retornamos.
        Si no hay ninguna que la evite, retornamos la que deja menos amenazas.
        """
        opponent = -my_player
        safe_moves = []

        for my_col in available:
            # Simular mi jugada
            board_after = board.copy()
            self._drop(board_after, my_col, my_player)

            # Contar cuántas columnas ganaría el oponente a continuación
            opp_available = [c for c in range(7) if board_after[0, c] == 0]
            opp_wins = []
            for opp_col in opp_available:
                tmp = board_after.copy()
                row = self._drop(tmp, opp_col, opponent)
                if row >= 0 and self._check_win(tmp, row, opp_col, opponent):
                    opp_wins.append(opp_col)

            if len(opp_wins) == 0:
                safe_moves.append(my_col)
            elif len(opp_wins) == 1:
                safe_moves.append(my_col)  # Habrá un bloqueo disponible

        # Si hay jugadas que dejan al oponente sin amenaza doble, dejar que MCTS decida entre ellas
        # Solo forzamos si TODAS las jugadas dejan amenaza doble (situación crítica)
        if len(safe_moves) > 0 and len(safe_moves) < len(available):
            # Hay jugadas peligrosas — retornar la mejor jugada segura directamente
            return random.choice(safe_moves)

        return None  # No hay amenaza doble o todas son seguras: dejar decidir al MCTS

    def _simulate(self, board: np.ndarray, current_player: int, original_player: int, last_action) -> float:
        sim_board = board.copy()
        player = current_player

        if last_action is not None:
            for r in range(6):
                if sim_board[r, last_action] != 0:
                    if self._check_win(sim_board, r, last_action, -player):
                        return 1.0 if -player == original_player else -1.0
                    break

        max_steps = 42
        steps = 0
        while steps < max_steps:
            available = [c for c in range(7) if sim_board[0, c] == 0]
            if not available:
                return 0.0
            action = random.choice(available)
            row = self._drop(sim_board, action, player)
            if self._check_win(sim_board, row, action, player):
                return 1.0 if player == original_player else -1.0
            player *= -1
            steps += 1

        return 0.0

    def _drop(self, board: np.ndarray, col: int, player: int) -> int:
        for row in range(5, -1, -1):
            if board[row, col] == 0:
                board[row, col] = player
                return row
        return -1

    def _check_win(self, board: np.ndarray, row: int, col: int, player: int) -> bool:
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < 6 and 0 <= c < 7 and board[r, c] == player:
                    count += 1
                    r += sign * dr
                    c += sign * dc
            if count >= 4:
                return True
        return False
