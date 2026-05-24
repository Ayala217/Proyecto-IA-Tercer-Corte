import math
import random
import time
import numpy as np
from connect4.policy import Policy

class MCTSNode:
    def __init__(self, board: np.ndarray, current_player: int, parent=None, action=None):
        self.board = board
        self.current_player = current_player  # Jugador que debe mover AHORA
        self.parent = parent
        self.action = action                  # Acción que llevó a este nodo
        self.children = []
        self.visits = 0
        self.value = 0.0
        # Acciones legales (columnas no llenas)
        self.untried_actions = [c for c in range(7) if board[0, c] == 0]

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def is_terminal(self):
        # Es terminal si ya no hay jugadas (empate) 
        # (La victoria se chequea antes de crear el nodo para ahorrar tiempo)
        return len(self.untried_actions) == 0 and len(self.children) == 0

    def ucb1(self, c_param=1.41):
        if self.visits == 0:
            return float('inf')
        # UCB1 estándar
        return (self.value / self.visits) + c_param * math.sqrt(math.log(self.parent.visits) / self.visits)


class AhaMCTS(Policy):

    def __init__(self):
        self.time_limit = 0.15 # Segundos para pensar por jugada

    def mount(self, timeout: float = 30.0) -> None:
        # En MCTS no pre-entrenamos todo el juego al inicio.
        # Guardamos nuestras energías para pensar durante el act()
        pass

    def act(self, s: np.ndarray) -> int:
        available = [c for c in range(7) if s[0, c] == 0]
        if not available:
            return 0
        if len(available) == 1:
            return available[0]

        # 1. Determinar quién soy (qué color juego) contando las fichas
        pieces = np.count_nonzero(s)
        my_player = 1 if pieces % 2 == 0 else -1

        # 2. Reglas expertas (de la versión de tu amigo, ¡son muy buenas!)
        for col in available:
            board_copy = s.copy()
            row = self._drop(board_copy, col, my_player)
            if row >= 0 and self._check_win(board_copy, row, col, my_player):
                return col

        for col in available:
            board_copy = s.copy()
            row = self._drop(board_copy, col, -my_player)
            if row >= 0 and self._check_win(board_copy, row, col, -my_player):
                return col

        # 3. MCTS al vuelo
        root = MCTSNode(board=s.copy(), current_player=my_player)
        end_time = time.time() + self.time_limit

        while time.time() < end_time:
            # Seleccion
            node = root
            while node.is_fully_expanded() and not node.is_terminal():
                node = max(node.children, key=lambda c: c.ucb1())

            # Expansion
            if not node.is_fully_expanded():
                action = random.choice(node.untried_actions)
                node.untried_actions.remove(action)
                
                next_board = node.board.copy()
                self._drop(next_board, action, node.current_player)
                
                child = MCTSNode(
                    board=next_board, 
                    current_player=-node.current_player, # Cambia el turno
                    parent=node, 
                    action=action
                )
                node.children.append(child)
                node = child

            # Simulacion
            reward = self._simulate(node.board, node.current_player, my_player, node.action)

            # Retropropagacion
            while node is not None:
                node.visits += 1
                node.value += reward
                node = node.parent

        # 4. Decisión final: Elegir el hijo más visitado
        if not root.children:
            return random.choice(available)
            
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action

    def _simulate(self, board: np.ndarray, current_player: int, original_player: int, last_action: int) -> float:
        """Juega aleatoriamente hasta el final y retorna la recompensa desde la perspectiva de original_player."""
        sim_board = board.copy()
        player = current_player
        
        # Primero revisamos si el nodo que acabamos de expandir ya es victoria
        if last_action is not None:
            # Buscar la fila donde cayó la última ficha (para usar el check_win de tu amigo)
            for r in range(6):
                if sim_board[r, last_action] != 0:
                    if self._check_win(sim_board, r, last_action, -player):
                        return 1.0 if -player == original_player else -1.0
                    break

        while True:
            available = [c for c in range(7) if sim_board[0, c] == 0]
            if not available:
                return 0.0 # Empate
                
            action = random.choice(available)
            row = self._drop(sim_board, action, player)
            
            if self._check_win(sim_board, row, action, player):
                return 1.0 if player == original_player else -1.0
                
            player *= -1

    # --- Funciones originales de tu amigo (intactas y eficientes) ---
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