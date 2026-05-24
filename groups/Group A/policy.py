import numpy as np
from connect4.policy import Policy



class Aha(Policy):

    def __init__(self):
        self.Q: dict[bytes, np.ndarray] = {}
        self.returns: dict[tuple[bytes, int], list[float]] = {}

    def mount(self, timeout: float = 30.0) -> None:
        self._train(n_episodes=8000)

    def _train(self, n_episodes: int) -> None:
        rng = np.random.default_rng(42)
        for _ in range(n_episodes):
            episode = self._generate_episode(rng)
            visited: set[tuple[bytes, int]] = set()
            G = 0.0
            gamma = 0.95
            for t in range(len(episode) - 1, -1, -1):
                state_bytes, action, reward = episode[t]
                G = gamma * G + reward
                sa = (state_bytes, action)
                if sa not in visited:
                    visited.add(sa)
                    if sa not in self.returns:
                        self.returns[sa] = []
                    self.returns[sa].append(G)
                    if state_bytes not in self.Q:
                        self.Q[state_bytes] = np.zeros(7)
                    self.Q[state_bytes][action] = np.mean(self.returns[sa])

    def _generate_episode(self, rng: np.random.Generator) -> list[tuple[bytes, int, float]]:
        board = np.zeros((6, 7), dtype=np.int8)
        episode: list[tuple[bytes, int, float]] = []
        epsilon = 0.2
        current_player = 1

        for _ in range(42):
            available = [c for c in range(7) if board[0, c] == 0]
            if not available:
                break

            state_key = board.tobytes()

            if current_player == 1:
                if rng.random() < epsilon or state_key not in self.Q:
                    action = int(rng.choice(available))
                else:
                    mask = np.full(7, -np.inf)
                    for c in available:
                        mask[c] = self.Q[state_key][c]
                    action = int(np.argmax(mask))
                episode.append((state_key, action, 0.0))
            else:
                action = int(rng.choice(available))

            row = self._drop(board, action, current_player)

            if self._check_win(board, row, action, current_player):
                reward = 1.0 if current_player == 1 else -1.0
                if episode:
                    s, a, _ = episode[-1]
                    episode[-1] = (s, a, reward)
                break

            current_player *= -1

        return episode

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


    def act(self, s: np.ndarray) -> int:
        available = [c for c in range(7) if s[0, c] == 0]
        state_key = s.tobytes()

        for col in available:
            board_copy = s.copy()
            row = self._drop(board_copy, col, 1)
            if row >= 0 and self._check_win(board_copy, row, col, 1):
                return col

        for col in available:
            board_copy = s.copy()
            row = self._drop(board_copy, col, -1)
            if row >= 0 and self._check_win(board_copy, row, col, -1):
                return col

        if state_key in self.Q:
            mask = np.full(7, -np.inf)
            for c in available:
                mask[c] = self.Q[state_key][c]
            return int(np.argmax(mask))

        return min(available, key=lambda c: abs(c - 3))
