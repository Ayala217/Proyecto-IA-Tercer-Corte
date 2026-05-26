import numpy as np
from connect4.policy import Policy


class Aha(Policy):

    def __init__(self, n_episodes=40000, gamma=0.97, epsilon_start=0.3, epsilon_end=0.05, selfplay_ratio=0.6):
        self.n_episodes = n_episodes
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.selfplay_ratio = selfplay_ratio
        self.Q: dict[bytes, np.ndarray] = {}
        self.returns: dict[tuple[bytes, int], list[float]] = {}

    def mount(self, timeout: float = 30.0) -> None:
        self._train()

    def _train(self) -> None:
        rng = np.random.default_rng(42)
        n_self = int(self.n_episodes * self.selfplay_ratio)
        n_rand = self.n_episodes - n_self

        for i in range(n_rand):
            eps = self._epsilon(i, n_rand)
            episode = self._generate_episode(rng, opponent="random", epsilon=eps)
            self._update_q(episode)

        for i in range(n_self):
            eps = self._epsilon(i, n_self)
            episode = self._generate_episode(rng, opponent="self", epsilon=eps)
            self._update_q(episode)

    def _epsilon(self, step: int, total: int) -> float:
        ratio = step / max(total - 1, 1)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * ratio

    def _update_q(self, episode: list) -> None:
        visited: set[tuple[bytes, int]] = set()
        G = 0.0
        for t in range(len(episode) - 1, -1, -1):
            state_bytes, action, reward = episode[t]
            G = self.gamma * G + reward
            sa = (state_bytes, action)
            if sa not in visited:
                visited.add(sa)
                if sa not in self.returns:
                    self.returns[sa] = []
                self.returns[sa].append(G)
                if state_bytes not in self.Q:
                    self.Q[state_bytes] = np.zeros(7)
                self.Q[state_bytes][action] = np.mean(self.returns[sa])

        for state_bytes, action, _ in episode:
            flipped_key = self._flip_key(state_bytes)
            flipped_action = 6 - action
            sa_f = (flipped_key, flipped_action)
            G_f = self.returns.get((state_bytes, action), [0.0])[-1]
            if flipped_key not in self.Q:
                self.Q[flipped_key] = np.zeros(7)
            if sa_f not in self.returns:
                self.returns[sa_f] = []
            self.returns[sa_f].append(G_f)
            self.Q[flipped_key][flipped_action] = np.mean(self.returns[sa_f])

    def _flip_key(self, state_bytes: bytes) -> bytes:
        board = np.frombuffer(state_bytes, dtype=np.int8).reshape(6, 7)
        return np.fliplr(board).tobytes()

    def _generate_episode(self, rng: np.random.Generator, opponent: str, epsilon: float) -> list:
        board = np.zeros((6, 7), dtype=np.int8)
        episode = []
        current_player = 1

        for _ in range(42):
            available = [c for c in range(7) if board[0, c] == 0]
            if not available:
                break

            state_key = board.tobytes()

            if current_player == 1:
                action = self._epsilon_greedy(state_key, available, rng, epsilon)
                episode.append((state_key, action, 0.0))
            else:
                if opponent == "self" and state_key in self.Q:
                    opp_board = (board * -1).tobytes()
                    action = self._epsilon_greedy(opp_board, available, rng, epsilon * 1.5)
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

    def _epsilon_greedy(self, state_key: bytes, available: list, rng: np.random.Generator, epsilon: float) -> int:
        if rng.random() < epsilon or state_key not in self.Q:
            return int(rng.choice(available))
        mask = np.full(7, -np.inf)
        for c in available:
            mask[c] = self.Q[state_key][c]
        return int(np.argmax(mask))

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
