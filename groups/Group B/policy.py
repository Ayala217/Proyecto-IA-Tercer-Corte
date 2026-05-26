import numpy as np
from connect4.policy import Policy
from connect4.connect_state import ConnectState


ROWS = 6
COLS = 7


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------

def extract_features(board: np.ndarray, player: int) -> np.ndarray:
    opp = -player

    # bias
    # own1 own2 own3 own4
    # opp1 opp2 opp3 opp4
    # center
    f = np.zeros(10, dtype=float)

    f[0] = 1.0

    def score_window(w):
        p = int(np.sum(w == player))
        o = int(np.sum(w == opp))

        if p > 0 and o > 0:
            return

        if p == 1 and o == 0:
            f[1] += 1
        elif p == 2 and o == 0:
            f[2] += 1
        elif p == 3 and o == 0:
            f[3] += 1
        elif p == 4:
            f[4] += 1

        if o == 1 and p == 0:
            f[5] += 1
        elif o == 2 and p == 0:
            f[6] += 1
        elif o == 3 and p == 0:
            f[7] += 1
        elif o == 4:
            f[8] += 1

    # Horizontal
    for r in range(ROWS):
        for c in range(COLS - 3):
            score_window(board[r, c:c + 4])

    # Vertical
    for c in range(COLS):
        for r in range(ROWS - 3):
            score_window(board[r:r + 4, c])

    # Diagonal ↘
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            w = np.array([board[r + i, c + i] for i in range(4)])
            score_window(w)

    # Diagonal ↙
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            w = np.array([board[r - i, c + i] for i in range(4)])
            score_window(w)

    # Centro
    f[9] = np.sum(board[:, COLS // 2] == player)

    return f


# ---------------------------------------------------------------------------
# Linear approximation
# ---------------------------------------------------------------------------

def approx_value(board: np.ndarray, player: int, theta: np.ndarray) -> float:
    return float(theta @ extract_features(board, player))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def drop_piece(board: np.ndarray, col: int, player: int) -> np.ndarray:
    nb = board.copy()

    for r in reversed(range(ROWS)):
        if nb[r, col] == 0:
            nb[r, col] = player
            return nb

    raise ValueError(f"Column {col} is full")


def check_win(board: np.ndarray, player: int) -> bool:

    # Horizontal
    for r in range(ROWS):
        for c in range(COLS - 3):
            if all(board[r, c + i] == player for i in range(4)):
                return True

    # Vertical
    for c in range(COLS):
        for r in range(ROWS - 3):
            if all(board[r + i, c] == player for i in range(4)):
                return True

    # Diagonal ↘
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if all(board[r + i, c + i] == player for i in range(4)):
                return True

    # Diagonal ↙
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            if all(board[r - i, c + i] == player for i in range(4)):
                return True

    return False


def legal_moves(board: np.ndarray):
    return [c for c in range(COLS) if board[0, c] == 0]


# ---------------------------------------------------------------------------
# TD(0) training
# ---------------------------------------------------------------------------

def train_adp(
    n_episodes: int = 12000,
    alpha_start: float = 0.01,
    alpha_min: float = 0.001,
    alpha_decay: float = 0.9997,
    gamma: float = 0.95,
    epsilon_start: float = 0.15,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.9995,
    seed: int = 42,
) -> np.ndarray:

    rng = np.random.default_rng(seed)

    theta = np.zeros(10)

    R_WIN = 1.0
    R_LOSE = -1.0
    R_DRAW = 0.0

    for episode in range(n_episodes):

        epsilon = max(
            epsilon_min,
            epsilon_start * (epsilon_decay ** episode)
        )

        alpha = max(
            alpha_min,
            alpha_start * (alpha_decay ** episode)
        )

        state = ConnectState()

        agent_player = -1 if episode % 2 == 0 else 1
        opp_player = -agent_player

        while not state.is_final():

            # Opponent turn
            if state.player != agent_player:

                legal = state.get_free_cols()

                opp_col = int(rng.choice(legal))

                state = state.transition(opp_col)

                continue

            board = state.board.copy()

            phi_s = extract_features(board, agent_player)

            v_s = float(theta @ phi_s)

            legal = state.get_free_cols()

            # ε-greedy
            if rng.random() < epsilon:

                col = int(rng.choice(legal))

            else:

                best_col = legal[0]
                best_val = -np.inf

                for c in legal:

                    next_state = state.transition(c)

                    winner = next_state.get_winner()

                    if winner == agent_player:
                        best_col = c
                        break

                    val = approx_value(
                        next_state.board,
                        agent_player,
                        theta
                    )

                    if val > best_val:
                        best_val = val
                        best_col = c

                col = best_col

            next_state = state.transition(col)

            winner = next_state.get_winner()

            if winner == agent_player:
                r = R_WIN

            elif winner == opp_player:
                r = R_LOSE

            elif next_state.is_final():
                r = R_DRAW

            else:
                r = 0.0

            # TD(0)
            if next_state.is_final():

                v_next = 0.0

            else:

                v_next = approx_value(
                    next_state.board,
                    agent_player,
                    theta
                )

            td_error = r + gamma * v_next - v_s

            theta += alpha * td_error * phi_s

            state = next_state

    return theta


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class MyPolicy(Policy):

    trained_theta = None

    def __init__(self):
        self.theta = np.zeros(10)

  
    def mount(self, *args, **kwargs) -> None:

        if MyPolicy.trained_theta is None:

            MyPolicy.trained_theta = train_adp(
                n_episodes=12000,
                alpha_start=0.01,
                alpha_min=0.001,
                alpha_decay=0.9997,
                gamma=0.95,
                epsilon_start=0.15,
                epsilon_min=0.01,
                epsilon_decay=0.9995,
                seed=42
            )

        self.theta = MyPolicy.trained_theta

    
    def act(self, s: np.ndarray) -> int:

        red_count = int(np.sum(s == -1))
        yellow_count = int(np.sum(s == 1))

        player = -1 if red_count == yellow_count else 1
        opp = -player

        legal = legal_moves(s)

        if not legal:
            raise RuntimeError("No legal columns")

        # -------------------------------------------------------------------
        # 1. Immediate win
        # -------------------------------------------------------------------

        for col in legal:

            ns = drop_piece(s, col, player)

            if check_win(ns, player):
                return col

        # -------------------------------------------------------------------
        # 2. Block opponent immediate win
        # -------------------------------------------------------------------

        for col in legal:

            ns = drop_piece(s, col, opp)

            if check_win(ns, opp):
                return col

        # -------------------------------------------------------------------
        # 3. Avoid suicidal moves
        # -------------------------------------------------------------------

        safe_moves = []

        for col in legal:

            ns = drop_piece(s, col, player)

            opponent_can_win = False

            for oc in legal_moves(ns):

                test = drop_piece(ns, oc, opp)

                if check_win(test, opp):
                    opponent_can_win = True
                    break

            if not opponent_can_win:
                safe_moves.append(col)

        if safe_moves:
            candidate_moves = safe_moves
        else:
            candidate_moves = legal

        # -------------------------------------------------------------------
        # 4. Prefer center
        # -------------------------------------------------------------------

        center = COLS // 2

        center_sorted = sorted(
            candidate_moves,
            key=lambda c: abs(c - center)
        )

        # -------------------------------------------------------------------
        # 5. ADP evaluation
        # -------------------------------------------------------------------

        best_col = center_sorted[0]
        best_val = -np.inf

        for col in center_sorted:

            ns = drop_piece(s, col, player)

            val = approx_value(
                ns,
                player,
                self.theta
            )

            # Extra center bonus
            val += (3 - abs(col - center)) * 0.25

            if val > best_val:
                best_val = val
                best_col = col

        return best_col