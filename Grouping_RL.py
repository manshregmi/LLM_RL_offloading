"""
Grouping RL — Tabular A2C
==========================

Chooses the total number of groups K (3 <= K <= 399) based on state
(bandwidth, mean_contention). The actual grouping of the 400 layers into K
segments is handled downstream. After grouping, another RL agent decides the
offloading; when that downstream RL finishes, it pushes a latency-based reward
back to this agent, which then updates its policy/value tables.

Design:
  - Tabular A2C (policy table + state-value table) — state space is small
    enough that this is cleaner than neural networks.
  - Discount factor gamma = 0.95 (as specified).
  - State: (bandwidth_bin, contention_bin) — 10 x 10 = 100 discrete states.
  - Action: K in [3, 399] — 397 discrete actions.
  - Async reward flow: downstream RL pushes latency reward onto a shared
    asyncio.Queue; this agent awaits it, updates tables, returns True on
    successful update.
"""

import asyncio
import numpy as np


# =========================================================================
# HYPERPARAMETERS
# =========================================================================

# --- Discount factor ---
GAMMA = 0.95                        # as specified by user

# --- Learning rates ---
ALPHA_POLICY = 0.05                 # policy table learning rate
ALPHA_VALUE = 0.1                   # value table learning rate (higher: critic
                                    # should track value faster than actor changes)

# --- Action space ---
K_MIN = 3                           # minimum groups: LLaMA1, LLaMA2, BART
K_MAX = 399                         # maximum groups (just under 400 layers)
NUM_ACTIONS = K_MAX - K_MIN + 1     # 397 possible K values

# --- State discretization ---
# 10 bins per dimension -> 100 states. Coarse enough for fast convergence,
# fine enough to capture meaningful (bw, contention) variation.
NUM_BW_BINS = 14
NUM_CONT_BINS = 19
NUM_STATES = NUM_BW_BINS * NUM_CONT_BINS

# Bin edges — adjust these to match your simulator's real BW/contention ranges.
BW_BIN_EDGES = np.linspace(9, 35.0, NUM_BW_BINS + 1)      # Mbps
CONT_BIN_EDGES = np.linspace(0, 20.0, NUM_CONT_BINS + 1)  # ms

# --- Exploration ---
# Entropy-style exploration: softmax temperature over policy logits.
# Higher temperature -> more random early; anneals down over training.
TEMP_START = 2.0
TEMP_END = 0.3
TEMP_DECAY_STEPS = 32000

# Epsilon-greedy floor: probability of picking a uniformly random action,
# independent of the softmax policy. Provides an exploration safety net so the
# policy can't get permanently locked into a local optimum.
# Currently 0 (disabled) to match the offloading RL's default. Set to e.g. 0.05
# if you want a guaranteed minimum exploration rate.
EPSILON_MIN = 0.0


# =========================================================================
# GROUPING RL
# =========================================================================
class GroupingRL:
    """
    Tabular A2C agent that picks K (number of groups) given the current
    (bandwidth, mean_contention). Receives latency-based reward asynchronously
    from a downstream offloading RL.
    """

    def __init__(self,
                 num_bw_bins=NUM_BW_BINS,
                 num_cont_bins=NUM_CONT_BINS,
                 num_actions=NUM_ACTIONS,
                 gamma=GAMMA,
                 alpha_policy=ALPHA_POLICY,
                 alpha_value=ALPHA_VALUE,
                 epsilon_min=EPSILON_MIN,
                 total_pipelines=1
                 ):

        self.num_bw_bins = num_bw_bins
        self.num_cont_bins = num_cont_bins
        self.num_states = num_bw_bins * num_cont_bins

        self.K_MIN = 3 * total_pipelines
        self.K_MAX = 399 * total_pipelines
        self.num_actions = self.K_MAX - self.K_MIN + 1

        self.gamma = gamma
        self.alpha_policy = alpha_policy
        self.alpha_value = alpha_value
        self.epsilon_min = epsilon_min

        # --- Policy table: dict mapping (state_key, action_key) -> preference score ---
        # Matches the offloading RL's dict-based table style.
        # Missing entries default to 0.0 (uniform policy at start).
        self.policy_table = {}

        # --- Value table: dict mapping state_key -> V(s) ---
        self.value_table = {}

        # --- Async reward queue: downstream RL pushes (reward, context) here ---
        # Context carries the info we need to run the A2C update when the reward
        # arrives (state index, chosen action, next state).
        # Initialize lazily to avoid event loop issues
        self.reward_queue = None

        # --- Exploration schedule ---
        self.step_count = 0

        # --- Bookkeeping for the last action chosen (for sync use) ---
        self.last_state_key = None
        self.last_action_key = None

    def _ensure_reward_queue(self):
        """
        Lazily create the async queue when we have an event loop.
        This prevents the RuntimeError when no event loop exists during __init__.
        """
        if self.reward_queue is None:
            try:
                self.reward_queue = asyncio.Queue()
            except RuntimeError:
                # No event loop running, create one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self.reward_queue = asyncio.Queue()
        return self.reward_queue

    # =====================================================================
    # STATE / ACTION KEY CONVERSIONS
    # =====================================================================
    def state_to_key(self, bandwidth: float, mean_contention: float) -> any:
        """
        Map continuous state (bandwidth, mean_contention) to an integer key
        that indexes into policy_table and value_table.

        Layout: bw_bin * num_cont_bins + ct_bin  (row-major flattening of the
        2D (bw_bin, ct_bin) grid into a single integer in [0, num_states)).
        """
        bw_bin = int(np.clip(np.digitize(bandwidth, BW_BIN_EDGES) - 1,
                             0, self.num_bw_bins - 1))
        ct_bin = int(np.clip(np.digitize(mean_contention, CONT_BIN_EDGES) - 1,
                             0, self.num_cont_bins - 1))
        
        return (bw_bin, ct_bin)

    def action_to_key(self, K: int) -> int:
        """
        Map a K value (number of groups, in [K_MIN, K_MAX]) to a column index
        into policy_table (in [0, num_actions)).
        """
        return int(K) - self.K_MIN

    def key_to_action(self, action_key: int) -> int:
        """Inverse of action_to_key: column index -> K value."""
        return int(action_key) + self.K_MIN

    # =====================================================================
    # POLICY & TEMPERATURE
    # =====================================================================
    def _current_temperature(self) -> float:
        """Linear temperature decay from TEMP_START -> TEMP_END."""
        frac = min(1.0, self.step_count / TEMP_DECAY_STEPS)
        return TEMP_START + frac * (TEMP_END - TEMP_START)

    def _policy_probs(self, state_key) -> np.ndarray:
        """Softmax over policy logits at state_key, with current temperature."""
        T = self._current_temperature()
        # Build logits array from dict (missing entries default to 0.0)
        logits = np.array([
            self.policy_table.get((state_key, a), 0.0)
            for a in range(self.num_actions)
        ])
        logits = logits / T
        logits -= logits.max()                      # numerical stability
        exps = np.exp(logits)
        return exps / exps.sum()

    # =====================================================================
    # ACTION SELECTION
    # =====================================================================
    def choose_action(self, bandwidth: float, mean_contention: float) -> int:
        """
        Sample K from the current policy given (bandwidth, mean_contention).

        Exploration is a hybrid of:
          - epsilon-greedy: with probability `epsilon_min`, pick a uniformly
            random K (forced exploration floor).
          - softmax over policy logits with annealed temperature (otherwise).

        This is called at the START OF THE PIPELINE (before grouping + offloading).

        Returns:
            K (int): chosen number of groups, in [K_MIN, K_MAX].
        """
        state_key = self.state_to_key(bandwidth, mean_contention)

        # Epsilon-greedy forced exploration
        if self.epsilon_min > 0.0 and np.random.random() < self.epsilon_min:
            action_key = int(np.random.randint(self.num_actions))
        else:
            probs = self._policy_probs(state_key)
            action_key = int(np.random.choice(self.num_actions, p=probs))

        K = self.key_to_action(action_key)

        # Remember for the upcoming update
        self.last_state_key = state_key
        self.last_action_key = action_key

        return K

    # =====================================================================
    # A2C UPDATE (the core table update logic)
    # =====================================================================
    def _update_tables(self, state_key, action_key: int,
                       reward: float, next_state_key=None,
                       done: bool = False) -> dict:
        """
        One-step A2C update:
            td_target  = r + gamma * V(s') * (1 - done)
            td_error   = td_target - V(s)
            V(s)      <- V(s) + alpha_value * td_error
            logit(s,a)<- logit(s,a) + alpha_policy * td_error

        The policy update uses td_error as the advantage — the classic actor-critic
        rule. The value update is standard TD(0).
        """
        V_s = self.value_table.get(state_key, 0.0)
        V_next = self.value_table.get(next_state_key, 0.0) if next_state_key is not None and not done else 0.0

        td_target = reward + self.gamma * V_next
        td_error = td_target - V_s

        # --- Critic update ---
        self.value_table[state_key] = V_s + self.alpha_value * td_error

        # --- Actor update ---
        old_pref = self.policy_table.get((state_key, action_key), 0.0)
        self.policy_table[(state_key, action_key)] = old_pref + self.alpha_policy * td_error

        self.step_count += 1

        return {
            "td_error": float(td_error),
            "td_target": float(td_target),
            "V_s_before": float(V_s),
            "V_s_after": float(self.value_table[state_key]),
        }

    # =====================================================================
    # TRAIN ENTRY POINT (sync)
    # =====================================================================
    def train(self, bandwidth: float, mean_contention: float) -> any:
        """
        Training entry point. Given current (bw, contention), pick an action
        (K) from the current policy and return everything the caller needs to
        later push a reward via `push_reward()`.

        Returns:
            K (int):          chosen number of groups.
            state_key (int):  discretized state index (for the reward payload).
            action_key (int): action index = K - K_MIN (for the reward payload).
        """
        K = self.choose_action(bandwidth, mean_contention)
        return K

    # =====================================================================
    # ASYNC REWARD RECEIPT
    # =====================================================================
    async def get_reward(self, reward) -> bool:
        """
        Await the latency-based reward from the downstream offloading RL,
        then update the policy and value tables.

        The downstream RL is expected to push a dict onto self.reward_queue:
            {
                "reward":          <float>,  # latency-based reward
                "state_key":       <int>,    # state when K was chosen
                "action_key":      <int>,    # K - K_MIN
                "next_state_key":  <int>,    # state after the episode (bw/contention
                                             #   after the grouped pipeline ran)
                "done":            <bool>,   # terminal flag
            }

        Returns:
            True  if the update succeeded.
            False if the queue entry was malformed.
        """
        self._ensure_reward_queue()
        
        try:
            reward = reward
            state_key = self.last_state_key       # tuple (bw_bin, ct_bin)
            action_key = self.last_action_key
            done = False
            next_state_key = None
        except (KeyError, TypeError, ValueError) as e:
            print(f"[GroupingRL] Malformed reward payload: {e}")
            return False

        # Update tables
        self._update_tables(state_key, action_key, reward, next_state_key, done)
        return True

    # =====================================================================
    # HELPER: let the downstream RL push a reward
    # =====================================================================
    async def push_reward(self, reward: float, done: bool = False):
        """
        Convenience method the downstream offloading RL can call to deliver
        the latency-based reward asynchronously.

        In a real system you might just hand the queue reference directly to
        the downstream RL; this method is a thin wrapper for clarity.
        """
        self._ensure_reward_queue()
        await self.reward_queue.put({
            "reward": reward,
            "state_key": self.last_state_key,
            "action_key": self.last_action_key,
            "done": done,
        })

    async def get_reward_from_queue(self):
        """
        Alternative method that actually waits for a reward from the queue.
        This is useful if you want to block until a reward is available.
        """
        self._ensure_reward_queue()
        try:
            payload = await self.reward_queue.get()
            state_key = payload.get("state_key", self.last_state_key)
            action_key = payload.get("action_key", self.last_action_key)
            reward = payload.get("reward", 0.0)
            next_state_key = payload.get("next_state_key", None)
            done = payload.get("done", False)
            
            self._update_tables(state_key, action_key, reward, next_state_key, done)
            self.reward_queue.task_done()
            return True
        except asyncio.CancelledError:
            return False
        except (KeyError, TypeError, ValueError) as e:
            print(f"[GroupingRL] Malformed reward payload: {e}")
            return False

    # =====================================================================
    # UTILITIES
    # =====================================================================
    def greedy_action(self, bandwidth: float, mean_contention: float) -> int:
        """Deterministic: pick the action with the highest logit (for inference)."""
        state_key = self.state_to_key(bandwidth, mean_contention)
        prefs = np.array([
            self.policy_table.get((state_key, a), 0.0)
            for a in range(self.num_actions)
        ])
        action_key = int(np.argmax(prefs))
        return self.key_to_action(action_key)

    def state_summary(self, bandwidth: float, mean_contention: float) -> dict:
        """Diagnostics for a given state."""
        state_key = self.state_to_key(bandwidth, mean_contention)
        probs = self._policy_probs(state_key)
        top_k = np.argsort(probs)[-5:][::-1]
        return {
            "state_key": state_key,
            "V(s)": float(self.value_table.get(state_key, 0.0)),
            "temperature": self._current_temperature(),
            "top5_actions_K": [int(i + self.K_MIN) for i in top_k],
            "top5_probs": [float(probs[i]) for i in top_k],
        }

    # =====================================================================
    # PERSISTENCE
    # =====================================================================
    def save(self, path: str = "grouping_rl_tables.pkl") -> None:
        """Save learned tables using pickle (matches offloading RL convention)."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump((self.policy_table, self.value_table, self.step_count), f)
        print(f"[GroupingRL] Saved to {path}")

    def load(self, path: str = "grouping_rl_tables.pkl") -> bool:
        """Load learned tables from a pickle file."""
        import pickle, os
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.policy_table, self.value_table, self.step_count = pickle.load(f)
            print(f"[GroupingRL] Loaded from {path} (step_count={self.step_count})")
            return True
        else:
            print(f"[GroupingRL] No checkpoint at {path} — starting fresh.")
            return False
