import numpy as np
import math

class UCB2:
    def __init__(self, allocations, eval_func, alpha=0.5, max_steps=1000):
        """
        allocations: list of feasible allocations (each is a tuple/list of raw numbers)
        eval_func: function(allocation) -> float (returns value for allocation)
        alpha: exploration parameter (0 < alpha <= 1)
        max_steps: number of total pulls
        """
        self.allocations = allocations
        self.eval_func = eval_func
        self.alpha = alpha
        self.max_steps = max_steps
        self.n_arms = len(allocations)
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.values = np.zeros(self.n_arms, dtype=float)
        self.r = np.zeros(self.n_arms, dtype=int)  # phase for each arm
        self.total_counts = 0

    def tau(self, r):
        return int(math.ceil((1 + self.alpha) ** r))

    def ucb2_bound(self, mean, n, total_n, r):
        if n == 0:
            return float('inf')
        bonus = math.sqrt((1 + self.alpha) * math.log(math.e * total_n / self.tau(r)) / (2 * self.tau(r)))
        return mean + bonus

    def select_arm(self):
        # Select arm with minimal phase not fully played
        for i in range(self.n_arms):
            if self.counts[i] < self.tau(self.r[i] + 1):
                return i
        # Otherwise, select arm with highest UCB2 bound
        total_n = max(1, self.total_counts)
        ucbs = [self.ucb2_bound(self.values[i], self.counts[i], total_n, self.r[i]) for i in range(self.n_arms)]
        return int(np.argmax(ucbs))

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.total_counts += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n
        # If finished phase, increment phase
        if self.counts[arm] == self.tau(self.r[arm] + 1):
            self.r[arm] += 1

    def run(self):
        for step in range(self.max_steps):
            arm = self.select_arm()
            allocation = self.allocations[arm]
            reward = self.eval_func(allocation)
            self.update(arm, reward)
        best_idx = int(np.argmax(self.values))
        return self.allocations[best_idx], self.values[best_idx]
