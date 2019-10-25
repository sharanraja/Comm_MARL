import numpy as np


class ReplayBuffer(object):
    """
        Simple replay buffer
        Ref: https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    """
    def __init__(self):
        self.storage = []

    def __len__(self):
        return len(self.storage)

    def clear(self):
        self.storage.clear()
        assert len(self.storage) == 0

    def sync(self, memory):
        self.clear()
        for exp in memory.storage:
            self.storage.append(exp)

        assert len(memory) == len(self.storage)

    def add(self, data):
        # Expects tuples of (state, next_state, action, reward, done)
        if len(self.storage) > 1e5:
            self.storage.pop(0)
        self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind: 
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)

    def centralized_sample(self, batch_size=100, n_agent=None):
        # NOTE Order is agent 0 and 1
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x_n, y_n, u_n, r_n, d_n = [], [], [], [], []

        for i_agent in range(n_agent):
            x, y, u, r, d = [], [], [], [], []

            for i in ind: 
                X, Y, U, R, D = self.storage[i]
                assert len(X) == n_agent

                x.append(np.array(X[i_agent], copy=False))
                y.append(np.array(Y[i_agent], copy=False))
                u.append(np.array(U[i_agent], copy=False))
                r.append(np.array(R[i_agent], copy=False))
                d.append(np.array(D[i_agent], copy=False))

            assert len(x) == batch_size
            x, y, u, r, d = np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)

            x_n.append(x)
            y_n.append(y)
            u_n.append(u)
            r_n.append(r)
            d_n.append(d)

        assert len(x_n) == n_agent
        return x_n, y_n, u_n, r_n, d_n
