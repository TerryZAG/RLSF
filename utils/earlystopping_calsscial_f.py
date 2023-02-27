# early stop for frequency domain learning
import numpy as np

class EarlyStopping:
    def __init__(self, patience=5, delta=0.5, q_shape=[0, 0]):

        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.reward_max = -np.Inf
        self.delta = delta
        self.Q_table = np.zeros(q_shape)

    def __call__(self, reward, q):

        score = reward

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(reward, q)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(reward, q)
            self.counter = 0

    def save_checkpoint(self, reward, q):
        self.reward_max = reward
        self.Q_table = q