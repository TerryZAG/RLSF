# early stop for time domain learning

import numpy as np

class EarlyStopping:

    def __init__(self, patience=5, delta=0.5):

        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.finetune_loss_min = np.Inf
        self.finetune_index = np.int32
        self.delta = delta

    def __call__(self, fine_tune_loss, index, epoch):

        score = fine_tune_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(fine_tune_loss, index)
        elif score > self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        elif score < self.best_score:
            self.best_score = score
            self.save_checkpoint(fine_tune_loss, index)
            self.counter = 0
        else:
            self.counter = 0

    def save_checkpoint(self, fine_tune_loss, index):
        self.finetune_loss_min = fine_tune_loss
        self.finetune_index = index