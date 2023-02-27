# early stop for DNSI agent

import numpy as np
import torch

class EarlyStopping:
    def __init__(self, save_path, patience=10, verbose=False, delta=0.2):

        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.reward_max = -np.Inf
        self.delta = delta

    def __call__(self, reward, i_episode, model, optimizer_state_dict):

        score = reward

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(reward, i_episode, model, optimizer_state_dict)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            print("best score={}, delta={}, score={}".format(self.best_score, self.delta, score))
            self.best_score = score
            self.save_checkpoint(reward, i_episode, model, optimizer_state_dict)
            self.counter = 0

    def save_checkpoint(self, reward, i_episode, model, optimizer_state_dict):
        # '''Saves model when reward max.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.reward_max:.6f} --> {reward:.6f}).  Saving model ...')
        state = {
            'episode': i_episode,
            'state_dict': model.state_dict(),
            'optimizer': optimizer_state_dict
        }
        path = str(self.save_path / 'model_best_dqn_earlystop.pth')
        torch.save(state, path)
        print(".....The model obtained by early stop has been saved in {}.....".format(path))
        self.reward_max = reward