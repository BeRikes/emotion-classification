from torch import save

class Early_stop:
    def __init__(self, patience, tolerance, save_path):
        self.cnt = 0
        self.patience = patience
        self.tolerance = tolerance
        self.save_path = save_path
        self.stop = False
        self.best_loss = None
        self.best_epoch = 0

    def __call__(self, test_loss, epoch, model):
        if self.best_loss is None:
            self.best_loss, self.best_epoch = test_loss, epoch
            save(model.state_dict(), self.save_path)
        elif test_loss > self.best_loss + self.tolerance:
            self.cnt += 1
            if self.cnt >= self.patience:
                self.stop = True
        elif test_loss < self.best_loss:
            self.best_loss, self.best_epoch = test_loss, epoch
            save(model.state_dict(), self.save_path)
            self.cnt = 0
        else:
            self.cnt = 0