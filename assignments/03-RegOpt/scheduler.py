from typing import List

from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """
    A custom learning rate scheduler.
    """

    STEPS_PER_EPOCH = 781
    # STEPS_PER_EPOCH = 100

    def __init__(self, optimizer, gamma, last_epoch=-1):
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        """
        self.gamma = gamma
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Return the current learning rate for each parameter group.
        """

        # update only once per epoch, not per batch
        if (self.last_epoch == 0) or (self.last_epoch % self.STEPS_PER_EPOCH != 0):
            return [group["lr"] for group in self.optimizer.param_groups]

        current_epoch = (self.last_epoch / self.STEPS_PER_EPOCH * 1.0) - 1

        vals = [
            group["lr"] * (self.gamma**current_epoch)
            for group in self.optimizer.param_groups
        ]
        return vals
