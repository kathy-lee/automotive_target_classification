from tensorflow.keras.callbacks import Callback
from tensorflow.keras.backend import get_value, set_value
import numpy as np

class MultiStepLR(Callback):
  """Learning rate scheduler.

  Arguments:
      schedule: a function that takes an epoch index as input
          (integer, indexed from 0) and returns a new
          learning rate as output (float).
      verbose: int. 0: quiet, 1: update messages.

  ```python
  # This function keeps the learning rate at 0.001 for the first ten epochs
  # and decreases it exponentially after that.
  def scheduler(epoch):
    if epoch < 10:
      return 0.001
    else:
      return 0.001 * tf.math.exp(0.1 * (10 - epoch))

  callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
  model.fit(data, labels, epochs=100, callbacks=[callback],
            validation_data=(val_data, val_labels))
  ```
  """

  def __init__(self, base_lr=3e-4, step_size=10, decay_rate=10, verbose=0):
    super(Callback, self).__init__()
    self.base_lr = base_lr
    self.decay_rate = decay_rate
    self.step_size = step_size
    self.verbose = verbose
    self.history = {}

  def on_epoch_begin(self, epoch, logs=None):
    if not hasattr(self.model.optimizer, 'lr'):
      raise ValueError('Optimizer must have a "lr" attribute.')
    try:  # new API
      lr = float(get_value(self.model.optimizer.lr))
      #lr = self.schedule(epoch, lr)
      lr = self.base_lr / pow(self.decay_rate, epoch//self.step_size)
    except TypeError:  # Support for old API for backward compatibility
      #lr = self.schedule(epoch)
      lr = self.base_lr / pow(self.decay_rate, epoch // self.step_size)
    if not isinstance(lr, (float, np.float32, np.float64)):
      raise ValueError('The output of the "schedule" function '
                       'should be float.')
    set_value(self.model.optimizer.lr, lr)
    if self.verbose > 0:
      print('\nEpoch %05d: LearningRateScheduler reducing learning '
            'rate to %s.' % (epoch + 1, lr))

  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}
    logs['lr'] = get_value(self.model.optimizer.lr)
    self.history.setdefault('lr', []).append(get_value(self.model.optimizer.lr))