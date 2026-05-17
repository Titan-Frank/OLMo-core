import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

from olmo_core.distributed.utils import get_rank

from .callback import Callback

log = logging.getLogger(__name__)


@dataclass
class TensorBoardCallback(Callback):
    """
    Logs metrics to TensorBoard from rank 0.

    Start the TensorBoard server with:
        tensorboard --logdir=<save_folder>/tensorboard
    """

    enabled: bool = True
    log_dir: Optional[str] = None

    _writer = None

    @property
    def writer(self):
        if self._writer is None:
            from torch.utils.tensorboard import SummaryWriter

            log_dir = self.log_dir or str(self.trainer.work_dir / "tensorboard")
            self._writer = SummaryWriter(log_dir=log_dir)
            log.info(f"TensorBoard logging to {log_dir}")
        return self._writer

    def log_metrics(self, step: int, metrics: Dict[str, float]):
        if self.enabled and get_rank() == 0:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, global_step=step)

    def close(self):
        if self.enabled and get_rank() == 0 and self._writer is not None:
            self.writer.flush()
            self.writer.close()
