"""Console + metrics.jsonl (+ optional WandB) logging shared by all trainers."""

import json
import time
from pathlib import Path

import numpy as np


def _to_py(v):
    v = np.asarray(v)
    return v.item() if v.ndim == 0 else v.tolist()


class RunLogger:
    def __init__(self, out_dir, cfg_dict=None, use_wandb=False, project=None, run_name=None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._f = open(self.out_dir / 'metrics.jsonl', 'a')
        self._t0 = time.time()
        self.wandb = None
        if use_wandb:
            from minidropper.utils.wandb_logger import WandbLogger
            self.wandb = WandbLogger(project, run_name, cfg_dict or {})

    def log(self, step, metrics, print_keys=None, prefix=''):
        rec = {'step': int(step), 'wall': time.time() - self._t0}
        rec.update({k: _to_py(v) for k, v in metrics.items()})
        self._f.write(json.dumps(rec) + '\n')
        self._f.flush()
        if self.wandb is not None:
            scalars = {k: v for k, v in rec.items() if isinstance(v, (int, float))}
            self.wandb.log_metrics(int(step), scalars)
        if print_keys:
            parts = [f'{prefix}step {int(step):>11,d}', f't {rec["wall"]:7.1f}s']
            for k in print_keys:
                if k in rec and isinstance(rec[k], (int, float)):
                    name = k[5:] if k.startswith('eval/') else k
                    parts.append(f'{name} {rec[k]:+.3f}')
            print(' | '.join(parts), flush=True)

    def finish(self):
        self._f.close()
        if self.wandb is not None:
            self.wandb.save_training_data(str(self.out_dir))
            self.wandb.finish()
