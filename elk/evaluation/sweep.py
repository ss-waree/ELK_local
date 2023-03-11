from dataclasses import dataclass
from typing import Sequence
from elk.training.train import train, RunConfig
from simple_parsing.helpers import Serializable


@dataclass
class SweepConfig(Serializable):
    runs: Sequence[RunConfig]

def sweep(cfg: SweepConfig):
    for run in cfg.runs:
        train(run, None)
