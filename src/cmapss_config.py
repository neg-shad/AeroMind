from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class CMAPSSColumns:
    index:    List[str] = field(default_factory=lambda: ["unit_nr", "time_cycles"])
    settings: List[str] = field(default_factory=lambda: ["setting_1", "setting_2", "setting_3"])
    sensors:  List[str] = field(default_factory=lambda: [f"s_{i}" for i in range(1, 22)])

    @property
    def all(self) -> List[str]:
        return self.index + self.settings + self.sensors

    @property
    def non_feature(self) -> set:
        return {"unit_nr", "time_cycles", "RUL"}


@dataclass
class CMAPSSDatasetConfig:
    train_file: str
    test_file:  str
    rul_file:   str


@dataclass
class PreprocessingConfig:
    clip_rul:      bool = True
    max_rul:       int  = 125
    drop_constant: bool = True
    scale:         bool = True


@dataclass
class CMAPSSConfig:
    data_path: Path = Path("../data/raw")
    columns:   CMAPSSColumns      = field(default_factory=CMAPSSColumns)
    preprocess: PreprocessingConfig = field(default_factory=PreprocessingConfig)

    datasets: dict = field(default_factory=lambda: {
        "FD001": CMAPSSDatasetConfig("train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"),
        "FD002": CMAPSSDatasetConfig("train_FD002.txt", "test_FD002.txt", "RUL_FD002.txt"),
        "FD003": CMAPSSDatasetConfig("train_FD003.txt", "test_FD003.txt", "RUL_FD003.txt"),
        "FD004": CMAPSSDatasetConfig("train_FD004.txt", "test_FD004.txt", "RUL_FD004.txt"),
    })
