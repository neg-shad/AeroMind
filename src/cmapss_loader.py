import pandas as pd
from typing import List, Optional, Tuple
from sklearn.preprocessing import StandardScaler

from cmapss_config import CMAPSSConfig


class CMAPSDataLoader:
    """
    NASA C-MAPSS Dataset Loader & Preprocessing Pipeline.
    All behaviour is driven by CMAPSSConfig.
    """

    def __init__(self, config: CMAPSSConfig):
        self.cfg = config
        self.scaler: Optional[StandardScaler] = None
        self.feature_cols_: Optional[List[str]] = None

    # ------------------------------------------------------------------ #
    #  Public pipeline entry-point#
    # ------------------------------------------------------------------ #

    def load_dataset(self, dataset_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Full pipeline: load → RUL → drop constants → clip → scale.
        Returns (train_df, test_df) ready for modelling.

        Args:
            dataset_id: one of "FD001" … "FD004"
        """
        ds = self.cfg.datasets[dataset_id]
        p  = self.cfg.preprocess

        train = self._load_raw(ds.train_file)
        test  = self._load_raw(ds.test_file)

        train = self._add_rul_train(train)
        test  = self._add_rul_test(test, ds.rul_file)

        if p.drop_constant:
            train = self._drop_constant(train)
            test  = test[train.columns]           # keep test aligned

        if p.clip_rul:
            train = self._clip_rul(train, p.max_rul)
            test  = self._clip_rul(test,  p.max_rul)

        if p.scale:
            feature_cols = self._feature_cols(train)
            train, test  = self._fit_transform(train, test, feature_cols)

        print(f"✅ Dataset {dataset_id} loaded — train {train.shape}, test {test.shape}")
        return train, test

    def split_by_engine(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """Partition DataFrame into per-engine trajectories."""
        return [
            g.sort_values("time_cycles").reset_index(drop=True)
            for _, g in df.groupby("unit_nr")
        ]

    # ------------------------------------------------------------------ #
    #  Private steps                                                       #
    # ------------------------------------------------------------------ #

    def _load_raw(self, filename: str) -> pd.DataFrame:
        path = self.cfg.data_path / filename
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path.resolve()}")

        col_names = self.cfg.columns.all
        df = pd.read_csv(path, sep=r"\s+", header=None, names=col_names)
        df = df.dropna(axis=1, how="all")

        if df.shape[1] != len(col_names):
            raise ValueError(
                f"Schema mismatch in '{filename}': "
                f"expected {len(col_names)} cols, got {df.shape[1]}."
            )
        return df.sort_values(["unit_nr", "time_cycles"]).reset_index(drop=True)

    def _add_rul_train(self, df: pd.DataFrame) -> pd.DataFrame:
        max_cycle = df.groupby("unit_nr")["time_cycles"].max().rename("max_cycle")
        df = df.join(max_cycle, on="unit_nr")
        df["RUL"] = df["max_cycle"] - df["time_cycles"]
        return df.drop(columns="max_cycle").reset_index(drop=True)

    def _add_rul_test(self, df: pd.DataFrame, rul_file: str) -> pd.DataFrame:
        rul_path = self.cfg.data_path / rul_file
        if not rul_path.exists():
            raise FileNotFoundError(f"RUL file not found: {rul_path.resolve()}")

        truth = pd.read_csv(rul_path, header=None, names=["true_rul"])
        truth["unit_nr"] = truth.index + 1

        max_cycle = df.groupby("unit_nr")["time_cycles"].max().rename("max_cycle_in_test")
        df = df.join(max_cycle, on="unit_nr").merge(truth, on="unit_nr", how="left")
        df["RUL"] = df["max_cycle_in_test"] + df["true_rul"] - df["time_cycles"]
        return df.drop(columns=["max_cycle_in_test", "true_rul"]).reset_index(drop=True)

    def _drop_constant(self, df: pd.DataFrame) -> pd.DataFrame:
        const_cols = [c for c in df.columns if df[c].nunique() <= 1]
        if const_cols:
            print(f"🔍 Dropping constant columns: {const_cols}")
            df = df.drop(columns=const_cols)
        return df

    def _clip_rul(self, df: pd.DataFrame, max_rul: int) -> pd.DataFrame:
        if "RUL" not in df.columns:
            raise ValueError("Column 'RUL' missing. Run _add_rul first.")
        df = df.copy()
        df["RUL"] = df["RUL"].clip(upper=max_rul)
        return df

    def _feature_cols(self, df: pd.DataFrame) -> List[str]:
        return [c for c in df.columns if c not in self.cfg.columns.non_feature]

    def _fit_transform(
        self,
        train: pd.DataFrame,
        test:  pd.DataFrame,
        feature_cols: List[str],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.scaler       = StandardScaler().fit(train[feature_cols])
        self.feature_cols_ = feature_cols

        train = train.copy()
        test  = test.copy()
        train[feature_cols] = self.scaler.transform(train[feature_cols])
        test[feature_cols]  = self.scaler.transform(test[feature_cols])
        print(f"✅ Scaled {len(feature_cols)} features (fit on train only).")
        return train, test


