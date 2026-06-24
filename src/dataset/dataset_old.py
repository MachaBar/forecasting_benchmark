import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Dict, Any
import os
import datetime


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

from torch.utils.data import ConcatDataset

class MixedDataset(ConcatDataset):
    def __init__(
        self,
        parquet_path,
        npy_path,
        real_proportion=1.0,
        max_real_ids=1024,
        **kwargs
    ):
        real_ds = CustomDataset(
            path=parquet_path,
            proportion_ids=real_proportion,
            max_ids=max_real_ids,
            synth=False,
            **kwargs,
        )

        synth_ds = CustomDataset(
            path=npy_path,
            synth=True,
            **kwargs,
        )

        super().__init__([real_ds, synth_ds])

class CustomDataset(Dataset):

    def __init__(
        self,
        path: str,
        # timestamp_col: str = "HORODATAGE",
        timestamp_col: str = "time",
        context_length: int = 512, # 2 weeks (30 min timestep)
        prediction_length: int = 96, # 2 days (30 min timestep)
        stride: int = 96,
        list_pdl: Optional[List[str]] = None,
        proportion_ids: Optional[float] = None,
        seed: int = 42,
        normalize: bool = True,
        synth: bool = True,
        max_ids: Optional[int] = 1024
    ):

        super().__init__()

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.window_size = context_length + prediction_length
        self.stride = stride
        self.seed = seed
        self.normalize = True
        self.synth = synth

        set_seed(self.seed)

        ext = os.path.splitext(path)[1]
        print("DEBUG : ",timestamp_col)
        if ext == ".parquet":

            df = pl.read_parquet(path) # format large (wide) :(T = timesteps, N = nb ids)
            print(df.columns)

            self.dates = df[timestamp_col].to_numpy()
            df = df.drop(timestamp_col)

            all_ids = df.columns

            if list_pdl is not None and proportion_ids is not None:
                raise ValueError("Provide either list_pdl OR proportion_ids.")

            if list_pdl is not None:
                selected_ids = list_pdl

            elif proportion_ids is not None:
                rng = np.random.default_rng(self.seed)
                n_select = int(len(all_ids) * proportion_ids)
                selected_ids = rng.choice(all_ids, size=n_select, replace=False).tolist()
    

            else:
                selected_ids = all_ids

            all_ids = df.columns

            # if max_ids is not None:
            #     all_ids = all_ids[:max_ids]

            # if proportion_ids is not None:
            #     rng = np.random.default_rng(self.seed)

            #     n_select = max(
            #         1,
            #         int(len(all_ids) * proportion_ids)
            #     )

            #     selected_ids = rng.choice(
            #         all_ids,
            #         size=n_select,
            #         replace=False
            #     ).tolist()
            # else:
            #     selected_ids = all_ids

                        # if max_ids is not None:
                        #     selected_ids = selected_ids[:max_ids]

            self.selected_ids = selected_ids
            selected_ids = [str(i) for i in selected_ids]

            self.data = df.select(pl.col(selected_ids)).to_numpy()
            self.T, self.N = self.data.shape
            print("File is parquet, shape : ", self.data.shape)

        elif ext == ".npy":

            data = np.load(path)   # (N_ids, T) or (N_ids, D, H)
            print("Data shape:", data.shape)

            if data.ndim == 3:
                data = data.reshape(data.shape[0], -1)

            if data.ndim != 2:
                raise ValueError(f"Expected shape (N_ids, T), got {data.shape}")

            # transpose → (T, N)
            data = data.T

            self.T, self.N = data.shape
            print("File is numpy, shape:", data.shape)

            if self.synth:
                print("Clipping synthetic data")
                data = np.clip(data, 1e-8, 60000)

            print("min:", data.min())
            print("max:", data.max())

            # use all series
            self.selected_ids = np.arange(self.N)
            self.data = data

            # synthetic timestamps
            start_date = datetime.datetime(2021, 1, 1)
            delta = datetime.timedelta(minutes=30)

            self.dates = np.array([
                start_date + i * delta for i in range(self.T)
            ])

        # elif ext == ".npy":

        #     data = np.load(path)   # shape (N_ids, T)
        #     print("Data shape : ",data.shape)

        #     if data.ndim == 3:
        #         # flatten last two dims
        #         data = data.reshape(data.shape[0], -1)

        #     if data.ndim != 2:
        #         raise ValueError(f"Expected shape (N_ids, T), got {data.shape}")

        #     # transpose → (T, N)
        #     data = data.T


        #     self.T, self.N = data.shape
        #     print("File is numpy, shape : ", data.shape)
        #     if self.synth == True:
        #         print("Clipping: ")
        #         data = np.clip(data, 1e-8, 60000)

        #     print("min:", data.min())
        #     print("max:", data.max())

        #     all_ids = np.arange(self.N)

        #     if list_pdl is not None and proportion_ids is not None:
        #         raise ValueError("Provide either list_pdl OR proportion_ids.")

        #     if list_pdl is not None:
        #         selected_ids = list_pdl

        #     elif proportion_ids is not None:
        #         rng = np.random.default_rng(self.seed)
        #         n_select = int(len(all_ids) * proportion_ids)
        #         selected_ids = rng.choice(all_ids, size=n_select, replace=False).tolist()

        #     else:
        #         selected_ids = all_ids.tolist()

        #     self.selected_ids = selected_ids

        #     self.data = data[:, selected_ids]

        #     # create synthetic timestamps
        #     start_date = datetime.datetime(2021, 1, 1)
        #     delta = datetime.timedelta(minutes=30)

        #     self.dates = np.array([
        #         start_date + i * delta for i in range(self.T)
        #     ])

        else:
            raise ValueError("Unsupported file format. Use .parquet or .npy")

        self.samples = []
        self._create_samples()

    def _create_samples(self):

        from collections import defaultdict

        self.samples = []

        self.total_windows = 0
        self.skipped_nan = 0
        self.skipped_constant_context = 0

        self.clients_with_constant = set()
        self.client_constant_counts = defaultdict(int)

        for col_idx in range(self.N):

            series = self.data[:, col_idx]
            client_id = self.selected_ids[col_idx]

            for start_pos in range(
                0,
                self.T - self.window_size + 1,
                self.stride
            ):

                self.total_windows += 1

                end_pos = start_pos + self.window_size

                X = series[start_pos : start_pos + self.context_length]
                y = series[start_pos + self.context_length : end_pos]

                # ---- skip NaNs ----
                if np.isnan(X).any() or np.isnan(y).any():
                    self.skipped_nan += 1
                    continue

                # ---- skip constant context ----
                if np.std(X) < 1e-8:
                    self.skipped_constant_context += 1
                    self.clients_with_constant.add(client_id)
                    self.client_constant_counts[client_id] += 1
                    continue

                self.samples.append(
                    {
                        "col_idx": col_idx,
                        "start_pos": start_pos,
                        "end_pos": end_pos,
                    }
                )

        # ---- stats ----
        print("========== Dataset Statistics ==========")
        print(f"Total windows: {self.total_windows}")
        print(f"Skipped NaN windows: {self.skipped_nan}")
        print(f"Skipped constant context: {self.skipped_constant_context}")
        print(f"Remaining windows: {len(self.samples)}")

        if self.total_windows > 0:
            print(f"Constant-context ratio: {self.skipped_constant_context / self.total_windows:.4f}")

        print(f"Clients affected: {len(self.clients_with_constant)}")
        print("=========================================")

    # def _create_samples(self):

    #     for col_idx in range(self.N): # je parcours les clients

    #         series = self.data[:, col_idx]

    #         for start_pos in range(
    #             0,
    #             self.T - self.window_size + 1,
    #             self.stride
    #         ):

    #             end_pos = start_pos + self.window_size

    #             X = series[start_pos : start_pos + self.context_length]
    #             y = series[start_pos + self.context_length : end_pos]

    #             if np.isnan(X).any() or np.isnan(y).any():
    #                 continue

    #             self.samples.append(
    #                 {
    #                     "col_idx": col_idx,
    #                     "start_pos": start_pos,
    #                     "end_pos": end_pos,
    #                 }
    #             )



    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx: int) -> Dict[str, Any]:

        sample_info = self.samples[idx]

        col_idx = sample_info["col_idx"]
        start_pos = sample_info["start_pos"]
        end_pos = sample_info["end_pos"]

        series = self.data[:, col_idx]

        X = series[start_pos : start_pos + self.context_length]
        y = series[start_pos + self.context_length : end_pos]

        if self.normalize:
            mean = np.mean(X)
            # scale = np.std(X) + 1e-8
            scale = np.std(X)
            scale = max(scale, 1e-3)

            if scale < 0.01:
                print("SMALL SCALE DETECTED:", scale)
                print("X min/max:", X.min(), X.max())
                print("y min/max:", y.min(), y.max())

            X = (X - mean) / scale
            y = (y - mean) / scale
        else:
            mean = 0.0
            scale = 1.0

        return {
            "X": torch.tensor(X, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.float32),
            "mean": torch.tensor(mean, dtype=torch.float32),
            "scale": torch.tensor(scale, dtype=torch.float32),
            "start_X": str(self.dates[start_pos]),
            "start_y": str(self.dates[start_pos + self.context_length]),
            "item_id": f"{self.selected_ids[col_idx]}",
        }
    
    # After dataloader : 
    # X → torch.float32, shape = (B, context_length)
    # y → torch.float32, shape = (B, prediction_length)