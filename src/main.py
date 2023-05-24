from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import recall_score

from model.CAP import CAP


class DataframeFormatter:
    def __init__(self, df: pd.DataFrame, mapping: dict, labeled: bool = False):
        self.df = df.copy()
        self.mapping = mapping
        self.labeled = labeled

    def format(self):
        self._label_encode("geo_uid")
        self._extract_hour("gps_timestamp")
        if self.labeled:
            self._feature_vec_to_list("feature_vec")
        return self.df

    def _label_encode(self, column: str):
        self.df[column] = self.df[column].map(self.mapping)

    def _extract_hour(self, column: str):
        self.df[column] = pd.to_datetime(self.df[column], unit="s").dt.hour

    def _feature_vec_to_list(self, column: str):
        self.df[column] = self.df[column].apply(lambda x: [int(i.split(":")[0]) for i in x.split()])


@hydra.main(version_base=None, config_path=".", config_name="cap_config")
def main(cfg: DictConfig):
    # Data retrieval
    parent_dir = Path(__file__).resolve().parents[1]
    labeled_df = pd.read_csv(parent_dir.joinpath(cfg.data.folder, cfg.data.labeled_data))
    unlabeled_df = pd.read_csv(parent_dir.joinpath(cfg.data.folder, cfg.data.unlabeled_data))

    # Data formatting
    counts = labeled_df["geo_uid"].value_counts()
    mapping = {k: i for i, k in enumerate(counts.index, 0)}

    labeled_df_formatter = DataframeFormatter(labeled_df, mapping=mapping, labeled=True)
    unlabeled_df_formatter = DataframeFormatter(unlabeled_df, mapping=mapping)
    labeled_df_formatted = labeled_df_formatter.format()
    unlabeled_df_formatted = unlabeled_df_formatter.format()
    D = np.vstack(
        (
            labeled_df_formatted.drop(labeled_df_formatted.columns[-2:], axis=1).to_numpy(),
            unlabeled_df_formatted.to_numpy(),
        )
    )
    unlabeled_df_formatted = unlabeled_df_formatted.assign(category=0)

    # Convert nested DictConfig to plain dict
    hyperparameters = OmegaConf.to_container(cfg.hyperparameters, resolve=True)
    entities = OmegaConf.to_container(cfg.entities, resolve=True)

    hyperparameters["mu_0"] = np.mean(D[:, [1, 2]], axis=0)
    hyperparameters["psi_0"] = np.cov(D[:, [1, 2]], rowvar=False)
    hyperparameters["nu_0"] = np.mean(D[:, [3]], axis=0)
    hyperparameters["epsilon_0"] = np.std(D[:, [3]], axis=0)

    globals().update(hyperparameters)
    globals().update(entities)

    # Collapsed gibbs sampling for N user
    for u in range(cfg.evaluation.N):
        user_u_labeled = labeled_df_formatted[labeled_df_formatted["geo_uid"] == u]
        user_u_unlabeled = unlabeled_df_formatted[unlabeled_df_formatted["geo_uid"] == u]

        cap = CAP(**hyperparameters, **entities)

        cap.collapsed_gibbs_sampling(
            cfg.collapsed_gibbs_sampling.M,
            labeled_df_formatted.drop(labeled_df_formatted.columns[-1], axis=1).to_numpy(),
            user_u_unlabeled.to_numpy(),
        )

        # GPS annotation
        user_u_labeled["prediction"] = cap.annotate_records(
            user_u_labeled.iloc[:, 0:4].to_numpy(), np.array(user_u_labeled["feature_vec"].to_list())
        )

        if u == 0:
            result = user_u_labeled
        else:
            result = pd.concat([result, user_u_labeled], axis=0, ignore_index=True)
        # Evaluation
        micro_recall = recall_score(result["correct_category_index"], result["prediction"], average="micro")
        print(f"Micro Recall: {micro_recall}")

    # Evaluation
    micro_recall = recall_score(result["correct_category_index"], result["prediction"], average="micro")
    print(result.head())
    print(f"Micro Recall: {micro_recall}")


if __name__ == "__main__":
    main()
