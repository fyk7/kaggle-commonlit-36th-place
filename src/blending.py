import numpy as np
import pandas as pd
import optuna
import hydra
import typing as t
from omegaconf import DictConfig
from sklearn.metrics import mean_squared_error
from pathlib import Path


def rmse(y_true: np.array, y_pred: np.array) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))


# TODO 汎用的に使用できるクラスにする(引数をpd.DataFrameにしないでnp.arrayにするetc...)
class CustomObjective(object):
    def __init__(self, n_models: int, blend_df: pd.DataFrame):
        self.n_models = n_models
        self.blend_df = blend_df.copy()

    def calc_score(
        self,
        y_preds_df: pd.DataFrame,
        y_true: t.Union[pd.Series, np.array],
        w: t.List[t.Any]
    ) -> float:
        y_preds_df_X = y_preds_df.drop(columns=["id", "target"])
        pred_blended = np.average(
            y_preds_df_X.values,
            axis=1, weights=w
        )
        return rmse(pred_blended, y_true)

    def __call__(self, trial):
        ws = [
            trial.suggest_uniform("w" + str(n), 0, 1)
            for n in range(self.n_models)
        ]
        return self.calc_score(
          self.blend_df, self.blend_df["target"], ws)


def get_best_weight(
    objective: CustomObjective,
    n_trials: int = 200,
    n_jobs: int = 1,
    seed: int = 88
) -> t.Tuple[np.array, float]:
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    best_w = list(study.best_params.values())
    best_w = np.array(best_w) / np.sum(best_w)
    best_score = study.best_value

    return best_w, best_score


def calc_bs_lg_ratio(y_true, y_lg_blend, y_bs_blend):
    """BertBase系とBertLarge系の割合を決定
    """
    if (len(y_true) != len(y_bs_blend)) or \
        (len(y_true) != len(y_bs_blend)):
        raise ValueError("正解のyの数と予測のyの数が合っていません!")
    min_rmse = 9999
    min_i = 0
    for i in np.arange(0, 1, 0.001):
        tmp_rmse = rmse(
            y_true,
            i * y_lg_blend + (1 - i) * y_bs_blend
        )
        if tmp_rmse < min_rmse:
            min_rmse = tmp_rmse
            min_i = i
    return min_rmse, min_i


@hydra.main(config_name="config", config_path="../conf")
def main(cfg: DictConfig):
    BASE_DIR = Path(cfg.base_dir)
    INPUT_DIR = BASE_DIR / "input"
    TRAIN_DATA_PATH = INPUT_DIR / "train.csv" # "train_toy.csv"
    OUTPUT_DIR = BASE_DIR / "outputs"

    np.random.seed(1000)

    base_df = pd.read_csv(TRAIN_DATA_PATH)
    base_df = base_df.drop(
        base_df[(base_df.target == 0) & (base_df.standard_error == 0)].index
    )
    base_df = base_df.reset_index(drop=True)

    blend_material_path_lg = OUTPUT_DIR / "commonlit_blending_material_lg"
    blend_material_path_bs = OUTPUT_DIR / "commonlit_blending_material_bs"

    # TODO configから渡す
    path_list_lg = [
        "rbt_lg1_pred.csv",
        "eltr_lg1_pred.csv",
        "dbt_lg_pred.csv",
        "cta_lg2_pred.csv"
    ]
    merge_df_lg = base_df.copy()
    for p in path_list_lg:
        tmp_df = pd.read_csv(blend_material_path_lg / p)
        tmp_df = tmp_df[["id", "target"]]
        # カラム名が"target"だとmergeした際に名称が被るためrenameする
        tmp_df.columns = ["id", f"target_{p.split('.')[0]}"]
        merge_df_lg = merge_df_lg.merge(tmp_df, how="left", on="id")

    path_list_bs = [
        "rbt-bs-strat-url-16-256.csv",
        "eltr-bs-strat-url-16-256.csv",
        "dbt-bs-strat-url-16-256.csv",
        "rbt-bs-cta-all-url-16-256.csv",
        "dbt-bs-cta-all-strat-url-16-256.csv",
    ]
    merge_df_bs = base_df.copy()
    for p in path_list_bs:
        tmp_df = pd.read_csv(blend_material_path_bs / p)
        tmp_df = tmp_df[["id", "target"]]
        # カラム名が"target"だとmergeした際に名称が被るためrenameする
        tmp_df.columns = ["id", f"target_{p.split('.')[0]}"]
        merge_df_bs = merge_df_bs.merge(tmp_df, how="left", on="id")


    blend_target_cols_lg = \
        [f"target_{p.split('.')[0]}" for p in path_list_lg]
    blend_df_lg = merge_df_lg[["id", "target"] + blend_target_cols_lg]
    blend_target_cols_bs = \
        [f"target_{p.split('.')[0]}" for p in path_list_bs]
    blend_df_bs = merge_df_bs[["id", "target"] + blend_target_cols_bs]


    objective_lg = CustomObjective(
        n_models=len(blend_target_cols_lg),
        blend_df=blend_df_lg
    )
    objective_bs = CustomObjective(
        n_models=len(blend_target_cols_bs),
        blend_df=blend_df_bs
    )
    best_w_lg, best_score_lg = \
        get_best_weight(objective_lg, n_trials=200, n_jobs=-1)
    best_w_bs, best_score_bs = \
        get_best_weight(objective_bs, n_trials=200, n_jobs=-1)


    lg_blend: np.array = np.zeros(len(blend_df_lg))
    for col, weight in zip(blend_target_cols_lg, best_w_lg):
        lg_blend += blend_df_lg[col].values * weight
    bs_blend: np.array = np.zeros(len(blend_df_bs))
    for col, weight in zip(blend_target_cols_bs, best_w_bs):
        bs_blend += blend_df_bs[col].values * weight


    min_rmse, min_i = \
        calc_bs_lg_ratio(merge_df_lg["target"], lg_blend, bs_blend)
    # TODO モデルとweightの割合のリストを出力する
    print(min_rmse)
    print(min_i)


if __name__ == "__main__":
    main()
