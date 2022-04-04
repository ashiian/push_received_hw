import numpy as np
from pandas import DataFrame
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.preprocessing import LabelEncoder


class BaselineModel:

    def __init__(self, time_col="push_time") -> None:
        self.time_col = time_col
        self.best_hour = None

    def fit(self, df):
        X = df[["user_id", "push_opened", self.time_col]].copy()
        X["push_hour"] = X[self.time_col].dt.hour
        X = X.groupby(["user_id", "push_hour"])[["push_opened"]].sum()
        X.reset_index("push_hour", inplace=True)
        X.loc[X["push_opened"]>0, "push_opened"] = 1
        X = X.groupby("push_hour")["push_opened"].agg(["sum", "count"])
        X["pct"] = X["sum"] / X["count"]
        self.best_hour = X["pct"].idxmax()

    def predict(self, df):
        assert self.best_hour is not None, "Model is not fitted yet"
        preds = df[["user_id"]].copy()
        preds["best_push_hour"] = self.best_hour
        return preds


class SimpleModel:

    def __init__(self, time_col="push_opened_time") -> None:
        self.time_col = time_col
        self.baseline = BaselineModel("push_time")
        self._preds_on_train = None

    def fit(self, df):
        self.baseline.fit(df)
        
        X = df[["user_id", "push_opened", self.time_col]].copy()
        X["open_hour"] = X[self.time_col].dt.hour
        X = X.groupby(["user_id", "open_hour"])[["push_opened"]].sum()
        X = X[X["push_opened"]>0]
        X.reset_index(inplace=True)
        X.sort_values(by=["user_id", "open_hour"], inplace=True)
        X["desc_rank"] = X.groupby("user_id")["push_opened"].rank(ascending=False, method="first")
        X = X.loc[X["desc_rank"]==1, ["user_id", "open_hour"]]
        X.rename(columns={"open_hour": "best_push_hour"}, inplace=True)
        self._preds_on_train = X

    def predict(self, df):
        assert self._preds_on_train is not None, "Model is not fitted yet"
        preds = df[["user_id"]].copy()
        preds = preds.merge(self._preds_on_train, on="user_id", how="left")
        preds.fillna({"best_push_hour": self.baseline.best_hour}, inplace=True)
        preds["best_push_hour"] = preds["best_push_hour"].astype("int8")
        return preds


class ExplicitCF:

    def __init__(self, time_col="push_opened_time", latent_space=12) -> None:
        self.time_col = time_col
        self.latent_space = latent_space
        self._le_user = LabelEncoder()
        self._le_hour = LabelEncoder()
        self.baseline = BaselineModel("push_time")
        self._preds_on_train = None

    def fit(self, df):
        self.baseline.fit(df)
        
        X = df[["user_id", "push_opened", self.time_col]].copy()
        X["push_hour"] = X[self.time_col].dt.hour
        X.loc[X["push_opened"]==0, "push_opened"] = -1
        X = X.groupby(["user_id", "push_hour"])[["push_opened"]].sum()
        X.loc[X["push_opened"]>0, "push_opened"] = 1
        X.loc[X["push_opened"]<0, "push_opened"] = -1
        X.reset_index(inplace=True)
        
        user_ind = self._le_user.fit_transform(X["user_id"])
        hour_ind = self._le_hour.fit_transform(X["push_hour"])
        pushes = X["push_opened"].values
        
        m = len(np.unique(user_ind))
        n = len(np.unique(hour_ind))
        sm = csr_matrix((pushes, (user_ind, hour_ind)), shape=(m, n))
        sm = sm.asfptype()
        U, s, C = svds(sm, k=self.latent_space)
        s = np.sqrt(s)
        s = np.diag(s)
        res = (U @ s) @ (s @ C)

        best_hour = DataFrame({"user_id_encoded": range(len(res)), "hour_encoded": res.argmax(1)})
        best_hour["user_id"] = self._le_user.inverse_transform(best_hour["user_id_encoded"])
        best_hour["best_push_hour"] = self._le_hour.inverse_transform(best_hour["hour_encoded"])
        
        self._preds_on_train = best_hour[["user_id", "best_push_hour"]]

    def predict(self, df):
        assert self._preds_on_train is not None, "Model is not fitted yet"
        preds = df[["user_id"]].copy()
        preds = preds.merge(self._preds_on_train, on="user_id", how="left")
        preds.fillna({"best_push_hour": self.baseline.best_hour}, inplace=True)
        preds["best_push_hour"] = preds["best_push_hour"].astype("int8")
        return preds
