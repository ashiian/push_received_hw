import pandas as pd
import pickle
import os
from model import ExplicitCF


class BestTimeTrainer:

    def __init__(self, data_folder) -> None:
        self.data_folder = data_folder
        self.data = None
        self.users = None
        self.model = None
        self.preds = None

    def run(self):
        print("started loading data")
        self._load_data()
        print("data loaded")
        self._get_unique_userids()
        self._process_data()
        print("data processed")
        self._apply_model()
        print("model applied")
        self._save_preds()
        print("preds saved")
        self._save_model()
        print("model saved")

    def _load_data(self):
        data_path = os.path.join(self.data_folder, "raw_push_stat_2022_01_13_.gz")
        self.data = pd.read_csv(data_path, delimiter=';')
    
    def _get_unique_userids(self):
        self.users = self.data[["user_id"]].drop_duplicates()

    def _process_data(self):
        self._drop_duplicates()
        self._date_cols_to_datetime()
        self._add_tag_pushes_same_time()
        self._add_tag_wrong_push_opened_time()
        self._filter_tags_above()

    def _drop_duplicates(self):
        self.data.drop_duplicates(inplace=True)
        
    def _date_cols_to_datetime(self):
        self.data["push_time"] = pd.to_datetime(self.data["push_time"])
        self.data["push_opened_time"] = pd.to_datetime(self.data["push_opened_time"])
        self.data["create_at"] = pd.to_datetime(self.data["create_at"])

    def _add_tag_pushes_same_time(self):
        tmp = self.data[["user_id", "push_time"]].value_counts()
        tmp.name = "is_two_push_same_time"
        tmp = tmp[tmp>1].reset_index()
        tmp["is_two_push_same_time"] = True
        if "is_two_push_same_time" in self.data.columns:
            self.data.drop(columns="is_two_push_same_time", inplace=True)
        self.data = self.data.merge(tmp, on=["user_id", "push_time"], how="left")
        self.data["is_two_push_same_time"].fillna(False, inplace=True)

    def _add_tag_wrong_push_opened_time(self):
        self.data["is_wrong_push_opened_time"] = False
        mask = (self.data["push_opened_time"] > self.data["push_time"]) & (self.data["push_opened"]==0)
        self.data.loc[mask, "is_wrong_push_opened_time"] = True

    def _filter_tags_above(self):
        drop_cols = [x for x in self.data.columns if x.startswith("is_")]
        mask = ~self.data[drop_cols].any(1)
        self.data = self.data[mask]
        self.data.drop(columns=drop_cols, inplace=True)

    def _apply_model(self):
        self.model = ExplicitCF()
        self.model.fit(self.data)
        self.preds = self.model.predict(self.users)

    def _save_preds(self):
        preds_path = os.path.join(self.data_folder, "best_time_preds.csv")
        self.preds.to_csv(preds_path, index=None)

    def _save_model(self):
        model_path = os.path.join(self.data_folder, "model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

if __name__ == "__main__":
    trainer = BestTimeTrainer("./src/data")
    trainer.run()
