import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def get_metrics_matrix(train, test, to_dataframe=False, return_origin=False):
    opened = train[train["push_opened"]==1]
    mins = (opened["push_opened_time"] - opened["push_time"]).dt.seconds / 60
    
    origin = test.copy()
    origin["push_hour"] = origin["push_opened_time"].dt.hour
    origin.loc[origin["push_opened"]==0, "push_opened"] = -1
    origin = origin.pivot_table(values="push_opened", index="user_id", columns="push_hour", aggfunc="sum")
    origin.fillna(0, inplace=True)
    origin = origin.reindex(columns=range(24), fill_value=0)
    
    total = origin.values
    positive = total.copy()
    positive[positive<0] = 0
    negative = total.copy()
    negative[negative>0] = 0
    
    not_negative = (total >= 0) * 1
    nn1 = np.roll(not_negative, -1, axis=1)
    nn2 = np.roll(not_negative, -2, axis=1)
    
    q1 = stats.percentileofscore(mins, 60) / 100
    q2 = stats.percentileofscore(mins, 120) / 100
    q3 = stats.percentileofscore(mins, 180) / 100
    
    result = total.copy()
    result += np.roll(positive, -1, axis=1) * (q1 - 0) * not_negative
    result += np.roll(positive, -2, axis=1) * (q2 - q1) * not_negative * nn1
    result += np.roll(positive, -3, axis=1) * (q3 - q2) * not_negative * nn1 * nn2
    result += np.roll(positive, 1, axis=1) * (1 - q3) * not_negative
    
    res_pos = result.copy()
    res_pos[res_pos < 0] = 0
    divider = np.maximum(res_pos.max(axis=1), np.ones(len(res_pos)))[:,None]
    res_pos /= divider

    res_neg = result.copy()
    res_neg[res_neg > 0] = 0
    divider = np.maximum(abs(res_neg.min(axis=1)), np.ones(len(res_neg)))[:,None]
    res_neg /= divider
    result = res_neg + res_pos
    
    if to_dataframe:
        result = pd.DataFrame(result, index=origin.index, columns=origin.columns)
    if return_origin:
        return result, origin
    return result

def plot_metrics_user(userid, origin, result):
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10, 6))
    fig.suptitle(f"M e t r i c s    f o r    u s e r i d    '{userid}'", fontsize=14)
    ax1.set_title("O R I G I N", fontsize=12)
    ax2.set_title("R E S U L T", fontsize=12)
    _origin = origin.loc[userid]
    _result = result.loc[userid]

    pal = sns.color_palette("flare", len(_origin))
    rank_origin = _origin.argsort().argsort()
    rank_result = _origin.argsort().argsort()
    
    ax1.axhline(y=0, color='k', alpha=0.5)
    ax2.axhline(y=0, color='k', alpha=0.5)
    
    sns.barplot(y=_origin.values, x=_origin.index, ax=ax1, palette=np.array(pal[::-1])[rank_origin])
    sns.barplot(y=_result.values, x=_result.index, ax=ax2, palette=np.array(pal[::-1])[rank_result])
    
    ax1.set_ylim(min(-1, _origin.min()), max(1, _origin.max()))
    ax2.set_ylim(-1, 1)
    
    plt.tight_layout()
    plt.show()

def plot_duration_dist(df):
    opened = df[df["push_opened"]==1]
    seconds = (opened["push_opened_time"] - opened["push_time"]).dt.seconds
    mins = seconds / 60
    hours = mins / 60

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("Распределение времени открытия пуша")
    seconds.hist(bins=range(60), ax=ax1)
    mins.hist(bins=range(60), ax=ax2)
    hours.hist(bins=range(10), ax=ax3)
    ax1.set_xlabel("seconds", fontsize=12)
    ax2.set_xlabel("minutes", fontsize=12)
    ax3.set_xlabel("hours", fontsize=12)
    plt.show()


def plot_push_time_dist(df):
    df["push_hour"] = df["push_time"].dt.hour
    df["push_day_hour"] = df["push_time"].dt.strftime("d%d h%H")
    day_hour = df.groupby("push_day_hour")["push_day_hour"].count()
    opened = df[df["push_opened"]==1]
    opened_day_hour = opened.groupby("push_day_hour")["push_day_hour"].count()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 8))
    fig.suptitle("Распределение времени отправки и чтения пушей", fontsize=16)
    df["push_hour"].hist(bins=24, ax=ax1, color="b")
    df["push_hour"].hist(bins=24, ax=ax3, color="b")
    sns.barplot(x=day_hour.index, y=day_hour, ax=ax2, color="b")
    sns.barplot(x=day_hour.index, y=day_hour, ax=ax4, color="b")
    opened["push_hour"].hist(bins=24, ax=ax1, color="r")
    opened["push_hour"].hist(bins=24, ax=ax3, color="r")
    sns.barplot(x=opened_day_hour.index, y=opened_day_hour, ax=ax2, color="r")
    sns.barplot(x=opened_day_hour.index, y=opened_day_hour, ax=ax4, color="r")
    ax2.locator_params(nbins=6)
    ax4.locator_params(nbins=6)
    ax1.set_xlim(0, 23)
    ax3.set_xlim(0, 23)
    ax2.set_xlim(0, len(day_hour))
    ax4.set_xlim(0, len(opened_day_hour))
    ax3.set_yscale('log')
    ax4.set_yscale('log')
    ax2.set_ylabel("")
    ax2.set_xlabel("")
    ax4.set_ylabel("")
    ax1.set_ylabel("absolute scale", fontsize=14)
    ax3.set_xlabel("hours", fontsize=14)
    ax3.set_ylabel("log scale", fontsize=14)
    ax4.set_xlabel("day-hours", fontsize=14)
    plt.tight_layout()
    plt.show()

def calc_metrics(metrics_matrix, preds):
    assert preds["user_id"].nunique() == len(preds), "column user_id in preds DataFrame is not unique"
    preds_matrix = preds.copy()
    preds_matrix["value"] = 1
    preds_matrix = preds_matrix.pivot_table(values="value", index="user_id", columns="best_push_hour", aggfunc="count")
    preds_matrix.fillna(0, inplace=True)
    preds_matrix = preds_matrix.reindex(columns=range(24), fill_value=0)
    users = preds_matrix.index
    
    _metrics_matrix = metrics_matrix.loc[users,:].copy()
    res = _metrics_matrix.values * preds_matrix.values
    res = res.sum(1)
    return res
