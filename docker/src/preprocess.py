import matplotlib.pyplot as plt
import pandas as pd
from dotmap import DotMap


# files on server
flist_server = DotMap(
    condition="/data/train/CONDITION_OCCURRENCE_NICU.csv",
    measurement="/data/train/MEASUREMENT_NICU.csv",
    outcome="/data/train/OUTCOME_COHORT.csv",
    person="/data/train/PERSON_NICU.csv"
)
# sampple files
flist_sample = DotMap(
    condition="/data/train/sample_condition_occurrence_table.csv",
    measurement="/data/train/sample_measurement_table.csv",
    outcome="/data/train/sample_outcome_cohort_table.csv",
    person="/data/train/sample_person_table.csv"
)

# local files
flist_local = DotMap({k: f"../..{v}" for k, v in flist_sample.items()})

# FIXME: change to flist_server before deployment
flist = flist_local

person = pd.read_csv(flist.person)
outcome = pd.read_csv(flist.outcome,
                      parse_dates=["COHORT_START_DATE", "COHORT_END_DATE"]
                      )
msmt = pd.read_csv(flist.measurement,
                   parse_dates=["MEASUREMENT_DATETIME"])


plist = person.PERSON_ID.unique().tolist()


def get_outcome_hourly(df, person_id):
    outcome_cols = ["COHORT_END_DATE", "LABEL"]
    p_outcome = df[df.SUBJECT_ID == person_id][outcome_cols]
    return p_outcome.set_index("COHORT_END_DATE")


def get_measurement_hourly(df, person_id, agg="mean"):
    msmt_grp_cols = [
        # "PERSON_ID",
        "MEASUREMENT_DATETIME",
        "MEASUREMENT_SOURCE_VALUE",
    ]
    p_msmt = df[df.PERSON_ID == person_id]
    msmt_grp = p_msmt.groupby(msmt_grp_cols).agg(
        "mean")[["VALUE_SOURCE_VALUE"]].unstack()
    return msmt_grp.resample("H").agg(agg)


def plot_measurement_label(df, savefig=None):
    fig, ax = plt.subplots(figsize=(15, 15))
    df.LABEL = df.LABEL * 300
    df.plot(ax=ax)
    plt.legend(loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.1))
    if savefig:
        fig.savefig(savefig)


msmt_hourly = get_measurement_hourly(msmt, plist[0])
outcome_hourly = get_outcome_hourly(outcome, plist[0])
merged_hourly = pd.concat([outcome_hourly, msmt_hourly], axis=1)

plot_measurement_label(merged_hourly, 'output.png')
