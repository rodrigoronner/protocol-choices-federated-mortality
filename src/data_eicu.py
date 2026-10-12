"""
eICU Collaborative Research Database (v2.0) cohort, built to mirror the
MIMIC-IV cohort in src/data.py so that both run through the same client
construction, preprocessing and training code.

The point of this second dataset is the federation itself: eICU-CRD holds
ICU stays from 208 US hospitals, so each client is a real institution with
its own case mix, practice patterns and record-keeping, rather than a care
unit of a single hospital.

Cohort: one row per hospital stay (patienthealthsystemstayid), described by
its first ICU stay; adults (age >= 18; eICU's "> 89" is coded as 90);
outcome = in-hospital death (hospitaldischargestatus == "Expired"); stays
with no recorded discharge status are excluded. Features: age, gender,
ethnicity, ICU type, hospital admission source, and the same 11-lab panel as
MIMIC-IV, averaged over the first 24 hours of the ICU stay (labresultoffset
in [0, 1440] minutes). Every feature is available 24 hours after ICU
admission.

Only patient.csv.gz, hospital.csv.gz and lab.csv.gz are read, from
EICU_DIR (default ~/Data/eicu-crd/physionet.org/files/eicu-crd/2.0). The
derived cohort is cached under ~/Data/eicu-crd/derived, never in this repository:
the PhysioNet data use agreement forbids redistributing it.
"""
import os

import numpy as np
import pandas as pd

EICU_DIR = os.environ.get(
    "EICU_DIR", os.path.expanduser("~/Data/eicu-crd/physionet.org/files/eicu-crd/2.0"))
COHORT_CSV = os.environ.get(
    "EICU_COHORT_CSV", os.path.expanduser("~/Data/eicu-crd/derived/eicu_cohort.csv"))
COHORT_APS_CSV = COHORT_CSV.replace(".csv", "_aps.csv")

# Sensitivity analysis: worst first-day (APACHE day 1) physiology from
# apacheApsVar.csv.gz, which the closest prior work (Dang et al., 2022) also
# draws on (vitals, GCS, urine output). -1 marks a missing value there.
APS_COLS = {
    "heartrate": "aps_heartrate", "meanbp": "aps_meanbp", "temperature": "aps_temperature",
    "respiratoryrate": "aps_respiratoryrate", "eyes": "aps_gcs_eyes", "motor": "aps_gcs_motor",
    "verbal": "aps_gcs_verbal", "urine": "aps_urine", "intubated": "aps_intubated",
    "vent": "aps_vent", "dialysis": "aps_dialysis",
}

# eICU lab names for the MIMIC-IV panel in data.LAB_ITEMIDS. Glucose is the
# laboratory value (not "bedside glucose") and bicarbonate the chemistry
# value (not the blood-gas "HCO3"), matching the MIMIC-IV itemids.
LAB_NAMES = {
    "creatinine": "creatinine",
    "BUN": "bun",
    "sodium": "sodium",
    "potassium": "potassium",
    "glucose": "glucose",
    "WBC x 1000": "wbc",
    "Hgb": "hemoglobin",
    "Hct": "hematocrit",
    "platelets x 1000": "platelet",
    "bicarbonate": "bicarbonate",
    "lactate": "lactate",
}
LAB_WINDOW_MIN = (0, 24 * 60)


def _first_day_labs(stay_ids):
    parts = []
    for chunk in pd.read_csv(os.path.join(EICU_DIR, "lab.csv.gz"),
                             usecols=["patientunitstayid", "labresultoffset", "labname", "labresult"],
                             chunksize=5_000_000):
        chunk = chunk[chunk["labname"].isin(LAB_NAMES)
                      & chunk["labresultoffset"].between(*LAB_WINDOW_MIN)
                      & chunk["patientunitstayid"].isin(stay_ids)
                      & chunk["labresult"].notna()]
        parts.append(chunk.groupby(["patientunitstayid", "labname"])["labresult"].agg(["sum", "count"]))
    agg = pd.concat(parts).groupby(level=[0, 1]).sum()
    means = (agg["sum"] / agg["count"]).unstack("labname").rename(columns=LAB_NAMES)
    return means.reindex(columns=list(LAB_NAMES.values()))


def extract_cohort(force=False, with_aps=False):
    if with_aps:
        return _with_aps(extract_cohort(force=force), force=force)
    if os.path.exists(COHORT_CSV) and not force:
        return pd.read_csv(COHORT_CSV)

    pat = pd.read_csv(os.path.join(EICU_DIR, "patient.csv.gz"))
    pat = pat[pat["hospitaldischargestatus"].notna()]
    first = (pat.sort_values(["patienthealthsystemstayid", "unitvisitnumber", "hospitaladmitoffset"])
                .groupby("patienthealthsystemstayid", as_index=False).first())
    age = pd.to_numeric(first["age"].replace("> 89", "90"), errors="coerce")
    first = first.assign(age=age)
    first = first[first["age"] >= 18]

    labs = _first_day_labs(set(first["patientunitstayid"]))
    df = pd.DataFrame({
        "hadm_id": first["patienthealthsystemstayid"].values,
        "subject_id": first["uniquepid"].values,
        "site": [f"Hospital {h}" for h in first["hospitalid"]],
        "hospitalid": first["hospitalid"].values,
        "age": first["age"].values,
        "gender": first["gender"].fillna("Unknown").values,
        "race": first["ethnicity"].fillna("Unknown").values,
        "unittype": first["unittype"].fillna("Unknown").values,
        "admit_source": first["hospitaladmitsource"].fillna("Unknown").values,
        "hospital_expire_flag": (first["hospitaldischargestatus"] == "Expired").astype(int).values,
    })
    stay_labs = labs.reindex(first["patientunitstayid"].values)
    df = df.assign(**{c: stay_labs[c].values for c in labs.columns})

    os.makedirs(os.path.dirname(COHORT_CSV), exist_ok=True)
    df.to_csv(COHORT_CSV, index=False)
    return df


def _with_aps(df, force=False):
    """The main cohort plus the first-day APACHE physiology of each stay's
    first ICU stay (same rows, same clients)."""
    if os.path.exists(COHORT_APS_CSV) and not force:
        return pd.read_csv(COHORT_APS_CSV)
    pat = pd.read_csv(os.path.join(EICU_DIR, "patient.csv.gz"),
                      usecols=["patientunitstayid", "patienthealthsystemstayid", "unitvisitnumber",
                               "hospitaladmitoffset", "hospitaldischargestatus"])
    pat = pat[pat["hospitaldischargestatus"].notna()]
    first = (pat.sort_values(["patienthealthsystemstayid", "unitvisitnumber", "hospitaladmitoffset"])
                .groupby("patienthealthsystemstayid", as_index=False).first())
    aps = pd.read_csv(os.path.join(EICU_DIR, "apacheApsVar.csv.gz"),
                      usecols=["patientunitstayid"] + list(APS_COLS))
    aps = aps.drop_duplicates("patientunitstayid").set_index("patientunitstayid")
    aps = aps.where(aps != -1).rename(columns=APS_COLS)
    # Urine output carries charting errors (negative totals and values above
    # 250 L/day); anything outside 0-20,000 mL per day is treated as missing.
    aps["aps_urine"] = aps["aps_urine"].where(aps["aps_urine"].between(0, 20000))
    stay = first.set_index("patienthealthsystemstayid")["patientunitstayid"]
    out = df.join(aps.reindex(stay.reindex(df["hadm_id"]).values).set_index(df.index))
    out.to_csv(COHORT_APS_CSV, index=False)
    return out


def hospital_table(df):
    """Per-hospital size, deaths and mortality, joined with hospital.csv.gz
    (region, bed-count category, teaching status)."""
    hosp = pd.read_csv(os.path.join(EICU_DIR, "hospital.csv.gz"))
    g = df.groupby("hospitalid")["hospital_expire_flag"].agg(stays="size", deaths="sum")
    g["mortality"] = g["deaths"] / g["stays"]
    return g.join(hosp.set_index("hospitalid")).sort_values("stays", ascending=False)
