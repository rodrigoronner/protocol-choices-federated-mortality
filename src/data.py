"""
Cohort extraction and federated client construction for MIMIC-IV.

The cohort is built from the official MIMIC-IV v3.1 hosp tables (see
extract_cohort); eICU-CRD is handled in data_eicu.py and shares everything
below the extraction step.

Design decisions that matter for the evaluation protocol:

1. Class-imbalance treatments (class weights, SMOTE-Tomek) are applied only
   to each client's training partition. Validation and test partitions keep
   each client's natural mortality rate, so every reported metric reflects
   the true class distribution.

2. Splits are made at the patient level (all stays of a patient fall in the
   same partition), the shared preprocessing is fitted on training rows only,
   and every predictor is available within 24 hours of admission.
"""
import os

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COL = "hospital_expire_flag"
# Categorical features, used when present in the cohort: MIMIC-IV has
# insurance; eICU-CRD has the ICU type and the admission source instead.
CATEGORICAL_CANDIDATES = ["gender", "race", "insurance", "unittype", "admit_source"]

# Basic metabolic panel + complete blood count + lactate, averaged over the
# first 24 h; itemids checked against mimiciv_hosp.d_labitems. ICU vital
# signs (chartevents) are not used because most MIMIC-IV admissions never
# reach an ICU, whereas laboratory tests are recorded hospital-wide (lactate
# has low coverage, and its absence is itself informative).
LAB_ITEMIDS = {
    "creatinine": 50912,
    "bun": 51006,
    "sodium": 50983,
    "potassium": 50971,
    "glucose": 50931,
    "wbc": 51301,
    "hemoglobin": 51222,
    "hematocrit": 51221,
    "platelet": 51265,
    "bicarbonate": 50882,
    "lactate": 50813,
}
LAB_COLS = list(LAB_ITEMIDS.keys())
# Every feature is available within 24 hours of admission. Counts of the
# admission's ICD codes are deliberately excluded: the codes are assigned at
# discharge, so the counts leak end-of-stay information (measured in
# experiments/leakage_check.py).
NUMERIC_COLS = ["age"] + LAB_COLS
# Missingness itself is informative in EHR data (a lab never ordered is not
# missing-at-random); one indicator per lab, added as extra numeric columns
# after imputation in _fit_shared_preprocessor/_transform.
LAB_MISSING_COLS = [f"{c}_missing" for c in LAB_COLS]
# First-day physiology used only by the eICU sensitivity analysis
# (dataset "eicu_aps", see data_eicu.APS_COLS); imputed and flagged when
# missing like the labs. Absent from the main cohorts, which keep exactly
# the feature set above.
EXTRA_NUMERIC_CANDIDATES = [
    "aps_heartrate", "aps_meanbp", "aps_temperature", "aps_respiratoryrate",
    "aps_gcs_eyes", "aps_gcs_motor", "aps_gcs_verbal", "aps_urine",
    "aps_intubated", "aps_vent", "aps_dialysis",
]

# The five largest care units by volume, kept as the "top5" variant only:
# a size cutoff is arbitrary here (the 5th and 6th largest units differ by
# about 1,000 admissions) and admits Labor & Delivery, which has a single
# death in 20,337 admissions.
CLIENT_CAREUNITS = [
    "Emergency Department",
    "Discharge Lounge",
    "Labor & Delivery",
    "Medicine/Cardiology",
    "Medicine",
]

# Eligibility rule for the default client set: a care unit becomes a client
# when it has at least this many in-hospital deaths. With a 70/10/20
# patient-level split this leaves every client roughly >=70 positive training
# examples (enough for SMOTE's k-NN search) and >=20 positive test examples
# (enough for a per-client AUC to be meaningful). On MIMIC-IV this selects
# 12 units, from the Emergency Department (374k admissions, 2.3% mortality)
# to the MICU (2.1k admissions, 20.3% mortality).
MIN_DEATHS_PER_CLIENT = 100
OTHER_CLIENT = "Other units"

CLIENT_CONFIGS = ("top5", "min_deaths", "min_deaths200", "min_deaths_other", "min_deaths_edsplit4")
IMBALANCE_MODES = ("smote_tomek", "weighted", "none")

MIN_SAMPLES_FOR_SMOTE = 6  # k_neighbors=5 + 1, matches the paper's appendix guard

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RAW_CSV_PATH = os.path.join(DATA_DIR, "mimiciv_cohort.csv")

# SQLAlchemy URL of a PostgreSQL database holding MIMIC-IV v3.1 (schema
# mimiciv_hosp, as built by the MIT-LCP mimic-code scripts).
DEFAULT_DSN = os.environ.get("MIMIC_DSN", "postgresql+psycopg2://localhost/mimiciv")


def extract_cohort(dsn=DEFAULT_DSN, force=False):
    """Pull the cohort for ALL care units from the live MIMIC-IV DB and cache
    it locally as CSV; the client configuration (assign_clients) decides
    which units are used. The CSV is never committed (see .gitignore): the
    PhysioNet data use agreement forbids redistributing MIMIC-IV data, even
    derived, to anyone without their own credentialed access.
    """
    if os.path.exists(RAW_CSV_PATH) and not force:
        df = pd.read_csv(RAW_CSV_PATH)
        return df.assign(site=df["first_careunit"])

    from sqlalchemy import create_engine

    itemids_sql = ", ".join(str(i) for i in LAB_ITEMIDS.values())
    lab_select = ",\n               ".join(
        f"AVG(CASE WHEN le.itemid = {itemid} THEN le.valuenum END) AS {name}"
        for name, itemid in LAB_ITEMIDS.items()
    )
    # Self-contained cohort query on the official MIMIC-IV v3.1 hosp tables:
    # adult admissions (anchor_age >= 18), the care unit of the first transfer
    # of each admission (ties on intime broken by transfer_id), the outcome,
    # and the 11 laboratory values averaged over the first 24 h.
    query = f"""
        WITH first_careunit AS (
            SELECT hadm_id, careunit AS first_careunit
            FROM (
                SELECT t.hadm_id, t.careunit,
                       ROW_NUMBER() OVER (PARTITION BY t.hadm_id
                                          ORDER BY t.intime, t.transfer_id) AS rn
                FROM mimiciv_hosp.transfers t
                WHERE t.hadm_id IS NOT NULL
            ) ranked
            WHERE rn = 1
        ),
        labs_24h AS (
            SELECT le.hadm_id,
               {lab_select}
            FROM mimiciv_hosp.labevents le
            JOIN mimiciv_hosp.admissions a ON le.hadm_id = a.hadm_id
            WHERE le.itemid IN ({itemids_sql})
              AND le.valuenum IS NOT NULL
              AND le.charttime BETWEEN a.admittime AND a.admittime + interval '24 hours'
            GROUP BY le.hadm_id
        )
        SELECT a.hadm_id, a.subject_id, p.anchor_age AS age, p.gender, a.race, a.insurance,
               f.first_careunit, a.hospital_expire_flag,
               l.creatinine, l.bun, l.sodium, l.potassium, l.glucose, l.wbc,
               l.hemoglobin, l.hematocrit, l.platelet, l.bicarbonate, l.lactate
        FROM mimiciv_hosp.admissions a
        JOIN mimiciv_hosp.patients p ON a.subject_id = p.subject_id
        LEFT JOIN first_careunit f ON a.hadm_id = f.hadm_id
        LEFT JOIN labs_24h l ON a.hadm_id = l.hadm_id
        WHERE p.anchor_age >= 18
          AND a.hospital_expire_flag IS NOT NULL
    """
    engine = create_engine(dsn)
    df = pd.read_sql(query, engine)

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(RAW_CSV_PATH, index=False)
    return df.assign(site=df["first_careunit"])


def load_cohort(dataset="mimic"):
    """Cohort with one row per hospital admission and the columns site
    (the care unit or hospital that may become a client), subject_id,
    hospital_expire_flag and the features."""
    if dataset == "mimic":
        return extract_cohort()
    if dataset in ("eicu", "eicu_aps"):
        import data_eicu
        return data_eicu.extract_cohort(with_aps=dataset == "eicu_aps")
    raise ValueError(dataset)


DATASETS = ("mimic", "eicu", "eicu_aps")


def _categorical_cols(df):
    return [c for c in CATEGORICAL_CANDIDATES if c in df.columns]


def _imputed_cols(df):
    """Columns imputed with the training median and given a missingness
    flag: the labs, plus the first-day physiology when present."""
    return LAB_COLS + [c for c in EXTRA_NUMERIC_CANDIDATES if c in df.columns]


def _numeric_cols(df):
    return NUMERIC_COLS + [c for c in EXTRA_NUMERIC_CANDIDATES if c in df.columns]


def _fit_shared_preprocessor(df):
    """Fit OneHotEncoder/median-imputer/StandardScaler on the POOLED data.
    This only shares encoding vocabulary/imputation values/scale across
    clients (a standard, label-free assumption — every client uses the same
    feature schema) — it never shares label information or crosses the
    train/test boundary, since it's refit inside each seed's pooled TRAIN
    split only (see build_clients).
    """
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ohe.fit(df[_categorical_cols(df)].astype(str))
    lab_medians = df[_imputed_cols(df)].astype(np.float32).median()
    imputed = df[_numeric_cols(df)].astype(np.float32).fillna(lab_medians)
    scaler = StandardScaler()
    scaler.fit(imputed)
    return ohe, scaler, lab_medians


def _transform(df, ohe, scaler, lab_medians):
    cat = ohe.transform(df[_categorical_cols(df)].astype(str))
    missing = df[_imputed_cols(df)].isna().astype(np.float32).values
    imputed = df[_numeric_cols(df)].astype(np.float32).fillna(lab_medians)
    num = scaler.transform(imputed)
    X = np.hstack([num, missing, cat]).astype(np.float32)
    y = df[TARGET_COL].astype(np.float32).values
    return X, y


def eligible_units(df, min_deaths=MIN_DEATHS_PER_CLIENT):
    """Sites (care units or hospitals) with at least `min_deaths` deaths,
    largest first."""
    g = df.groupby("site")[TARGET_COL].agg(["size", "sum"])
    g = g[g["sum"] >= min_deaths].sort_values("size", ascending=False)
    return list(g.index)


def assign_clients(df, config="min_deaths", seed=42, ed_shards=4):
    """Maps every admission to a client name, or NaN when the admission's
    care unit is not part of the configuration.

      top5               the five largest units (the original client set)
      min_deaths         units with >= MIN_DEATHS_PER_CLIENT deaths
      min_deaths200      units with >= 200 deaths (fewer, larger clients)
      min_deaths_other   min_deaths + one extra client pooling every other unit
      min_deaths_edsplit4  min_deaths, with the Emergency Department split
                         into `ed_shards` clients by patient (no patient is
                         in two shards), to test more, smaller participants
    """
    unit = df["site"]
    if config == "top5":
        return unit.where(unit.isin(CLIENT_CAREUNITS))
    units = eligible_units(df, 200 if config == "min_deaths200" else MIN_DEATHS_PER_CLIENT)
    client = unit.where(unit.isin(units))
    if config == "min_deaths200":
        return client
    if config == "min_deaths":
        return client
    if config == "min_deaths_other":
        known = unit.notna() & ~unit.isin(["UNKNOWN", "Unknown"])
        return client.where(client.notna() | ~known, OTHER_CLIENT)
    if config == "min_deaths_edsplit4":
        ed = "Emergency Department"
        pats = np.sort(df.loc[unit == ed, "subject_id"].unique())
        rng = np.random.RandomState(seed)
        shard = pd.Series(rng.randint(0, ed_shards, size=len(pats)), index=pats)
        is_ed = unit == ed
        client = client.astype(object)
        client[is_ed] = [f"{ed} {k + 1}" for k in df.loc[is_ed, "subject_id"].map(shard)]
        return client
    raise ValueError(config)


def apply_smote_tomek(X, y, seed=42):
    """Safely applies SMOTE-Tomek, guarding against clients whose minority
    class is too small for SMOTE's k-neighbors search (k_neighbors=5 by
    default, so >=6 minority samples are required). Falls back to returning
    the data unchanged when the guard trips — the caller is expected to
    compensate with class-weighted loss in that case.
    """
    n_minority = int(min((y == 0).sum(), (y == 1).sum()))
    if n_minority < MIN_SAMPLES_FOR_SMOTE:
        return X, y, False
    k = min(5, n_minority - 1)
    from imblearn.over_sampling import SMOTE

    smt = SMOTETomek(random_state=seed, smote=SMOTE(random_state=seed, k_neighbors=k))
    X_res, y_res = smt.fit_resample(X, y)
    return X_res.astype(np.float32), y_res.astype(np.float32), True


def _patient_level_split(df, seed, test_size, val_size):
    """Assigns every PATIENT (subject_id) -- not every admission -- to exactly
    one of train/val/test, so that no patient's admissions ever straddle the
    train/test boundary. This is the fix for the patient-level leakage that an
    admission-level split allows: the cohort contains 546,028 admissions from
    only ~223,000 distinct patients (~2.4 admissions/patient), so an
    admission-level split would place different admissions of the same patient
    in both train and test.

    Patients are stratified by whether they EVER died in any of their
    admissions (max of hospital_expire_flag over the patient's rows), so the
    overall mortality prevalence is preserved across the three splits.
    Returns a Series aligned to df's index mapping each row -> {train,val,test}.
    """
    pat = df.groupby("subject_id")[TARGET_COL].max()
    pat_ids = pat.index.values
    pat_y = pat.values

    train_ids, temp_ids = train_test_split(
        pat_ids, test_size=(test_size + val_size), random_state=seed, stratify=pat_y)
    temp_y = pat.loc[temp_ids].values
    rel_test = test_size / (test_size + val_size)
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=rel_test, random_state=seed, stratify=temp_y)

    assignment = {}
    assignment.update({sid: "train" for sid in train_ids})
    assignment.update({sid: "val" for sid in val_ids})
    assignment.update({sid: "test" for sid in test_ids})
    return df["subject_id"].map(assignment)


def build_clients(df, seed=42, test_size=0.2, val_size=0.1, config="min_deaths",
                  imbalance="weighted"):
    """Builds the non-IID federated clients plus the pooled global
    train/val/test splits used by the centralized baselines.

    Split order:
      1. GLOBAL patient-level stratified split (see _patient_level_split):
         every subject_id lands in exactly one of train/val/test, so no
         patient appears in more than one split (no patient-level leakage).
      2. Each care unit's (client's) train/val/test partition is the subset
         of that unit's admissions whose patient falls in the corresponding
         global split.
      3. The shared encoder/scaler/imputer is fit on the pooled TRAIN
         admissions ONLY (never val/test), so no feature statistic crosses
         the train/test boundary either.
      4. SMOTE-Tomek (or class-weight fallback) is applied to TRAIN ONLY;
         val/test keep each client's true, natural mortality rate.

    `imbalance` selects the class-imbalance treatment, for the ablation:
      weighted     no resampling, capped class-weighted loss (the default)
      smote_tomek  SMOTE-Tomek on each client's training partition (ablation)
      none         no resampling and unweighted loss (ablation)

    Class weighting is the default because, in the 10-seed ablation, SMOTE-
    Tomek lowered AUC-PR for every strategy on both datasets (synthetic rows
    were up to 99.7% of a client's positives) and badly miscalibrated the
    predicted probabilities.

    Returns a dict keyed by care unit name with:
      X_train_res, y_train_res  (post-SMOTE, or raw+flag if SMOTE skipped)
      used_smote                (bool)
      X_val, y_val, X_test, y_test  (natural distribution, untouched)
      pos_weight                 (for class-weighted BCE loss)
    """
    if imbalance not in IMBALANCE_MODES:
        raise ValueError(imbalance)
    df = df.copy()
    df["_client"] = assign_clients(df, config, seed=seed)
    df = df[df["_client"].notna()]
    client_names = list(df["_client"].value_counts().index)  # largest first
    df["_split"] = _patient_level_split(df, seed, test_size, val_size)

    splits = {}
    for unit in client_names:
        sub = df[df["_client"] == unit]
        train_df = sub[sub["_split"] == "train"]
        val_df = sub[sub["_split"] == "val"]
        test_df = sub[sub["_split"] == "test"]
        splits[unit] = (train_df, val_df, test_df)

    pooled_train_df = pd.concat([s[0] for s in splits.values()], ignore_index=True)
    ohe, scaler, lab_medians = _fit_shared_preprocessor(pooled_train_df)

    clients = {}
    test_parts, val_parts, test_client = [], [], []

    for unit in client_names:
        train_df, val_df, test_df = splits[unit]

        X_train, y_train = _transform(train_df, ohe, scaler, lab_medians)
        X_val, y_val = _transform(val_df, ohe, scaler, lab_medians) if len(val_df) else (
            np.zeros((0, X_train.shape[1]), dtype=np.float32), np.zeros((0,), dtype=np.float32))
        X_test, y_test = _transform(test_df, ohe, scaler, lab_medians) if len(test_df) else (
            np.zeros((0, X_train.shape[1]), dtype=np.float32), np.zeros((0,), dtype=np.float32))

        if imbalance == "smote_tomek":
            X_train_res, y_train_res, used_smote = apply_smote_tomek(X_train, y_train, seed=seed)
        else:
            X_train_res, y_train_res, used_smote = X_train, y_train, False

        # The class weight is computed on the rows the client actually trains
        # on: ~1 after SMOTE-Tomek has balanced them, the raw imbalance ratio
        # otherwise (computing it on the raw rows and applying it on top of
        # SMOTE output would compensate the imbalance twice).
        # Capped at 20x: an uncapped ratio blows up to >20,000x for Labor &
        # Delivery (1 positive in 20,337 rows), which would let that one
        # client's gradient dominate/destabilize the round's aggregation.
        n_pos_train = max(int(y_train_res.sum()), 1)
        n_neg_train = max(int((1 - y_train_res).sum()), 1)
        pos_weight = min(n_neg_train / n_pos_train, 20.0) if imbalance != "none" else 1.0

        clients[unit] = dict(
            X_train_res=X_train_res, y_train_res=y_train_res, used_smote=used_smote,
            X_train_raw=X_train, y_train_raw=y_train,
            X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test,
            pos_weight=pos_weight, n_train_raw=len(train_df),
            n_pos_train_raw=int(y_train.sum()), n_train_res=len(y_train_res),
            n_pos_train_res=int(y_train_res.sum()),
        )
        val_parts.append((X_val, y_val))
        test_parts.append((X_test, y_test))
        test_client.extend([unit] * len(y_test))

    X_val_global = np.vstack([p[0] for p in val_parts])
    y_val_global = np.concatenate([p[1] for p in val_parts])
    X_test_global = np.vstack([p[0] for p in test_parts])
    y_test_global = np.concatenate([p[1] for p in test_parts])

    n_features = X_val_global.shape[1]

    return dict(
        clients=clients,
        X_val_global=X_val_global, y_val_global=y_val_global,
        X_test_global=X_test_global, y_test_global=y_test_global,
        test_client_global=np.array(test_client), n_features=n_features, config=config, imbalance=imbalance,
    )


# Built client sets are cached outside the (cloud-synced) project folder:
# SMOTE-Tomek's nearest-neighbour search over the Emergency Department's
# ~500k training rows dominates a run's start-up time, and every strategy,
# experiment and worker that shares a (seed, config, imbalance) reuses the
# same clients. The cache holds MIMIC-IV-derived data, so it never leaves
# this machine (PhysioNet DUA).
CACHE_DIR = os.environ.get("FEDMIMIC_CACHE", os.path.expanduser("~/Library/Caches/FedMIMIC"))


def build_clients_cached(df, seed=42, config="min_deaths", imbalance="weighted", dataset="mimic"):
    import pickle

    path = os.path.join(CACHE_DIR, f"{dataset}_clients_{config}_{imbalance}_seed{seed}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    bundle = build_clients(df, seed=seed, config=config, imbalance=imbalance)
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return bundle


def client_summary(bundle):
    """One row per client: raw vs post-resampling training sizes, i.e. how
    many synthetic minority rows SMOTE added and how many majority rows
    Tomek-link cleaning removed."""
    rows = []
    for unit, c in bundle["clients"].items():
        n_pos_raw, n_pos_res = c["n_pos_train_raw"], c["n_pos_train_res"]
        n_neg_raw = c["n_train_raw"] - n_pos_raw
        n_neg_res = c["n_train_res"] - n_pos_res
        rows.append(dict(
            client=unit, n_train=c["n_train_raw"], n_pos_train=n_pos_raw,
            n_val=len(c["y_val"]), n_test=len(c["y_test"]), n_pos_test=int(c["y_test"].sum()),
            used_smote=c["used_smote"], n_synthetic_pos=n_pos_res - n_pos_raw,
            n_removed_neg=n_neg_raw - n_neg_res,
            synthetic_share_of_pos=(n_pos_res - n_pos_raw) / n_pos_res if n_pos_res else 0.0,
            synthetic_share_of_train=(n_pos_res - n_pos_raw) / c["n_train_res"],
            pos_weight=c["pos_weight"],
        ))
    return pd.DataFrame(rows)


def build_pooled_centralized_splits(bundle, seed=42):
    """Pools every client's RAW (pre-SMOTE) training rows from `build_clients`
    into one centralized training set, and reuses the EXACT SAME global
    val/test arrays as the FL experiments.

    Centralized baselines are therefore evaluated on exactly the same held-out
    data as the federated strategies: X_val_global / X_test_global are the
    identical arrays passed to both.
    """
    clients = bundle["clients"]
    X_train = np.vstack([c["X_train_raw"] for c in clients.values()])
    y_train = np.concatenate([c["y_train_raw"] for c in clients.values()])
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(y_train))
    X_train, y_train = X_train[perm], y_train[perm]
    return dict(
        X_train=X_train, y_train=y_train,
        X_val=bundle["X_val_global"], y_val=bundle["y_val_global"],
        X_test=bundle["X_test_global"], y_test=bundle["y_test_global"],
    )
