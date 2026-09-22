"""Training-only temporal permutation selection and saved-run contracts."""
import copy
from sklearn.model_selection import train_test_split


def release_inputs(data):
    if hasattr(data["X"], "_mmap") and not data["X"]._mmap.closed:
        data["X"]._mmap.close()
    if "temporary_directory" in data:
        data["temporary_directory"].cleanup()


def json_blob(value):
    return canonical_json(value).encode()


def top10_lift(y, scores, metadata):
    y, scores = _validate(y, scores)
    if not 0 < y.sum() < len(y):
        raise ValueError("Lift requires both classes.")
    frame = metadata[["PATIENT_ID", "END_DT"]].reset_index(drop=True).copy()
    if len(frame) != len(y):
        raise ValueError("Unaligned ranking metadata.")
    frame["score"], frame["position"] = scores, np.arange(len(y))
    order = frame.sort_values(["score", "PATIENT_ID", "END_DT"], ascending=[False, True, True]).position.to_numpy()
    k = max(1, int(np.ceil(.1 * len(y))))
    return float(y[order[:k]].mean() / y.mean())


def make_internal_plan(metadata, seed=42):
    """60/20/20 split of TRAIN patients; stopping and ranking patients are separate."""
    train = metadata.loc[metadata.SPLIT.eq("train")]
    patients = train.groupby("PATIENT_ID", sort=True).RESP.max()
    fit_ids, other_ids = train_test_split(patients.index.to_numpy(), test_size=.4,
        stratify=patients.to_numpy(), random_state=seed)
    stop_ids, rank_ids = train_test_split(other_ids, test_size=.5,
        stratify=patients.loc[other_ids].to_numpy(), random_state=seed + 1)
    assignments = {**dict.fromkeys(fit_ids, "fit"), **dict.fromkeys(stop_ids, "stop"),
                   **dict.fromkeys(rank_ids, "rank")}
    plan = train[["PATIENT_ID", "END_DT", "RESP"]].copy()
    plan["ROLE"] = plan.PATIENT_ID.map(assignments)
    # One fixed latest snapshot per ranking patient. Choice does not use labels or scores.
    latest = plan.loc[plan.ROLE.eq("rank")].sort_values(["PATIENT_ID", "END_DT"]).groupby("PATIENT_ID").tail(1)
    counts = latest.groupby("END_DT").PATIENT_ID.transform("size")
    eligible = latest.loc[counts >= 2]
    plan["RANK_EVALUATE"] = plan.index.isin(eligible.index)
    for role in ("fit", "stop", "rank"):
        part = plan.loc[plan.ROLE.eq(role)]
        if set(part.RESP) != {0, 1}:
            raise ValueError("Internal partitions require both classes; inspect TRAIN population.")
    if len(eligible) < 20 or set(eligible.RESP) != {0, 1}:
        raise ValueError("Too few date-matched ranking snapshots with both classes.")
    if len(eligible) < .8 * len(latest):
        raise ValueError("Less than 80% of ranking patients have same-date donors; review ranking design.")
    plan = plan.reset_index(drop=True)
    return plan


def validate_plan(metadata, plan):
    expected = metadata.loc[metadata.SPLIT.eq("train"), ["PATIENT_ID", "END_DT", "RESP"]].reset_index(drop=True)
    if not expected.equals(plan[["PATIENT_ID", "END_DT", "RESP"]].reset_index(drop=True)):
        raise ValueError("Internal plan must contain precisely the original TRAIN snapshots.")
    if set(plan.ROLE) != {"fit", "stop", "rank"} or plan.groupby("PATIENT_ID").ROLE.nunique().max() != 1:
        raise ValueError("Internal roles overlap patients or are invalid.")
    if not plan.RANK_EVALUATE.isin([True, False]).all():
        raise ValueError("Invalid ranking mask.")
    chosen = plan.loc[plan.RANK_EVALUATE]
    if not chosen.ROLE.eq("rank").all() or chosen.PATIENT_ID.duplicated().any():
        raise ValueError("Ranking requires one snapshot per separate ranking patient.")
    if chosen.groupby("END_DT").size().min() < 2:
        raise ValueError("Ranking snapshot has no same-date donor.")
    return plan


def plan_rows(data, plan, role):
    validate_plan(data["metadata"], plan)
    mask = plan.RANK_EVALUATE.to_numpy() if role == "rank" else plan.ROLE.eq(role).to_numpy()
    return np.asarray(data["indices"]["train"])[mask]


class SelectedTensor:
    """Lazy feature slicing, retaining every monthly timestep without a full tensor copy."""
    def __init__(self, source, columns):
        self.source, self.columns = source, np.asarray(columns, dtype=int)
        self.shape = (source.shape[0], source.shape[1], len(columns))

    def __getitem__(self, row):
        if not isinstance(row, (int, np.integer)):
            raise TypeError("Selected tensor uses bounded single-snapshot reads.")
        return self.source[int(row)][:, self.columns]


def selected_view(data, columns, fit=None, stop=None):
    if (not columns or len(set(columns)) != len(columns)
            or any(type(i) is not int or not 0 <= i < len(data["features"]) for i in columns)):
        raise ValueError("Invalid selected feature indices.")
    view = dict(data, X=SelectedTensor(data["X"], columns), features=[data["features"][i] for i in columns])
    if fit is not None:
        view["indices"] = {"train": np.asarray(fit), "validation": np.asarray(stop)}
    return view


def same_date_donors(metadata, seed):
    """A patient derangement within each exact cutoff date; no month shuffling."""
    if metadata.PATIENT_ID.duplicated().any():
        raise ValueError("Use one ranking snapshot per patient.")
    rng = np.random.default_rng(seed)
    donors = np.arange(len(metadata))
    for _, positions in metadata.reset_index(drop=True).groupby("END_DT", sort=True).indices.items():
        if len(positions) < 2:
            raise ValueError("Each cutoff requires at least two patients.")
        order = rng.permutation(positions)
        donors[order] = np.roll(order, 1)
    return donors


def score_rows(model, data, rows, device, batch_size=256, feature=None, donors=None):
    model.eval()
    scores = []
    if feature is not None and (donors is None or len(donors) != len(rows)):
        raise ValueError("Permutation needs one donor per row.")
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            subset = rows[start:start + batch_size]
            values = np.stack([np.asarray(data["X"][int(i)], dtype=np.float32) for i in subset])
            if feature is not None:
                source_rows = rows[donors[start:start + batch_size]]
                values[:, :, feature] = np.stack([data["X"][int(i)][:, feature] for i in source_rows])
            p = torch.sigmoid(model(checked_log1p(torch.as_tensor(values, device=device)))).cpu().numpy()
            scores.append(p)
    scores = np.concatenate(scores)
    _validate(data["y"][rows], scores)
    return scores


def rank_feature(model, data, rows, device, index, repeats, seed, baseline):
    meta, y = data["metadata"].iloc[rows].reset_index(drop=True), data["y"][rows]
    lift_drops, ap_drops = [], []
    for repeat in range(repeats):
        # Common permutations across features reduce unnecessary comparison noise.
        donors = same_date_donors(meta, seed + repeat)
        scores = score_rows(model, data, rows, device, feature=index, donors=donors)
        lift_drops.append(baseline["lift"] - top10_lift(y, scores, meta))
        ap_drops.append(baseline["ap"] - float(average_precision_score(y, scores)))
    return {"feature_index": int(index), "feature_name": data["features"][index],
            "mean_lift_drop": float(np.mean(lift_drops)),
            "std_lift_drop": float(np.std(lift_drops, ddof=1)) if repeats > 1 else 0.,
            "mean_ap_drop": float(np.mean(ap_drops)), "lift_drops": lift_drops}


def selection_from_ranking(records, features, top_k):
    if type(top_k) is not int or not 0 < top_k < len(features):
        raise ValueError("TOP_K must be positive and smaller than the full vocabulary.")
    table = pd.DataFrame(records)
    if (len(table) != len(features) or table.feature_index.duplicated().any()
            or sorted(table.feature_index) != list(range(len(features)))):
        raise ValueError("Ranking is incomplete or has duplicate features.")
    ordered = table.sort_values("feature_index")
    if ordered.feature_name.tolist() != features:
        raise ValueError("Ranking feature names disagree with original vocabulary.")
    if not np.isfinite(table[["mean_lift_drop", "std_lift_drop", "mean_ap_drop"]].to_numpy()).all():
        raise ValueError("Invalid importance values.")
    table = table.sort_values(["mean_lift_drop", "mean_ap_drop", "feature_index"], ascending=[False, False, True]).reset_index(drop=True)
    table["rank"] = np.arange(1, len(table) + 1)
    table["selected"] = table["rank"] <= top_k
    # Model input remains in the original vocabulary order, not importance order.
    columns = sorted(int(i) for i in table.loc[table.selected, "feature_index"])
    return table, columns


def verify_audits(data, preparation, split_audit):
    compare_preparation_to_split(preparation, split_audit)
    if data["hashes"] != preparation["reference_input_hashes"]:
        raise ValueError("Tensor, labels, features or split differs from original RUN_001.")


def model_artifacts(data, config, settings, run_id, contract):
    blob, summary, history = train_run(data, config, settings, run_id)
    summary["contract"] = contract
    return {"checkpoint.pt": blob, "summary.json": json_blob(summary),
            "history.csv": history.to_csv(index=False).encode()}


def checked_model(blobs, data, run_id, contract):
    summary = json.loads(blobs["summary.json"])
    if summary.get("contract") != contract:
        raise ValueError("Saved model uses different code/settings/inputs; choose a new RUN_ID.")
    model, payload, device = load_verified_model(blobs["checkpoint.pt"], data, run_id)
    if payload["model_config"] != summary["model_config"] or payload["training_settings"] != summary["training_settings"]:
        raise ValueError("Checkpoint and summary disagree.")
    return model, payload, device


def plot_history(history, title):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    history.plot(x="epoch", y=["training_loss", "validation_loss"], ax=axes[0], title=title + " loss (eval mode)")
    history.plot(x="epoch", y=["training_top10_lift", "validation_top10_lift"], ax=axes[1], title=title + " top-10% lift")
    fig.tight_layout()
    return fig
