from pathlib import Path

import pandas as pd

from pylgm import Experiment


def synthetic_nic_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2023-01-01", periods=42, freq="MS")
    for region_number, region_code in enumerate(("ITC1", "ITC2", "ITC3")):
        for coicop_number, coicop in enumerate(("CP01", "CP02", "CP03")):
            for month_number, date in enumerate(dates):
                rows.append(
                    {
                        "date": date,
                        "region_code": region_code,
                        "coicop": coicop,
                        "yoy_pct": (
                            1.5
                            + 0.35 * region_number
                            - 0.2 * coicop_number
                            + 0.18 * month_number
                        ),
                    }
                )
    return pd.DataFrame(rows)


def test_nic_shaped_backtest_has_no_leakage_and_beats_global_mean() -> None:
    source = synthetic_nic_frame()
    result = Experiment.from_yaml(Path("examples/nic_backtest/config.yaml")).compare(
        source, output=None
    )
    predictions = result.predictions
    actual = source.rename(columns={"date": "target_time", "yoy_pct": "expected_actual"})
    merged = predictions.merge(
        actual,
        on=["region_code", "coicop", "target_time"],
        how="left",
        validate="many_to_one",
    )

    assert result.metrics["log_predictive_density"].notna().all()
    assert "persistence" in set(predictions["candidate"])
    assert merged["target_time"].gt(merged["origin"]).all()
    assert merged["actual"].eq(merged["expected_actual"]).all()
    model_predictions = merged.loc[~merged["is_benchmark"]].copy()
    global_means = {
        origin: source.loc[source["date"].le(origin), "yoy_pct"].mean()
        for origin in model_predictions["origin"].drop_duplicates()
    }
    model_mse = model_predictions.groupby("candidate").apply(
        lambda group: ((group["actual"] - group["mean"]) ** 2).mean(),
        include_groups=False,
    )
    global_mse = (
        model_predictions["actual"] - model_predictions["origin"].map(global_means)
    ).pow(2).mean()

    assert model_mse.lt(global_mse).any()
