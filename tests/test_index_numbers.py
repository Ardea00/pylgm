import numpy as np
import pandas as pd
import pytest

from pylgm import Fixed, Gaussian, IID, LGM, LinearConstraint
from pylgm.index_numbers import align_factors, chain, overlap_factors, unchain
from pylgm.operators import aggregation_operator

GROUPS = ("a", "b", "c")
YEARS = list(range(2019, 2024))
QUARTERS = 4


def _make_oracle(seed=12345):
    rng = np.random.default_rng(seed)
    inflation = {"a": 0.03, "b": 0.08, "c": -0.02}
    base_price = {"a": 10.0, "b": 20.0, "c": 6.0}

    q = {}
    p = {}
    for g in GROUPS:
        for yi, year in enumerate(YEARS):
            p[(g, year)] = (
                base_price[g] * (1.0 + inflation[g]) ** yi * rng.uniform(0.9, 1.1, QUARTERS)
            )
            q[(g, year)] = rng.uniform(50.0, 150.0, QUARTERS)

    CP = {(g, year): p[(g, year)] * q[(g, year)] for g in GROUPS for year in YEARS}
    CP_A = {(g, year): CP[(g, year)].sum() for g in GROUPS for year in YEARS}
    Q_A = {(g, year): q[(g, year)].sum() for g in GROUPS for year in YEARS}

    PYP = {}
    for g in GROUPS:
        for year in YEARS[1:]:
            Pbar_prev = CP_A[(g, year - 1)] / Q_A[(g, year - 1)]
            PYP[(g, year)] = Pbar_prev * q[(g, year)]

    CL = {}
    CL_A = {}
    for g in GROUPS:
        CL_A[(g, YEARS[0])] = CP_A[(g, YEARS[0])]
        CL[(g, YEARS[0])] = CP[(g, YEARS[0])].copy()
        for year in YEARS[1:]:
            CL[(g, year)] = PYP[(g, year)] * CL_A[(g, year - 1)] / CP_A[(g, year - 1)]
            CL_A[(g, year)] = CL[(g, year)].sum()

    CP_total = {year: sum(CP[(g, year)] for g in GROUPS) for year in YEARS}
    CP_A_total = {year: CP_total[year].sum() for year in YEARS}
    PYP_total = {year: sum(PYP[(g, year)] for g in GROUPS) for year in YEARS[1:]}

    CL_total = {}
    CL_A_total = {}
    CL_A_total[YEARS[0]] = CP_A_total[YEARS[0]]
    CL_total[YEARS[0]] = CP_total[YEARS[0]].copy()
    for year in YEARS[1:]:
        CL_total[year] = PYP_total[year] * CL_A_total[year - 1] / CP_A_total[year - 1]
        CL_A_total[year] = CL_total[year].sum()

    return dict(
        q=q, p=p, CP=CP, CP_A=CP_A, Q_A=Q_A, PYP=PYP, CL=CL, CL_A=CL_A,
        CP_total=CP_total, CP_A_total=CP_A_total, PYP_total=PYP_total,
        CL_total=CL_total, CL_A_total=CL_A_total,
    )


def _quarterly_frame(oracle):
    rows = []
    for g in GROUPS:
        for year in YEARS:
            for k in range(QUARTERS):
                rows.append({
                    "group": g,
                    "year": year,
                    "quarter": k,
                    "quarter_label": f"{year}Q{k + 1}",
                    "row_id": f"{g}_{year}_{k}",
                    "cl": oracle["CL"][(g, year)][k],
                    "cp": oracle["CP"][(g, year)][k],
                })
    for year in YEARS:
        for k in range(QUARTERS):
            rows.append({
                "group": "total",
                "year": year,
                "quarter": k,
                "quarter_label": f"{year}Q{k + 1}",
                "row_id": f"total_{year}_{k}",
                "cl": oracle["CL_total"][year][k],
                "cp": oracle["CP_total"][year][k],
            })
    return pd.DataFrame(rows)


def _annual_totals_frame(oracle):
    rows = []
    for g in GROUPS:
        for year in YEARS:
            rows.append({
                "group": g, "year": year,
                "volume": oracle["CL_A"][(g, year)], "current": oracle["CP_A"][(g, year)],
            })
    for year in YEARS:
        rows.append({
            "group": "total", "year": year,
            "volume": oracle["CL_A_total"][year], "current": oracle["CP_A_total"][year],
        })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def bundle():
    oracle = _make_oracle()
    quarterly = _quarterly_frame(oracle)
    totals = _annual_totals_frame(oracle)
    factors = overlap_factors(totals, volume="volume", current="current", period="year", group="group")
    return oracle, quarterly, totals, factors


def test_unchain_recovers_previous_period_prices(bundle):
    oracle, quarterly, _totals, factors = bundle
    unchained = unchain(quarterly, "cl", factors, period="year", group="group")

    expected = np.full(len(quarterly), np.nan)
    for i, row in quarterly.iterrows():
        if row["year"] == YEARS[0]:
            continue
        if row["group"] == "total":
            expected[i] = oracle["PYP_total"][row["year"]][row["quarter"]]
        else:
            expected[i] = oracle["PYP"][(row["group"], row["year"])][row["quarter"]]

    year_2019 = (quarterly["year"] == YEARS[0]).to_numpy()
    assert np.isnan(unchained[year_2019]).all()
    later = ~year_2019
    np.testing.assert_allclose(unchained[later], expected[later], rtol=1e-12)


def test_previous_period_prices_are_additive(bundle):
    oracle, quarterly, _totals, factors = bundle
    unchained = unchain(quarterly, "cl", factors, period="year", group="group")
    frame = quarterly.assign(unchained=unchained)

    for year in YEARS[1:]:
        components = frame[(frame["year"] == year) & (frame["group"].isin(GROUPS))]
        component_sum = components.groupby("quarter")["unchained"].sum().sort_index().to_numpy()
        total_row = frame[(frame["year"] == year) & (frame["group"] == "total")].sort_values("quarter")
        np.testing.assert_allclose(component_sum, total_row["unchained"].to_numpy(), rtol=1e-12)

    max_relative_gap = 0.0
    for year in YEARS:
        component_cl = frame[(frame["year"] == year) & (frame["group"].isin(GROUPS))]
        component_cl_sum = component_cl.groupby("quarter")["cl"].sum().sort_index().to_numpy()
        total_cl = frame[(frame["year"] == year) & (frame["group"] == "total")].sort_values("quarter")["cl"].to_numpy()
        gap = np.max(np.abs(component_cl_sum - total_cl) / np.abs(total_cl))
        max_relative_gap = max(max_relative_gap, gap)
    assert max_relative_gap > 1e-6


def test_period_totals_of_chain_linked_values_match(bundle):
    oracle, quarterly, _totals, _factors = bundle
    for g in GROUPS:
        for year in YEARS:
            subset = quarterly[(quarterly["group"] == g) & (quarterly["year"] == year)]
            np.testing.assert_allclose(subset["cl"].sum(), oracle["CL_A"][(g, year)], rtol=1e-12)
    for year in YEARS:
        subset = quarterly[(quarterly["group"] == "total") & (quarterly["year"] == year)]
        np.testing.assert_allclose(subset["cl"].sum(), oracle["CL_A_total"][year], rtol=1e-12)


def test_chain_inverts_unchain(bundle):
    _oracle, quarterly, _totals, factors = bundle
    values = quarterly["cl"].to_numpy()
    unchained = unchain(quarterly, values, factors, period="year", group="group")
    rechained = chain(quarterly, unchained, factors, period="year", group="group")

    has_factor = ~np.isnan(align_factors(quarterly, factors, period="year", group="group"))
    np.testing.assert_allclose(rechained[has_factor], values[has_factor], rtol=1e-12)
    assert np.isnan(rechained[~has_factor]).all()

    scales = np.array([1.0, 2.0, 0.5, -3.0, 10.0]).reshape(-1, 1)
    stacked = values.reshape(1, -1) * scales
    unchained_2d = unchain(quarterly, stacked, factors, period="year", group="group")
    rechained_2d = chain(quarterly, unchained_2d, factors, period="year", group="group")
    np.testing.assert_allclose(
        rechained_2d[:, has_factor], stacked[:, has_factor], rtol=1e-12
    )


def test_unpublished_totals_and_carry_forward(bundle):
    _oracle, _quarterly, totals, _factors = bundle
    gapped = totals.copy()
    gapped.loc[gapped["year"].isin([2022, 2023]), ["volume", "current"]] = np.nan

    factors_no_through = overlap_factors(
        gapped, volume="volume", current="current", period="year", group="group"
    )
    assert set(factors_no_through["year"].unique()) == {2020, 2021, 2022}
    assert not factors_no_through["carried"].any()

    factors_with_through = overlap_factors(
        gapped, volume="volume", current="current", period="year", group="group", through=2023
    )
    for g in (*GROUPS, "total"):
        group_rows = factors_with_through[factors_with_through["group"] == g]
        assert set(group_rows["year"]) == {2020, 2021, 2022, 2023}
        row_2022 = group_rows[group_rows["year"] == 2022].iloc[0]
        row_2023 = group_rows[group_rows["year"] == 2023].iloc[0]
        assert row_2022["carried"] == False  # noqa: E712
        assert row_2023["carried"] == True  # noqa: E712
        assert row_2023["factor"] == pytest.approx(row_2022["factor"])
    carried_rows = factors_with_through[factors_with_through["carried"]]
    assert set(carried_rows["year"].unique()) == {2023}


def test_constraint_with_overlap_weights_holds(bundle):
    _oracle, quarterly, _totals, factors = bundle
    frame = quarterly[
        quarterly["year"].isin([2021, 2022]) & quarterly["group"].isin(GROUPS)
    ].reset_index(drop=True)

    y = np.log(frame["cl"].to_numpy())
    unobserved = (frame["year"] == 2022) & (frame["group"] == "c")
    y[unobserved.to_numpy()] = np.nan
    frame = frame.assign(y=y)

    weight_group = align_factors(frame, factors, period="year", group="group")
    total_frame = frame[["year"]].assign(group="total")
    weight_total = align_factors(total_frame, factors, period="year", group="group")
    weights = weight_group / weight_total

    mask = (frame["year"] == 2022).to_numpy()
    operator, keys = aggregation_operator(frame, "quarter_label", weights=weights, rows=mask)

    total_2022 = quarterly[
        (quarterly["group"] == "total") & (quarterly["year"] == 2022)
    ].sort_values("quarter_label")
    assert list(keys["quarter_label"]) == list(total_2022["quarter_label"])
    rhs = total_2022["cl"].to_numpy()

    constraint = LinearConstraint(operator, rhs, scale="log")
    model = LGM(
        response="y",
        likelihood=Gaussian(0.05),
        predictor=Fixed("1") + IID("u", index="row_id", precision=1.0),
        panel=("row_id",),
    )
    result = model.fit(frame, constraints=[constraint], engine="exact_gaussian")

    np.testing.assert_allclose(operator @ np.exp(result.predictive_mean), rhs, rtol=1e-8)


def test_overlap_factors_duplicate_keys_raise():
    totals = pd.DataFrame({
        "group": ["a", "a"], "year": [2020, 2020], "volume": [1.0, 1.0], "current": [1.0, 1.0],
    })
    with pytest.raises(ValueError, match="duplicate"):
        overlap_factors(totals, volume="volume", current="current", period="year", group="group")


def test_overlap_factors_string_period_raises_type_error():
    totals = pd.DataFrame({
        "group": ["a"], "year": ["2020"], "volume": [1.0], "current": [1.0],
    })
    with pytest.raises(TypeError, match=r"period values must support \+ 1"):
        overlap_factors(totals, volume="volume", current="current", period="year", group="group")


def test_align_factors_duplicate_keys_raise():
    frame = pd.DataFrame({"group": ["a"], "year": [2020]})
    factors = pd.DataFrame({
        "group": ["a", "a"], "year": [2020, 2020], "factor": [1.0, 1.0], "carried": [False, False],
    })
    with pytest.raises(ValueError, match="duplicate"):
        align_factors(frame, factors, period="year", group="group")


def test_values_length_mismatch_raises(bundle):
    _oracle, quarterly, _totals, factors = bundle
    with pytest.raises(ValueError):
        unchain(quarterly, np.zeros(3), factors, period="year", group="group")
    with pytest.raises(ValueError):
        chain(quarterly, np.zeros(3), factors, period="year", group="group")
