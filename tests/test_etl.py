import numpy as np
import pandas as pd
import pytest

from src.etl import clean_dataframe, build_pivot, MONTH_ORDER


# Csv loading tests (mocking file system)
def test_load_raw_csvs_combines_files(tmp_path):
    """load_raw_csvs reads CSVs, strips BOM, renames columns, drops junk."""
    csv_a = tmp_path / "AguaEmbalsada_RioLozoya_ElAtazar.csv"
    csv_b = tmp_path / "AguaEmbalsada_RioManzanares_Santillana.csv"

    content = "anio;mes;hec_cub;;;\n1998;enero;100,0;;;\n;febrero;110,0;;;\n"
    csv_a.write_text(content, encoding="utf-8-sig")
    csv_b.write_text(content, encoding="utf-8-sig")

    from src.etl import load_raw_csvs
    df = load_raw_csvs(tmp_path)

    assert set(df.columns) == {"year", "month", "capacity_hm3", "reservoir"}
    assert set(df["reservoir"].unique()) == {"RioLozoya_ElAtazar", "RioManzanares_Santillana"}
    assert df["year"].iloc[0] == 1998
    assert len(df) == 4

def test_load_raw_csvs_raises_if_empty_dir(tmp_path):
    from src.etl import load_raw_csvs
    with pytest.raises(FileNotFoundError):
        load_raw_csvs(tmp_path)


# Fixtures
@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Minimal synthetic raw DataFrame mimicking load_raw_csvs output."""
    return pd.DataFrame({
        "year": [1998, np.nan, np.nan, 1999, np.nan, np.nan,
                    1998, np.nan, np.nan, 1999, np.nan, np.nan],
        "month": ["enero", "febrero", "marzo", "enero", "febrero", "marzo",
                    "enero", "febrero", "marzo", "enero", "febrero", "marzo"],
        "capacity_hm3": ["100,5", "110,2", "120,0", "130,0", "140,0", "150,0",
                            "200,0", "210,0", "220,0", "230,0", "240,0", "250,0"],
        "reservoir": ["ReservoirA"] * 6 + ["ReservoirB"] * 6,
    })


@pytest.fixture
def clean_df(raw_df) -> pd.DataFrame:
    return clean_dataframe(raw_df)


# clean_dataframe

class TestCleanDataframe:

    def test_year_forward_filled_per_reservoir(self, raw_df):
        out = clean_dataframe(raw_df)
        assert out["year"].isna().sum() == 0

    def test_capacity_converted_to_float(self, raw_df):
        out = clean_dataframe(raw_df)
        assert out["capacity_hm3"].dtype in [float, np.float64]

    def test_capacity_comma_replaced(self, raw_df):
        out = clean_dataframe(raw_df)
        assert out["capacity_hm3"].isna().sum() == 0

    def test_excluded_reservoir_dropped(self):
        df = pd.DataFrame({
            "year":         [1998, 1998],
            "month":        ["enero", "enero"],
            "capacity_hm3": ["100,0", "200,0"],
            "reservoir":    ["RioLosMorales_LosMorales", "ReservoirA"],
        })
        out = clean_dataframe(df)
        assert "RioLosMorales_LosMorales" not in out["reservoir"].values

    def test_truncated_to_march_2021(self):
        df = pd.DataFrame({
            "year":         [2021, 2021, 2021],
            "month":        ["marzo", "abril", "mayo"],
            "capacity_hm3": ["100,0", "200,0", "300,0"],
            "reservoir":    ["ReservoirA"] * 3,
        })
        out = clean_dataframe(df)
        assert "abril" not in out["month"].values
        assert "mayo"  not in out["month"].values
        assert "marzo" in out["month"].values

    def test_does_not_mutate_input(self, raw_df):
        original_len = len(raw_df)
        clean_dataframe(raw_df)
        assert len(raw_df) == original_len

    def test_riosequillo_year_reconstructed(self):
        """RioLozoya_Riosequillo with all years missing gets reconstructed."""
        rows = []
        for i, month in enumerate(MONTH_ORDER):
            for year_idx in range(24):
                rows.append({
                    "year":         np.nan,
                    "month":        month,
                    "capacity_hm3": f"{100 + i},0",
                    "reservoir":    "RioLozoya_Riosequillo",
                })
        df = pd.DataFrame(rows)
        out = clean_dataframe(df)
        assert out["year"].isna().sum() == 0
        assert set(out["year"].unique()).issubset(set(range(1998, 2022)))


# build_pivot

class TestBuildPivot:

    def test_returns_datetime_index(self, clean_df):
        pivot = build_pivot(clean_df)
        assert pd.api.types.is_datetime64_any_dtype(pivot["ds"])

    def test_ds_monotonically_increasing(self, clean_df):
        pivot = build_pivot(clean_df)
        assert pivot["ds"].is_monotonic_increasing

    def test_total_hm3_no_nulls(self, clean_df):
        pivot = build_pivot(clean_df)
        assert pivot["total_hm3"].isna().sum() == 0

    def test_total_hm3_is_sum_of_reservoirs(self, clean_df):
        pivot = build_pivot(clean_df)
        reservoir_cols = [c for c in pivot.columns
                            if c not in ["ds", "year", "month", "total_hm3"]]
        expected = pivot[reservoir_cols].sum(axis=1, skipna=True)
        pd.testing.assert_series_equal(
            pivot["total_hm3"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_one_row_per_month(self, clean_df):
        pivot = build_pivot(clean_df)
        assert pivot["ds"].nunique() == len(pivot)



