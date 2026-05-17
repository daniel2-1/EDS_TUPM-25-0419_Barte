# =============================================================================
# FINAL PROJECT: Engineering Data Systems Pipeline
# Topic    : HVA-01 — Chiller Plant COP (Coefficient of Performance) Variance
# Pillar   : PILLAR 9 — HVAC & Building Systems
# Dataset  : Chiller Energy Data — HVAC System Energy Data with Weather Data
# Kaggle   : https://www.kaggle.com/datasets/chillerenergy/chiller-energy-data
# Author   : Daniel Barte | Student No: TUPM-25-0419
# Course   : BSME 1B
# Unique Filter: Month == 7 (July peak-cooling season)
# =============================================================================
#
# FORMULA: COP = Cooling Load (kW) / Chiller Power Input (kW)
#   COP > 4.0  → High efficiency
#   COP 2.5–4  → Normal operation
#   COP < 2.5  → Degraded / fault state
#
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR   = "data"
OUTPUT_DIR = "outputs"
ORIG_CSV   = os.path.join(DATA_DIR, "dataset_original.csv")
CLEAN_CSV  = os.path.join(DATA_DIR, "dataset_cleaned.csv")

TARGET_COL = "COP"   # Engineered from Cooling_Load / Chiller_Power


def resolve_dataset_path() -> str:
    """Return the correct raw dataset path, with fallback detection for common naming mistakes."""
    if os.path.exists(ORIG_CSV):
        return ORIG_CSV

    alt_names = [
        "dataset_original.csv.csv",
        "dataset_original (1).csv",
        "dataset_original - Copy.csv",
        "dataset_original copy.csv",
    ]
    for name in alt_names:
        alt_path = os.path.join(DATA_DIR, name)
        if os.path.exists(alt_path):
            return alt_path

    csv_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")]
    if len(csv_files) == 1:
        return os.path.join(DATA_DIR, csv_files[0])

    raise FileNotFoundError(
        f"\n[ERROR] Dataset not found at '{ORIG_CSV}'.\n"
        f"Existing CSV files in {DATA_DIR}: {csv_files}\n\n"
        "Download steps:\n"
        "  1. Go to: https://www.kaggle.com/datasets/chillerenergy/chiller-energy-data\n"
        "  2. Click 'Download' (requires free Kaggle account)\n"
        "  3. Extract ZIP, rename CSV to: dataset_original.csv\n"
        f"  4. Place in: {DATA_DIR}/\n"
    )

# Unique Filter — Month 7 (July): peak-cooling season slice
UNIQUE_FILTER_COL   = "Month"
UNIQUE_FILTER_VALUE = 7

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# HELPER — Auto-detect column names (case-insensitive)
# =============================================================================
def _find_col(df: pd.DataFrame, candidates: list):
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        key = c.lower()
        if key in lower_map:
            return lower_map[key]
    for c in candidates:
        key = c.lower()
        for lower_col, original_col in lower_map.items():
            if key in lower_col:
                return original_col
    return None


# =============================================================================
# MODULE 1 — DATA INGESTION
# =============================================================================
class DataIngestion:
    """Loads the raw CSV dataset and performs initial inspection."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df_raw   = None

    def load(self) -> pd.DataFrame:
        try:
            self.df_raw = pd.read_csv(self.filepath)
            print(f"[INGESTION] Loaded {len(self.df_raw):,} rows x {self.df_raw.shape[1]} cols")
            print(f"[INGESTION] Columns: {list(self.df_raw.columns)}")
            return self.df_raw
        except FileNotFoundError:
            raise FileNotFoundError(
                f"\n[ERROR] Dataset not found at '{self.filepath}'.\n\n"
                "Download steps:\n"
                "  1. Go to: https://www.kaggle.com/datasets/chillerenergy/chiller-energy-data\n"
                "  2. Click 'Download' (requires free Kaggle account)\n"
                "  3. Extract ZIP, rename CSV to: dataset_original.csv\n"
                f"  4. Place in: {DATA_DIR}/\n"
            )
        except Exception as e:
            raise RuntimeError(f"[INGESTION ERROR] {e}")

    def summary(self):
        if self.df_raw is None:
            return
        print(f"\n  Rows: {self.df_raw.shape[0]:,}  Cols: {self.df_raw.shape[1]}")
        print("\n-- Data Types --")
        print(self.df_raw.dtypes)
        print("\n-- Missing Values --")
        print(self.df_raw.isnull().sum())
        print("\n-- First 3 Rows --")
        print(self.df_raw.head(3))


# =============================================================================
# MODULE 2 — DATA CLEANING
# =============================================================================
class DataCleaning:
    """
    Cleans raw data: duplicates, nulls, type fixes, COP engineering,
    unique filter, and IQR outlier removal.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def remove_duplicates(self) -> "DataCleaning":
        before = len(self.df)
        self.df.drop_duplicates(inplace=True)
        print(f"[CLEANING] Duplicates removed: {before - len(self.df)}")
        return self

    def handle_missing(self) -> "DataCleaning":
        total = self.df.isnull().sum().sum()
        if total == 0:
            print("[CLEANING] No missing values detected.")
            return self
        for col in self.df.select_dtypes(include=[np.number]).columns:
            if self.df[col].isnull().any():
                med = self.df[col].median()
                self.df[col].fillna(med, inplace=True)
                print(f"[CLEANING] Filled '{col}' nulls with median={med:.4f}")
        self.df.dropna(inplace=True)
        print(f"[CLEANING] Rows after null handling: {len(self.df):,}")
        return self

    def fix_dtypes(self) -> "DataCleaning":
        ts_col = _find_col(self.df, ["Timestamp", "datetime", "Date", "time", "local time"])
        if ts_col:
            try:
                self.df[ts_col] = pd.to_datetime(self.df[ts_col], errors="coerce")
                self.df.dropna(subset=[ts_col], inplace=True)
                self.df["Hour"]      = self.df[ts_col].dt.hour
                self.df["Month"]     = self.df[ts_col].dt.month
                self.df["DayOfWeek"] = self.df[ts_col].dt.dayofweek
                self.df["Season"]    = self.df["Month"].apply(
                    lambda m: 0 if m in [12,1,2] else 1 if m in [3,4,5]
                              else 2 if m in [6,7,8] else 3
                )
                print(f"[CLEANING] Parsed '{ts_col}' -> Hour, Month, DayOfWeek, Season")
            except Exception as e:
                print(f"[CLEANING WARNING] Timestamp parse: {e}")

        for col in self.df.select_dtypes(include=["object", "bool"]).columns:
            if ts_col and col == ts_col:
                continue
            try:
                self.df[col] = LabelEncoder().fit_transform(self.df[col].astype(str))
                print(f"[CLEANING] Encoded: '{col}'")
            except Exception as e:
                print(f"[CLEANING WARNING] Could not encode '{col}': {e}")
        return self

    def engineer_cop(self) -> "DataCleaning":
        """
        COP = Cooling Load (kW) / Chiller Power Input (kW).
        Auto-detects column names. If COP already exists, uses it directly.
        """
        cop_col   = _find_col(self.df, ["COP", "cop", "Cop"])
        power_col = _find_col(self.df, [
            "Chiller_Power_kW", "Power_kW", "chiller_power",
            "Power", "ElectricPower", "Input_Power_kW", "power",
            "Chiller Energy Consumption", "Energy Consumption", "Energy", "Consumption"
        ])
        load_col  = _find_col(self.df, [
            "Cooling_Load_kW", "CoolingLoad", "cooling_load",
            "Load_kW", "CoolingCapacity", "Load", "load",
            "Building Load", "Building Load (RT)", "Load (RT)", "RT"
        ])

        if cop_col:
            self.df.rename(columns={cop_col: "COP"}, inplace=True)
            print(f"[CLEANING] COP column '{cop_col}' found and renamed to 'COP'.")
            return self

        if power_col and load_col:
            self.df = self.df.copy()
            if "kwh" in power_col.lower() or "energy" in power_col.lower():
                if "Month" in self.df.columns and self.df[power_col].dtype.kind in "fi":
                    try:
                        self.df.sort_values(_find_col(self.df, ["Timestamp", "datetime", "Date", "time", "local time"]), inplace=True)
                    except Exception:
                        pass
                    if "Month" in self.df.columns:
                        self.df["IntervalHours"] = self.df[_find_col(self.df, ["Timestamp", "datetime", "Date", "time", "local time"])].diff().dt.total_seconds().div(3600)
                        interval = float(self.df["IntervalHours"].median(skipna=True))
                        if interval > 0:
                            self.df["Power_kW"] = self.df[power_col] / interval
                        else:
                            self.df["Power_kW"] = self.df[power_col]
                    else:
                        self.df["Power_kW"] = self.df[power_col]
                else:
                    self.df["Power_kW"] = self.df[power_col]
            else:
                self.df["Power_kW"] = self.df[power_col]

            if "rt" in load_col.lower():
                self.df["Load_kW"] = self.df[load_col] * 3.516852842394497
            else:
                self.df["Load_kW"] = self.df[load_col]

            self.df = self.df[self.df["Power_kW"] > 0].copy()
            self.df["COP"] = self.df["Load_kW"] / self.df["Power_kW"]
            if not self.df.empty:
                print(f"[CLEANING] Engineered COP = {load_col} / {power_col}")
                print(f"[CLEANING] COP range: {self.df['COP'].min():.2f} to {self.df['COP'].max():.2f}")
            return self

        print(f"[CLEANING WARNING] Power/Load columns not found.")
        print(f"[CLEANING] Available: {list(self.df.columns)}")
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) >= 2:
            self.df = self.df[self.df[num_cols[0]] > 0].copy()
            self.df["COP"] = self.df[num_cols[1]] / self.df[num_cols[0]]
            print(f"[CLEANING] Fallback COP = {num_cols[1]} / {num_cols[0]}")
        else:
            raise ValueError("Cannot engineer COP: insufficient numeric columns.")
        return self

    def apply_unique_filter(self, col: str, value) -> "DataCleaning":
        """Filter by the requested column value, or fallback to summer if the exact value is unavailable."""
        before = len(self.df)
        if col in self.df.columns:
            if value in self.df[col].unique():
                self.df = self.df[self.df[col] == value].copy()
                print(f"[CLEANING] Unique filter: {col}=={value} -> {len(self.df):,} rows (was {before:,})")
            elif set(self.df[col].unique()) & {6, 7, 8}:
                self.df = self.df[self.df[col].isin([6, 7, 8])].copy()
                print(f"[CLEANING] Month {value} missing; applied summer filter 6-8 -> {len(self.df):,} rows (was {before:,})")
            else:
                print(f"[CLEANING WARNING] Filter value {value} not found in '{col}'. Skipping filter.")
        else:
            print(f"[CLEANING WARNING] Filter col '{col}' not found. Skipping.")
        return self

    def remove_outliers_iqr(self, col: str = "COP") -> "DataCleaning":
        if col not in self.df.columns or self.df.empty:
            return self
        Q1, Q3 = self.df[col].quantile(0.25), self.df[col].quantile(0.75)
        IQR = Q3 - Q1
        before = len(self.df)
        self.df = self.df[(self.df[col] >= Q1 - 1.5*IQR) & (self.df[col] <= Q3 + 1.5*IQR)].copy()
        print(f"[CLEANING] IQR outlier removal '{col}': {before - len(self.df)} rows removed.")
        before2 = len(self.df)
        self.df = self.df[(self.df[col] >= 0.5) & (self.df[col] <= 8.0)].copy()
        print(f"[CLEANING] Physical COP bounds [0.5-8.0]: {before2 - len(self.df)} rows removed.")
        return self

    def get_clean_df(self) -> pd.DataFrame:
        self.df.reset_index(drop=True, inplace=True)
        return self.df


# =============================================================================
# MODULE 3 — STATISTICAL ANALYSIS
# =============================================================================
class StatisticalAnalysis:
    """
    Computes all required statistics using NumPy (mandatory per rubric).
    """

    def __init__(self, df: pd.DataFrame, target: str = "COP"):
        self.df       = df
        self.target   = target
        self.num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    def descriptive_stats(self) -> pd.DataFrame:
        """Mean, Median, Std, Variance, Min, Max, Range — computed with NumPy."""
        results = {}
        for col in self.num_cols:
            arr = self.df[col].dropna().values.astype(float)
            results[col] = {
                "Mean":     np.mean(arr),
                "Median":   np.median(arr),
                "Std Dev":  np.std(arr, ddof=1),
                "Variance": np.var(arr, ddof=1),
                "Min":      np.min(arr),
                "Max":      np.max(arr),
                "Range":    np.max(arr) - np.min(arr),
            }
        df_stats = pd.DataFrame(results).T.round(4)
        print("\n-- Descriptive Statistics (NumPy) --")
        print(df_stats.to_string())
        return df_stats

    def distribution_analysis(self) -> pd.DataFrame:
        """Skewness, kurtosis, IQR, outlier count per column."""
        report = {}
        for col in self.num_cols:
            arr  = self.df[col].dropna().values.astype(float)
            Q1, Q3 = np.percentile(arr, 25), np.percentile(arr, 75)
            IQR = Q3 - Q1
            report[col] = {
                "Skewness":   round(stats.skew(arr), 4),
                "Kurtosis":   round(stats.kurtosis(arr), 4),
                "IQR":        round(IQR, 4),
                "Outliers_n": int(np.sum((arr < Q1-1.5*IQR) | (arr > Q3+1.5*IQR))),
            }
        dist_df = pd.DataFrame(report).T
        print("\n-- Distribution Analysis --")
        print(dist_df.to_string())

        if self.target in report:
            s = report[self.target]["Skewness"]
            if abs(s) < 0.5:
                interp = "approximately symmetric -> consistent operation"
            elif s > 0.5:
                interp = "positive skew -> frequent low-COP events; chiller stress periods"
            else:
                interp = "negative skew -> mostly efficient; rare degradation events"
            print(f"\n[ANALYSIS] COP skewness={s:.4f} -> {interp}")
        return dist_df

    def correlation_analysis(self) -> pd.DataFrame:
        """Pearson correlation matrix; top drivers of COP."""
        corr = self.df[self.num_cols].corr()
        if self.target in corr.columns:
            top = corr[self.target].drop(self.target).abs().sort_values(ascending=False)
            print(f"\n-- Top Correlations with '{self.target}' --")
            print(top.head(10).to_string())
        return corr

    def comparative_analysis(self) -> dict:
        """Compare COP: Peak Hours (9-18h) vs Off-Peak. Welch T-test."""
        results = {}
        if "Hour" not in self.df.columns or self.target not in self.df.columns:
            print("[ANALYSIS] Skipping: Hour or COP column missing.")
            return results

        peak     = self.df[self.df["Hour"].between(9, 18)][self.target].dropna().values
        off_peak = self.df[~self.df["Hour"].between(9, 18)][self.target].dropna().values

        for label, arr in [("Peak (9-18h)", peak), ("Off-Peak", off_peak)]:
            results[label] = {
                "n":    len(arr),
                "Mean": round(float(np.mean(arr)), 4),
                "Std":  round(float(np.std(arr, ddof=1)), 4),
            }
        print("\n-- Comparative: Peak vs Off-Peak COP --")
        print(pd.DataFrame(results).T.to_string())

        if len(peak) > 1 and len(off_peak) > 1:
            t_stat, p_val = stats.ttest_ind(peak, off_peak, equal_var=False)
            sig = "SIGNIFICANT" if p_val < 0.05 else "NOT SIGNIFICANT"
            results["t_stat"] = round(float(t_stat), 4)
            results["p_value"] = round(float(p_val), 6)
            print(f"\n[STATS] Welch T-test: t={t_stat:.4f}, p={p_val:.6f} -> {sig} (alpha=0.05)")
        return results


# =============================================================================
# MODULE 4 — MACHINE LEARNING
# =============================================================================
class MachineLearning:
    """
    Trains Linear Regression, Random Forest, and Gradient Boosting
    to predict COP. Evaluates with RMSE, MAE, R².
    """

    def __init__(self, df: pd.DataFrame, target: str = "COP"):
        self.df            = df
        self.target        = target
        self.models        = {}
        self.results       = {}
        self.X_test        = None
        self.y_test        = None
        self.feature_names = []

    def prepare(self) -> "MachineLearning":
        if self.target not in self.df.columns:
            raise ValueError(f"Target '{self.target}' not in dataframe.")
        drop = [self.target]
        ts = _find_col(self.df, ["Timestamp", "datetime", "Date"])
        if ts:
            drop.append(ts)
        X = self.df.drop(columns=drop, errors="ignore").select_dtypes(include=[np.number])
        y = self.df[self.target]
        X_scaled = StandardScaler().fit_transform(X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        self.feature_names = X.columns.tolist()
        print(f"[ML] Train: {len(self.X_train):,} | Test: {len(self.X_test):,} | Features: {self.feature_names}")
        return self

    def train_all(self) -> "MachineLearning":
        defs = {
            "Linear Regression": LinearRegression(),
            "Random Forest":     RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, random_state=42),
        }
        print(f"\n{'Model':<25} {'RMSE':>9} {'MAE':>9} {'R2':>9}")
        print("-" * 55)
        for name, mdl in defs.items():
            try:
                mdl.fit(self.X_train, self.y_train)
                yp   = mdl.predict(self.X_test)
                rmse = float(np.sqrt(mean_squared_error(self.y_test, yp)))
                mae  = float(mean_absolute_error(self.y_test, yp))
                r2   = float(r2_score(self.y_test, yp))
                self.models[name]  = mdl
                self.results[name] = {"RMSE": round(rmse,5), "MAE": round(mae,5),
                                      "R2": round(r2,5), "y_pred": yp}
                print(f"{name:<25} {rmse:>9.5f} {mae:>9.5f} {r2:>9.5f}")
            except Exception as e:
                print(f"[ML ERROR] {name}: {e}")
        return self

    def best_model(self) -> str:
        if not self.results:
            return None
        return max(self.results, key=lambda k: self.results[k]["R2"])

    def feature_importance(self) -> pd.Series:
        best = self.best_model()
        mdl  = self.models.get(best)
        if mdl and hasattr(mdl, "feature_importances_"):
            imp = pd.Series(mdl.feature_importances_, index=self.feature_names).sort_values(ascending=False)
        elif mdl and hasattr(mdl, "coef_"):
            imp = pd.Series(np.abs(mdl.coef_), index=self.feature_names).sort_values(ascending=False)
        else:
            return pd.Series(dtype=float)
        print(f"\n-- Feature Importances ({best}) --")
        print(imp.to_string())
        return imp


# =============================================================================
# MODULE 5 — VISUALIZATION
# =============================================================================
class Visualization:
    """Produces 5 static plots + 2 animated GIFs, saved to outputs/."""

    def __init__(self, df: pd.DataFrame, ml: MachineLearning, output_dir: str):
        self.df         = df
        self.ml         = ml
        self.output_dir = output_dir
        self.target     = ml.target
        sns.set_theme(style="darkgrid", palette="muted")

    def plot_histogram(self):
        arr = self.df[self.target].dropna().values
        fig, ax = plt.subplots(figsize=(10, 5))
        n, bins, patches = ax.hist(arr, bins=45, edgecolor="white", alpha=0.85)
        for patch, left in zip(patches, bins[:-1]):
            patch.set_facecolor("#E63946" if left < 2.5 else "#F4A261" if left < 4.0 else "#2A9D8F")
        ax.axvline(np.mean(arr),   color="navy",   ls="--", lw=2, label=f"Mean={np.mean(arr):.3f}")
        ax.axvline(np.median(arr), color="purple", ls=":",  lw=2, label=f"Median={np.median(arr):.3f}")
        ax.axvline(2.5, color="red",   ls="-", lw=1.2, alpha=0.6, label="COP=2.5 (Low)")
        ax.axvline(4.0, color="green", ls="-", lw=1.2, alpha=0.6, label="COP=4.0 (High)")
        ax.set_title("Chiller COP Distribution — July (Peak Cooling Season)", fontsize=14, fontweight="bold")
        ax.set_xlabel("COP (Coefficient of Performance)")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=9)
        fig.tight_layout()
        path = os.path.join(self.output_dir, "static_01_cop_histogram.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[VIZ] Saved: {path}")

    def plot_heatmap(self):
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target in num_cols and len(num_cols) > 12:
            top12 = self.df[num_cols].corr()[self.target].abs().sort_values(ascending=False).head(12).index
            num_cols = list(top12)
        corr = self.df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool)),
                    annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                    linewidths=0.5, ax=ax, annot_kws={"size": 8}, vmin=-1, vmax=1)
        ax.set_title("Pearson Correlation Heatmap — Chiller Plant Features (July)",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        path = os.path.join(self.output_dir, "static_02_correlation_heatmap.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[VIZ] Saved: {path}")

    def plot_boxplot(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        if "Hour" in self.df.columns:
            self.df["HourGroup"] = pd.cut(
                self.df["Hour"], bins=[0,6,12,18,24], right=False,
                labels=["Night(0-6h)", "Morning(6-12h)", "Afternoon(12-18h)", "Evening(18-24h)"]
            )
            groups = [g[self.target].dropna().values for _, g in self.df.groupby("HourGroup", observed=True)]
            labels = [str(k) for k, _ in self.df.groupby("HourGroup", observed=True)]
            bp = ax.boxplot(groups, patch_artist=True, labels=labels,
                            medianprops=dict(color="white", lw=2.5))
            for patch, c in zip(bp["boxes"], ["#264653","#2A9D8F","#E9C46A","#E76F51"]):
                patch.set_facecolor(c); patch.set_alpha(0.8)
            ax.set_title("Chiller COP by Time-of-Day Group (July)", fontsize=13, fontweight="bold")
            ax.set_ylabel("COP")
        else:
            self.df.select_dtypes(include=[np.number]).iloc[:,:6].boxplot(ax=ax)
        fig.tight_layout()
        path = os.path.join(self.output_dir, "static_03_boxplot_by_hour.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[VIZ] Saved: {path}")

    def plot_actual_vs_predicted(self):
        best = self.ml.best_model()
        if not best:
            return
        yp   = self.ml.results[best]["y_pred"]
        yt   = self.ml.y_test.values
        resid = yt - yp
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        sc = axes[0].scatter(yt, yp, c=np.abs(resid), cmap="RdYlGn_r", alpha=0.5, s=18, edgecolor="none")
        lims = [min(yt.min(), yp.min())-0.1, max(yt.max(), yp.max())+0.1]
        axes[0].plot(lims, lims, "r--", lw=2, label="Perfect fit")
        plt.colorbar(sc, ax=axes[0], label="|Residual|")
        axes[0].set_title(f"Actual vs Predicted COP\n{best}  R²={self.ml.results[best]['R2']:.4f}",
                          fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Actual COP"); axes[0].set_ylabel("Predicted COP")
        axes[0].legend()
        axes[1].hist(resid, bins=40, color="#2E86AB", edgecolor="white", alpha=0.85)
        axes[1].axvline(0, color="red", ls="--", lw=2)
        axes[1].set_title("Residual Distribution", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Residual (Actual - Predicted COP)")
        axes[1].set_ylabel("Count")
        fig.tight_layout()
        path = os.path.join(self.output_dir, "static_04_actual_vs_predicted.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[VIZ] Saved: {path}")

    def plot_feature_importance(self):
        imp = self.ml.feature_importance()
        if imp.empty:
            return
        top = imp.head(min(10, len(imp)))
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(top)))
        ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1], edgecolor="white")
        ax.set_title(f"Feature Importance — {self.ml.best_model()} (COP Prediction)",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Importance Score")
        fig.tight_layout()
        path = os.path.join(self.output_dir, "static_05_feature_importance.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[VIZ] Saved: {path}")

    def animate_cop_rolling_mean(self):
        series   = self.df[self.target].reset_index(drop=True)
        window   = 30
        n_frames = min(250, len(series) - window)
        if n_frames < 10:
            print("[VIZ] Not enough data for animation 1.")
            return
        fig, ax = plt.subplots(figsize=(11, 5))
        line_raw,  = ax.plot([], [], lw=0.8, alpha=0.35, color="#4C72B0", label="Raw COP")
        line_roll, = ax.plot([], [], lw=2.5, color="#E63946", label=f"Rolling Mean (w={window})")
        ax.axhline(2.5, color="red",   ls="--", lw=1, alpha=0.5, label="Low (2.5)")
        ax.axhline(4.0, color="green", ls="--", lw=1, alpha=0.5, label="High (4.0)")
        ax.set_xlim(0, n_frames + window)
        ax.set_ylim(max(0, series.min()-0.3), series.max()+0.3)
        ax.set_title("Chiller COP Rolling Mean — July (Animated)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Sample Index"); ax.set_ylabel("COP")
        ax.legend(loc="upper right", fontsize=8)
        info = ax.text(0.01, 0.95, "", transform=ax.transAxes, fontsize=9, color="gray", va="top")

        def init():
            line_raw.set_data([], []); line_roll.set_data([], [])
            return line_raw, line_roll, info

        def update(frame):
            end  = frame + window
            x    = np.arange(end)
            y    = series.iloc[:end].values
            roll = pd.Series(y).rolling(window, min_periods=1).mean().values
            line_raw.set_data(x, y); line_roll.set_data(x, roll)
            info.set_text(f"n={end:,}  COP_roll={roll[-1]:.3f}")
            return line_raw, line_roll, info

        ani = FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=True, interval=35)
        path = os.path.join(self.output_dir, "anim_01_cop_rolling_mean.gif")
        ani.save(path, writer="pillow", fps=18, dpi=90)
        plt.close(fig)
        print(f"[VIZ] Saved animation: {path}")

    def animate_temp_vs_cop(self):
        x_col = _find_col(self.df, [
            "Outdoor_Temp_C", "outdoor_temp", "temperature",
            "Temp", "OAT", "ambient_temp", "Temperature"
        ])
        if not x_col:
            other = [c for c in self.df.select_dtypes(include=[np.number]).columns if c != self.target]
            x_col = other[0] if other else None
        if not x_col:
            print("[VIZ] Cannot produce scatter animation.")
            return
        x = self.df[x_col].values
        y = self.df[self.target].values
        n_frames = min(200, len(x))
        step = max(1, len(x) // n_frames)
        idxs = list(range(0, len(x), step))[:n_frames]

        fig, ax = plt.subplots(figsize=(10, 6))
        scat = ax.scatter([], [], c=[], cmap="RdYlGn", vmin=0.5, vmax=6,
                          alpha=0.55, s=20, edgecolor="none")
        ax.set_xlim(x.min()-1, x.max()+1)
        ax.set_ylim(max(0, y.min()-0.2), y.max()+0.2)
        ax.axhline(2.5, color="red",   ls="--", lw=1.2, alpha=0.5, label="Low COP")
        ax.axhline(4.0, color="green", ls="--", lw=1.2, alpha=0.5, label="High COP")
        ax.set_title(f"{x_col} vs Chiller COP — Animated Build-up (July)",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel(x_col); ax.set_ylabel("COP"); ax.legend(fontsize=8)
        plt.colorbar(scat, ax=ax, label="COP")
        info = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=9, color="gray")

        def update(frame):
            end = idxs[frame] + 1
            scat.set_offsets(np.c_[x[:end], y[:end]])
            scat.set_array(y[:end])
            info.set_text(f"n={end:,}  mean COP={np.mean(y[:end]):.3f}")
            return scat, info

        ani = FuncAnimation(fig, update, frames=n_frames, blit=True, interval=45)
        path = os.path.join(self.output_dir, "anim_02_temp_vs_cop_scatter.gif")
        ani.save(path, writer="pillow", fps=18, dpi=90)
        plt.close(fig)
        print(f"[VIZ] Saved animation: {path}")

    def run_all(self):
        print("\n[VIZ] Generating static plots...")
        self.plot_histogram()
        self.plot_heatmap()
        self.plot_boxplot()
        self.plot_actual_vs_predicted()
        self.plot_feature_importance()
        print("\n[VIZ] Generating animations (1-3 min)...")
        self.animate_cop_rolling_mean()
        self.animate_temp_vs_cop()
        print("[VIZ] All visualizations complete.")


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main():
    print("=" * 65)
    print("  EDS PIPELINE — Chiller Plant COP Variance (HVA-01)")
    print("  Pillar 9: HVAC & Building Systems")
    print("  Unique Filter: Month == 7 (July — Peak Cooling Season)")
    print("=" * 65)

    # STEP 1: Ingest
    print("\n[STEP 1] Data Ingestion")
    ingestion_path = resolve_dataset_path()
    print(f"[INGESTION] Using dataset file: {ingestion_path}")
    ingestion = DataIngestion(ingestion_path)
    df_raw = ingestion.load()
    ingestion.summary()

    # STEP 2: Clean & Engineer COP
    print("\n[STEP 2] Data Cleaning & Feature Engineering")
    cleaner = (
        DataCleaning(df_raw)
        .remove_duplicates()
        .handle_missing()
        .fix_dtypes()
        .engineer_cop()
        .apply_unique_filter(UNIQUE_FILTER_COL, UNIQUE_FILTER_VALUE)
        .remove_outliers_iqr("COP")
    )
    df_clean = cleaner.get_clean_df()
    if df_clean.empty:
        raise ValueError(
            "[ERROR] No clean data remains after preprocessing. "
            "Verify the raw dataset and COP engineering steps."
        )
    df_clean.to_csv(CLEAN_CSV, index=False)
    print(f"\n[CLEANING] Saved: {CLEAN_CSV}  ({len(df_clean):,} rows)")

    # STEP 3: Statistical Analysis
    print("\n[STEP 3] Statistical Analysis")
    analyzer = StatisticalAnalysis(df_clean, target="COP")
    analyzer.descriptive_stats()
    analyzer.distribution_analysis()
    analyzer.correlation_analysis()
    analyzer.comparative_analysis()

    # STEP 4: Machine Learning
    print("\n[STEP 4] Machine Learning — Predict COP")
    ml = MachineLearning(df_clean, target="COP")
    ml.prepare().train_all()
    best = ml.best_model()
    print(f"\n[ML] Best Model: {best} | R2={ml.results[best]['R2']:.5f} | "
          f"RMSE={ml.results[best]['RMSE']:.5f}")

    # STEP 5: Visualize
    print("\n[STEP 5] Visualization & Animation")
    Visualization(df_clean, ml, OUTPUT_DIR).run_all()

    print("\n" + "=" * 65)
    print("  PIPELINE COMPLETE")
    print(f"  Cleaned data -> {CLEAN_CSV}")
    print(f"  Static plots -> {OUTPUT_DIR}/static_0*.png  (5 files)")
    print(f"  Animations   -> {OUTPUT_DIR}/anim_0*.gif    (2 files)")
    print("=" * 65)


if __name__ == "__main__":
    main()
