import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import warnings

# ignore divide by 0 warnings
warnings.filterwarnings("ignore")


# ============================================================
# UTILITIES
# ============================================================

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def get_feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c.startswith("F") or c.startswith("P")]


def load_and_combine(csv_paths: dict) -> pd.DataFrame:
    """
    csv_paths: dict of {model_name: path_to_csv}
    """
    dfs = []
    for model, path in csv_paths.items():
        df = pd.read_csv(path)
        df["model"] = model
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


# ============================================================
# FEATURE EXTRACTION CONSISTENCY MEASURES (Section 3.4.1)
# ============================================================

def feature_activation_stats(df: pd.DataFrame):
    """
    Mean and std of feature activation per class.
    Used for Tables 3 and 4.
    """
    feature_cols = get_feature_columns(df)
    means = df.groupby("subcategory")[feature_cols].mean()
    stds = df.groupby("subcategory")[feature_cols].std()
    return means, stds


def pearson_correlation(df1: pd.DataFrame, df2: pd.DataFrame) -> float:
    """
    Pearson Correlation determines cross-model corpus-level agreement
    """
    feature_cols = get_feature_columns(df1)

    mean1 = df1[feature_cols].mean()
    mean2 = df2[feature_cols].mean()

    # Exclude zero-variance features (std == 0)
    valid_cols = [
        c for c in feature_cols
        if df1[c].std() > 0 and df2[c].std() > 0
    ]

    return float(mean1[valid_cols].corr(mean2[valid_cols]))


def cross_model_similarity(df1: pd.DataFrame, df2: pd.DataFrame) -> dict:
    """
    Per-example cosine similarity between two models for the same input.
    Used for cross-model example-level agreement.
    """
    feature_cols = get_feature_columns(df1)

    merged = df1.merge(df2, on="id", suffixes=("_1", "_2"))

    sims = []
    for _, row in merged.iterrows():
        v1 = row[[f + "_1" for f in feature_cols]].values.astype(float)
        v2 = row[[f + "_2" for f in feature_cols]].values.astype(float)
        sim = cosine_similarity([v1], [v2])[0][0]
        sims.append(sim)

    return {
        "mean_similarity": np.mean(sims),
        "std_similarity": np.std(sims),
    }


# ============================================================
# FEATURE SPACE STRUCTURE MEASURES (Section 3.4.2)
# ============================================================

def compute_similarity_stats(df_model: pd.DataFrame) -> dict:
    """
    Within-class and between-class cosine similarity statistics.
    Used for Table 5.
    """
    feature_cols = get_feature_columns(df_model)

    X = df_model[feature_cols].values
    labels = df_model["subcategory"].values

    sim = cosine_similarity(X)

    same = []
    diff = []

    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            if labels[i] == labels[j]:
                same.append(sim[i, j])
            else:
                diff.append(sim[i, j])

    return {
        "mean_same": np.mean(same),
        "mean_diff": np.mean(diff),
        "std_same": np.std(same),
        "std_diff": np.std(diff),
    }


def similarity_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wrapper to compute similarity stats per model.
    """
    results = {}
    for model in df["model"].unique():
        df_model = df[df["model"] == model]
        results[model] = compute_similarity_stats(df_model)
    return pd.DataFrame(results).T


def compute_pca(df_model: pd.DataFrame):
    """
    PCA projection of feature vectors into 2D.
    """
    feature_cols = get_feature_columns(df_model)
    X = df_model[feature_cols].values
    y = df_model["subcategory"].values
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    return X_2d, y


def plot_pca(df_model: pd.DataFrame, filename: str):
    """
    Plot and save PCA projection for a single model.
    """
    X_2d, y = compute_pca(df_model)

    plt.figure(figsize=(6, 5))
    for label in set(y):
        idx = (y == label)
        plt.scatter(X_2d[idx, 0], X_2d[idx, 1], label=label, alpha=0.7)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

# ============================================================
# RUNNER
# ============================================================

if __name__ == "__main__":
    # Load data
    models = {
        "gpt-5-4":   load_data("data/fallacy-features-gpt.csv"),
        "claude":    load_data("data/fallacy-features-claude.csv"),
        "llama-3-1": load_data("data/fallacy-features-ollama.csv"),
    }

    # Combine into single dataframe
    dfs = []
    for model, df_model in models.items():
        df_model = df_model.copy()
        df_model["model"] = model
        dfs.append(df_model)
    df = pd.concat(dfs, ignore_index=True)

    # Feature activation stats (Tables 3 and 4)
    print("=== FEATURE ACTIVATION STATS ===")
    for model, df_model in models.items():
        print(f"\n{model}")
        means, stds = feature_activation_stats(df_model)
        print("Means:\n", means.round(3))
        print("Stds:\n", stds.round(3))

    # Pearson cross-model correlation
    print("\n=== PEARSON CORRELATION ===")
    pairs = [
        ("gpt-5-4", "claude"),
        ("gpt-5-4", "llama-3-1"),
        ("claude",  "llama-3-1"),
    ]
    for m1, m2 in pairs:
        r = pearson_correlation(models[m1], models[m2])
        print(f"{m1} x {m2}: {r:.3f}")

    # Per-example cross-model cosine similarity
    print("\n=== CROSS-MODEL COSINE SIMILARITY ===")
    for m1, m2 in pairs:
        result = cross_model_similarity(models[m1], models[m2])
        print(f"{m1} x {m2}: {result}")

    # Within/between class similarity (Table 6)
    print("\n=== WITHIN/BETWEEN CLASS SIMILARITY ===")
    print(similarity_by_model(df).round(3))

    # PCA plots
    print("\n=== GENERATING PCA PLOTS ===")
    for model, df_model in models.items():
        df_model = df_model.copy()
        df_model["model"] = model
        plot_pca(df_model, f"figures/{model}-pca.png")
        print(f"Saved figures/{model}-pca.png")
