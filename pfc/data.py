import datetime
import io
import os
from pathlib import Path
from typing import Dict, List, Tuple
from zipfile import ZipFile

import httpx
import polars as pl


def get_hf_raw_dataframes() -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Fetch the dataset and test data from GitHub and return as Polars DataFrames.
    Uses local cache if available.

    Returns:
        Tuple[pl.DataFrame, pl.DataFrame]: Tuple containing the train and test DataFrames
    """
    project_root = Path(__file__).parent.parent
    raw_dir = project_root / "data/raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    train_cache = raw_dir / "hf_train.csv"
    test_cache = raw_dir / "hf_test.csv"

    repository_url = (
        "https://raw.githubusercontent.com/deepfunding/mini-contest/refs/heads/main/"
    )

    # Try to load from cache first
    if train_cache.exists() and test_cache.exists():
        df_train = pl.read_csv(train_cache)
        df_test = pl.read_csv(test_cache)
    else:
        # Download and cache if not available
        df_train = pl.read_csv(f"{repository_url}/dataset.csv")
        df_test = pl.read_csv(f"{repository_url}/test.csv")

        # Cache the raw files
        df_train.write_csv(train_cache)
        df_test.write_csv(test_cache)

    # Light preprocessing to get project IDs instead of full URLs
    df_train = df_train.with_columns(
        pl.col("project_a").str.split("github.com/").list.last().alias("project_a"),
        pl.col("project_b").str.split("github.com/").list.last().alias("project_b"),
    )

    df_test = df_test.with_columns(
        pl.col("project_a").str.split("github.com/").list.last().alias("project_a"),
        pl.col("project_b").str.split("github.com/").list.last().alias("project_b"),
    )

    return df_train, df_test


def get_pond_raw_dataframes() -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Fetch the dataset and test data from CryptoPond and return as Polars DataFrames.
    Uses local cache if available.

    Returns:
        Tuple[pl.DataFrame, pl.DataFrame]: Tuple containing the train and test DataFrames
    """
    project_root = Path(__file__).parent.parent
    raw_dir = project_root / "data/raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    train_cache = raw_dir / "pond_train.csv"
    test_cache = raw_dir / "pond_test.csv"

    zip_url = "https://pond-open-files.s3.us-east-1.amazonaws.com/frontier/others/6dsYE4Dz/dataset.zip"

    response = httpx.get(zip_url)
    f = ZipFile(io.BytesIO(response.content))

    if train_cache.exists() and test_cache.exists():
        df_train = pl.read_csv(train_cache)
        df_test = pl.read_csv(test_cache)
    else:
        # Download and cache if not available
        df_train = pl.read_csv(f.read("dataset.csv"))
        df_test = pl.read_csv(f.read("test.csv"))

        # Cache the raw files
        df_train.write_csv(train_cache)
        df_test.write_csv(test_cache)

    # Light preprocessing to get project IDs instead of full URLs
    df_train = df_train.with_columns(
        pl.col("project_a").str.split("github.com/").list.last().alias("project_a"),
        pl.col("project_b").str.split("github.com/").list.last().alias("project_b"),
    )

    # Cast columns to the correct types
    df_train = df_train.select(
        [
            pl.col("id"),
            pl.col("project_a").cast(pl.String),
            pl.col("project_b").cast(pl.String),
            pl.col("weight_a"),
            pl.col("weight_b"),
            pl.col("total_amount_usd"),
            pl.col("funder").fill_null("unknown").alias("funder"),
            pl.col("quarter").str.split("-").list.get(0).cast(pl.Int32).alias("year"),
            pl.col("quarter").str.split("-").list.get(1).cast(pl.Int32).alias("month"),
        ]
    )

    df_test = df_test.with_columns(
        pl.col("project_a").str.split("github.com/").list.last().alias("project_a"),
        pl.col("project_b").str.split("github.com/").list.last().alias("project_b"),
    )

    df_test = df_test.select(
        [
            pl.col("id"),
            pl.col("project_a").cast(pl.String),
            pl.col("project_b").cast(pl.String),
            pl.col("total_amount_usd"),
            pl.col("funder").alias("funder"),
            pl.col("quarter").str.split("-").list.get(0).cast(pl.Int32).alias("year"),
            pl.col("quarter").str.split("-").list.get(1).cast(pl.Int32).alias("month"),
        ]
    )

    return df_train, df_test


def get_repository_info(repository_id: str, client: httpx.Client) -> Dict:
    """
    Fetch repository information from GitHub API for a given repo URL.

    Args:
        repo_url: GitHub repository URL
        client: httpx.Client instance to use for requests

    Returns:
        Dict containing repository information or empty dict if request fails
    """
    api_url = f"https://api.github.com/repos/{repository_id}"

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    try:
        response = client.get(api_url, headers=headers)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError:
        print(f"Error fetching data for {repository_id}")
        print(response.text)
        return {}


def get_projects_info(projects: List[str]) -> pl.DataFrame:
    """
    Fetch project information from GitHub API for a list of project IDs and return as a Polars DataFrame.
    Uses local cache if available.

    Args:
        projects: List of GitHub repository IDs

    Returns:
        pl.DataFrame containing GitHub project information for all projects
    """
    project_root = Path(__file__).parent.parent
    processed_dir = project_root / "data/processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_file = processed_dir / "projects_info.parquet"

    if cache_file.exists():
        return pl.read_parquet(cache_file)

    data = []
    with httpx.Client(
        transport=httpx.HTTPTransport(retries=5, verify=False),
        follow_redirects=True,
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
    ) as client:
        for project_id in projects:
            info = get_repository_info(project_id, client)
            if info:
                data.append(info)

    df = pl.DataFrame(data)
    df.write_parquet(cache_file)
    return df


def get_unique_projects(challenge: str) -> List[str]:
    if challenge == "hf":
        df_train, df_test = get_hf_raw_dataframes()
    elif challenge == "pond":
        df_train, df_test = get_pond_raw_dataframes()
    else:
        raise ValueError(f"Invalid challenge: {challenge}")

    projects = (
        pl.concat(
            [
                df_train.get_column("project_a"),
                df_train.get_column("project_b"),
                df_test.get_column("project_a"),
                df_test.get_column("project_b"),
            ]
        )
        .unique()
        .to_list()
    )

    return projects


def mirror_projects(df: pl.DataFrame) -> pl.DataFrame:
    return pl.concat(
        [
            df,
            df.with_columns(
                "id",
                pl.col("project_b").alias("project_a"),
                pl.col("project_a").alias("project_b"),
                pl.col("weight_b").alias("weight_a"),
                pl.col("weight_a").alias("weight_b"),
            ),
        ]
    )


def add_github_projects_data(df: pl.DataFrame, challenge: str) -> pl.DataFrame:
    df_projects = get_projects_info(get_unique_projects(challenge))

    df_projects = df_projects.select(
        pl.col("full_name").str.to_lowercase().alias("project_id"),
        pl.col("full_name").str.split("/").list.get(0).alias("organization"),
        pl.col("private").alias("is_private"),
        pl.col("description"),
        pl.col("created_at"),
        pl.col("updated_at"),
        pl.col("homepage").is_not_null().alias("has_homepage"),
        pl.col("size"),
        pl.col("stargazers_count").alias("stars"),
        pl.col("watchers_count").alias("watchers"),
        pl.col("language"),
        pl.col("has_projects"),
        pl.col("has_pages"),
        pl.col("has_wiki"),
        pl.col("has_discussions"),
        pl.col("forks_count").alias("forks"),
        pl.col("archived").alias("is_archived"),
        pl.col("disabled").alias("is_disabled"),
        pl.col("open_issues_count").alias("open_issues"),
        pl.col("network_count").alias("network_count"),
        pl.col("subscribers_count"),
    )

    df = df.join(
        df_projects,
        left_on="project_a",
        right_on="project_id",
        how="left",
        suffix="_a",
    )

    df = df.join(
        df_projects,
        left_on="project_b",
        right_on="project_id",
        how="left",
        suffix="_b",
    )

    return df


def extract_ratio_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Extract ratio-based features from repository data.

    Args:
        df: Input DataFrame with repository data

    Returns:
        DataFrame with added ratio features
    """
    features = df.clone()

    # Basic ratios
    features = features.with_columns(
        [
            (pl.col("stars") / (pl.col("stars") + pl.col("stars_b"))).alias(
                "stars_ratio"
            ),
            (pl.col("watchers") / (pl.col("watchers") + pl.col("watchers_b"))).alias(
                "watchers_ratio"
            ),
            (pl.col("forks") / (pl.col("forks") + pl.col("forks_b"))).alias(
                "forks_ratio"
            ),
            (pl.col("size") / (pl.col("size") + pl.col("size_b"))).alias("size_ratio"),
            (
                pl.col("open_issues")
                / (pl.col("open_issues") + pl.col("open_issues_b"))
            ).alias("issues_ratio"),
            (
                pl.col("subscribers_count")
                / (pl.col("subscribers_count") + pl.col("subscribers_count_b"))
            ).alias("subscribers_count_ratio"),
        ]
    )

    return features


def extract_interaction_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Extract interaction features from repository data.
    """
    features = df.clone()

    features = features.with_columns(
        [
            (pl.col("stars") * pl.col("forks")).alias("stars_forks_interaction"),
            (pl.col("stars_b") * pl.col("forks_b")).alias("stars_forks_interaction_b"),
            (pl.col("watchers") * pl.col("subscribers_count")).alias(
                "engagement_score"
            ),
            (pl.col("watchers_b") * pl.col("subscribers_count_b")).alias(
                "engagement_score_b"
            ),
            (pl.col("stars") * pl.col("watchers")).alias("stars_watchers_interaction"),
            (pl.col("stars_b") * pl.col("watchers_b")).alias(
                "stars_watchers_interaction_b"
            ),
            (pl.col("stars") * pl.col("size")).alias("stars_size_interaction"),
            (pl.col("stars_b") * pl.col("size_b")).alias("stars_size_interaction_b"),
        ]
    )

    return features


def extract_temporal_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Extract temporal features from repository data.

    Args:
        df: Input DataFrame with repository data

    Returns:
        DataFrame with added temporal features
    """
    features = df.clone()

    features = features.with_columns(
        [
            pl.col("created_at").cast(pl.Datetime).alias("created_at"),
            pl.col("updated_at").cast(pl.Datetime).alias("updated_at"),
            pl.col("created_at_b").cast(pl.Datetime).alias("created_at_b"),
            pl.col("updated_at_b").cast(pl.Datetime).alias("updated_at_b"),
        ]
    )

    # Calculate days since last update
    now = pl.lit(datetime.datetime.now())
    features = features.with_columns(
        [
            ((now - pl.col("updated_at")).dt.total_days()).alias("days_since_update"),
            ((now - pl.col("updated_at_b")).dt.total_days()).alias(
                "days_since_update_b"
            ),
            ((now - pl.col("created_at")).dt.total_days()).alias("days_since_creation"),
            ((now - pl.col("created_at_b")).dt.total_days()).alias(
                "days_since_creation_b"
            ),
        ]
    )

    return features


def generate_submission(predictions, data: str = "", challenge: str = "hf") -> None:
    if challenge == "hf":
        df_test = get_hf_raw_dataframes()[1]
    elif challenge == "pond":
        df_test = get_pond_raw_dataframes()[1]
    else:
        raise ValueError(f"Invalid challenge: {challenge}")

    predictions = pl.Series(predictions)

    # Cap at 0 and at 1
    predictions = predictions.clip(0, 1)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    submissions_dir = Path(__file__).parent.parent / "data/submissions"
    df_test.select(pl.col("id"), predictions.alias("pred")).write_csv(
        submissions_dir
        / challenge
        / f"submission_{timestamp}{f'_{data}' if data else ''}.csv"
    )
