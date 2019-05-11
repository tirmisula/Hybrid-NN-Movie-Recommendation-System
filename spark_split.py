from pyspark.sql.functions import col, row_number, broadcast, rand
from pyspark.sql import Window
import numpy as np

# Default column names
DEFAULT_USER_COL = "userID"
DEFAULT_ITEM_COL = "itemID"
DEFAULT_RATING_COL = "rating"
DEFAULT_LABEL_COL = "label"
DEFAULT_TIMESTAMP_COL = "timestamp"
DEFAULT_PREDICTION_COL = "prediction"
COL_DICT = {
    "col_user": DEFAULT_USER_COL, 
    "col_item": DEFAULT_ITEM_COL, 
    "col_rating": DEFAULT_RATING_COL, 
    "col_prediction": DEFAULT_PREDICTION_COL
}

# Filtering variables
DEFAULT_K = 10
DEFAULT_THRESHOLD = 10

# Other
SEED = 42

def process_split_ratio(ratio):
    """Generate split ratio lists
    Args:
        ratio (float or list): a float number that indicates split ratio or a list of float
        numbers that indicate split ratios (if it is a multi-split).
    Returns:
        tuple: a tuple containing
            bool: A boolean variable multi that indicates if the splitting is multi or single.
            list: A list of normalized split ratios.
    """
    if isinstance(ratio, float):
        if ratio <= 0 or ratio >= 1:
            raise ValueError("Split ratio has to be between 0 and 1")

        multi = False
    elif isinstance(ratio, list):
        if any([x <= 0 for x in ratio]):
            raise ValueError(
                "All split ratios in the ratio list should be larger than 0."
            )

        # normalize split ratios if they are not summed to 1
        if sum(ratio) != 1.0:
            ratio = [x / sum(ratio) for x in ratio]

        multi = True
    else:
        raise TypeError("Split ratio should be either float or a list of floats.")

    return multi, ratio

def _check_min_rating_filter(filter_by, min_rating, col_user, col_item):
    if not (filter_by == "user" or filter_by == "item"):
        raise ValueError("filter_by should be either 'user' or 'item'.")

    if min_rating < 1:
        raise ValueError("min_rating should be integer and larger than or equal to 1.")

    split_by_column = col_user if filter_by == "user" else col_item
    split_with_column = col_item if filter_by == "user" else col_user
    return split_by_column, split_with_column

def min_rating_filter_spark(
    data,
    min_rating=1,
    filter_by="user",
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
):
    """Filter rating DataFrame for each user with minimum rating.
    Filter rating data frame with minimum number of ratings for user/item is usually useful to
    generate a new data frame with warm user/item. The warmth is defined by min_rating argument. For
    example, a user is called warm if he has rated at least 4 items.
    Args:
        data (spark.DataFrame): DataFrame of user-item tuples. Columns of user and item
            should be present in the DataFrame while other columns like rating, 
            timestamp, etc. can be optional.
        min_rating (int): minimum number of ratings for user or item.
        filter_by (str): either "user" or "item", depending on which of the two is to 
            filter with min_rating.
        col_user (str): column name of user ID.
        col_item (str): column name of item ID.
    Returns:
        spark.DataFrame: DataFrame with at least columns of user and item that has been 
            filtered by the given specifications.
    """
    split_by_column, split_with_column = _check_min_rating_filter(
        filter_by, min_rating, col_user, col_item
    )
    rating_temp = (
        data.groupBy(split_by_column)
        .agg({split_with_column: "count"})
        .withColumnRenamed("count(" + split_with_column + ")", "n" + split_with_column)
        .where(col("n" + split_with_column) >= min_rating)
    )

    rating_filtered = data.join(broadcast(rating_temp), split_by_column).drop(
        "n" + split_with_column
    )
    return rating_filtered

def spark_stratified_split(
    data,
    ratio=0.75,
    min_rating=1,
    filter_by="user",
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_rating=DEFAULT_RATING_COL,
    seed=42,
):
    """Spark stratified splitter
    For each user / item, the split function takes proportions of ratings which is
    specified by the split ratio(s). The split is stratified.
    Args:
        data (spark.DataFrame): Spark DataFrame to be split.
        ratio (float or list): Ratio for splitting data. If it is a single float number
            it splits data into two halves and the ratio argument indicates the ratio of
            training data set; if it is a list of float numbers, the splitter splits
            data into several portions corresponding to the split ratios. If a list is
            provided and the ratios are not summed to 1, they will be normalized.
            Earlier indexed splits will have earlier times
            (e.g the latest time per user or item in split[0] <= the earliest time per user or item in split[1])
        seed (int): Seed.
        min_rating (int): minimum number of ratings for user or item.
        filter_by (str): either "user" or "item", depending on which of the two is to filter
            with min_rating.
        col_user (str): column name of user IDs.
        col_item (str): column name of item IDs.
    Returns:
        list: Splits of the input data as spark.DataFrame.
    """
    if not (filter_by == "user" or filter_by == "item"):
        raise ValueError("filter_by should be either 'user' or 'item'.")

    if min_rating < 1:
        raise ValueError("min_rating should be integer and larger than or equal to 1.")

    multi_split, ratio = process_split_ratio(ratio)

    split_by_column = col_user if filter_by == "user" else col_item

    if min_rating > 1:
        data = min_rating_filter_spark(
            data,
            min_rating=min_rating,
            filter_by=filter_by,
            col_user=col_user,
            col_item=col_item,
        )

    ratio = ratio if multi_split else [ratio, 1 - ratio]
    ratio_index = np.cumsum(ratio)

    window_spec = Window.partitionBy(split_by_column).orderBy(rand(seed=seed))

    rating_grouped = (
        data.groupBy(split_by_column)
        .agg({col_rating: "count"})
        .withColumnRenamed("count(" + col_rating + ")", "count")
    )
    rating_all = data.join(broadcast(rating_grouped), on=split_by_column)

    rating_rank = rating_all.withColumn(
        "rank", row_number().over(window_spec) / col("count")
    )

    splits = []
    for i, _ in enumerate(ratio_index):
        if i == 0:
            rating_split = rating_rank.filter(col("rank") <= ratio_index[i])
        else:
            rating_split = rating_rank.filter(
                (col("rank") <= ratio_index[i]) & (col("rank") > ratio_index[i - 1])
            )

        splits.append(rating_split)

    return splits

