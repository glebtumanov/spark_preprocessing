def prepare_k_fold_data(df, cv=3, positive_frac=0.25, with_validation=True) -> DataFrame:
    pos_count = df.filter(f.col('target') == 1).count()
    neg_count = df.filter(f.col('target') == 0).count()
    neg_frac = pos_count * (1 / positive_frac) / neg_count

    folds = list(range(cv))
    folds_df = spark.createDataFrame([(f_num,) for f_num in folds], ['fold']).coalesce(1)

    train_df = (
        df.join(folds_df, f.col('fold') != f.col('cross_val_group'))
        .filter((f.col('target') == 1) | (f.rand() <= neg_frac))
        .withColumn('cross_val_group', f.col('fold'))
        .drop('fold')
    )

    if with_validation:
        result_df = (
            train_df.withColumn('is_train', f.lit(1))
            .unionByName(
                df.withColumn('is_train', f.lit(0))
            )
        )
    else:
        result_df = train_df

    return result_df

prepared_df = prepare_k_fold_data(train_df, bagging_count, freq_positive_target)
