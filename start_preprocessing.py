#!/bin/env python
import sys, os
from campaignds.sc_connector import SparkHelper
from preprocessing import PreprocessingPipeline

helper = SparkHelper(
    dynamicAllocation='True',
    name='shadow_smz_preprocessing',
    instances=30,
    cores=3,
    spark_version='3.2.1',
    executor_memory='12g',
    driver_memory='8g',
    parcel='MPACK',
    mpack_name='sota'
)

config = {
    'spark.sql.execution.arrow.enabled': 'true',
    'spark.executor.memoryOverhead': '32g',
    'spark.sql.shuffle.partitions': '600',
    'spark.sql.broadcastTimeout': '600s',
    'spark.memory.offHeap.size': 'true',
    'spark.memory.offHeap.size': "50g",
    'spark.shuffle.spill.compress': 'true',
    'spark.shuffle.compress': 'true',
    'spark.ui.showConsoleProgress': 'true',
    'hive.exec.dynamic.partition.mode': 'nonstrict',
    'spark.sql.sources.partitionOverwriteMode': 'dynamic',
    'spark.sql.hive.metastore.version': '2.3.9',
    'mapreduce.input.fileinputformat.input.dir.recursive': 'true',
}
sc, sqlContext = helper.create_sql_context(config)

config_file = 'shadow_smz_lama.yaml'
print(f"\nUsing config file: {config_file}\n")


preprocessor = PreprocessingPipeline(config_file, sc, sqlContext)
# 74 каолонки, сырые фичи
train_df = sqlContext.read.parquet("hdfs://arnsdpsbx/user/17126416_omega-sbrf-ru/business_shadow/202502_shadow_smz_2504")
valid_df = sqlContext.read.parquet("hdfs://arnsdpsbx/user/17126416_omega-sbrf-ru/business_shadow/202503_shadow_smz_2504")
scoring_df = sqlContext.table("arnsdpsbx_t_team_cvm.scoring_result_202504_shadow_smz_2504")

# Запускаем препроцессинг с переданными DataFrame
train_processed, valid_processed, scoring_processed = preprocessor.run_preprocessing(train_df, valid_df, scoring_df)
