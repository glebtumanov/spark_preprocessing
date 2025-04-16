#!/bin/env python
import sys, os
from campaignds.sc_connector import SparkHelper
from preprocessing import PreprocessingPipeline

try:
    config_file = sys.argv[1]
except IndexError:
    print("Не указан конфиг-файл!")
    sys.exit(0)

if not os.path.exists(config_file):
    print("Указан не существующий файл!")
    sys.exit(0)

print(f"Using config file: {config_file}\n")

helper = SparkHelper(
    dynamicAllocation='True',
    name='tenders preprocessing',
    instances=30,
    cores=3,
    spark_version='3.2.1',
    executor_memory='12g',
    driver_memory='16g',
    parcel='MPACK',
    mpack_name='sota'
)

config = {
    'spark.ui.showConsoleProgress': 'true',
    'hive.exec.dynamic.partition.mode': 'nonstrict',
    'spark.sql.sources.partitionOverwriteMode': 'dynamic',
    'spark.sql.hive.metastore.version': '2.3.9',
    'mapreduce.input.fileinputformat.input.dir.recursive': 'true',
}
sc, sqlContext = helper.create_sql_context(config)

preprocessor = PreprocessingPipeline(config_file, sc, sqlContext)
preprocessor.run_preprocessing()
