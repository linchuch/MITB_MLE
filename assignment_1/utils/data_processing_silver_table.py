import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_silver_table(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_features_attribute_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: fix data errors
    df = df.withColumn("Name_clean", F.regexp_replace(F.col("Name"), r'"[^"]*$', '')).withColumn("Name_clean", F.regexp_replace(F.col("Name_clean"), '"', '')).withColumn("Name_clean", F.trim(F.col("Name_clean")))
    df = df.withColumn("Age_clean", F.regexp_replace(F.col("Age"), "_", ""))
    df = df.withColumn("Age_capped", F.least(F.lit(100), F.greatest(F.lit(0), F.col("Age_clean"))))
    df = df.replace("_______", "Unknown", subset=["Occupation"])
    df = df.replace("#F%$D@*&8","Unknown", subset=["SSN"])
    
    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "Customer_ID": StringType(),
        "Name": StringType(),
        "Name_clean": StringType(),
        "Age": StringType(),
        "Age_clean": IntegerType(),
        "Age_capped": IntegerType(),
        "SSN": StringType(),
        "Occupation": StringType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))


    # save silver table - IRL connect to database to write
    partition_name = "silver_features_attribute_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df