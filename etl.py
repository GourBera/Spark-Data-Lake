import os
import configparser
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, monotonically_increasing_id
from pyspark.sql.types import TimestampType, DateType
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, \
    date_format, dayofweek

# Read config
config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID'] = config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY'] = config['AWS']['AWS_SECRET_ACCESS_KEY']
path_join = os.path.join


# Function to create spark session
def create_spark_session():
    """
    :return: spark session
    """
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.5") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    print("Spark session created")
    return spark


# File to process song data
def process_song_data(spark, input_data, output_data):
    """
    :param spark: spark session id
    :param input_data: input path
    :param output_data: output path
    :return:
    """
    # get filepath to song data file
    song_data = path_join(input_data, "song-data/A/A/A/*.json")

    # read song data file
    df = spark.read.json(song_data)

    print("Read Song data file... Successful")
    print(df.head())

    # extract columns to create songs table
    songs_table = df['song_id', 'title', 'artist_id', 'year', 'duration']
    songs_table = songs_table.dropDuplicates(['song_id'])

    # write songs table to parquet files partitioned by year and artist
    songs_table.write.partitionBy('year', 'artist_id').parquet(
        f"{output_data}songs/songs_table.parquet", 'overwrite')

    print("Song table uploaded successfully...")

    # extract columns to create artists table
    artists_table = df['artist_id', 'artist_name', 'artist_location',
                       'artist_latitude', 'artist_longitude']
    artists_table = artists_table.dropDuplicates(['artist_id'])

    # write artists table to parquet files
    artists_table.write.parquet(
        f"{output_data}artists/artists_table.parquet", 'overwrite')

    print("artists table uploaded successfully...")


# Function to process log data
def process_log_data(spark, input_data, output_data):
    """
    :param spark: spark seession id
    :param input_data: input file path
    :param output_data: output file path
    :return:
    """

    # get filepath to log data file
    log_data = path_join(input_data, 'log_data/*/*/*.json')

    # read log data file
    df = spark.read.json(log_data)
    print("Read log data file... Successful")
    print(df.head())

    # filter by actions for song plays
    df = df.filter(df.page == 'NextSong')

    # extract columns for users table
    artists_table = df['userId', 'firstName', 'lastName', 'gender', 'level']
    artists_table = artists_table.drop_duplicates(['userId'])

    # write users table to parquet files
    artists_table.write.parquet(
        f"{output_data}users/users_table.parquet", 'overwrite')
    print("Load artists_table... Successful")

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: datetime.fromtimestamp(x / 1000),
                        TimestampType())
    df = df.withColumn("timestamp", get_timestamp(col("ts")))

    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: to_date(x), TimestampType())
    df = df.withColumn("start_time", get_timestamp(col("ts")))

    # extract columns to create time table
    df = df.withColumn("hour", hour("timestamp"))
    df = df.withColumn("day", dayofmonth("timestamp"))
    df = df.withColumn("month", month("timestamp"))
    df = df.withColumn("year", year("timestamp"))
    df = df.withColumn("week", weekofyear("timestamp"))
    df = df.withColumn("weekday", date_format("timestamp", "E"))
    #     df = df.withColumn("weekday", dayofweek("timestamp"))

    # read in song data to use for songplays table
    song_df = spark.read.json(path_join(
        input_data, "song-data/A/A/A/*.json"))
    print("Read song data... Successful...")

    # extract columns from joined song and
    # log datasets to create songplays table
    song_df = song_df['artist_id', 'artist_name',
                      'artist_location', 'song_id', 'title']

    songplays_table = df.join(
        song_df,
        song_df.artist_name == df.artist, "inner").distinct().select(
        col("start_time"),
        col("userId"),
        col("level"),
        col("sessionId"),
        col("location"),
        col("userAgent"),
        col("song_id"),
        col("year"),
        col("month"),
        col("artist_id")).withColumn("songplay_id",
                                     monotonically_increasing_id())

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.partitionBy("year", "month") \
        .parquet(f"{output_data}songplays/songplays_table.parquet",
                 'overwrite')

    print("Load songplays_table... Successful...")


def main():
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://datalake21032020/"

    process_song_data(spark, input_data, output_data)
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
