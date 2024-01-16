import logging
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json
from pyspark.sql.types import StructType, StructField, StringType, LongType
from pyspark.ml.recommendation import ALS
import pyspark.sql.functions as F
from pyspark.sql.functions import col



# Initialize logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s:%(funcName)s:%(levelname)s:%(message)s')
logger = logging.getLogger("spark_structured_streaming")


def initialize_spark_session():
    """
    Initialize the Spark Session with provided configurations.
    Params:
    :return: Spark session object or None if there's an error.
    """
    try:
        
        spark = SparkSession \
                .builder \
                .appName("SparkStructuredStreaming") \
                .config("spark.jars.packages", "com.datastax.spark:spark-cassandra-connector_2.12:3.0.0, org.apache.spark:spark-sql-kafka-0-10_2.12:3.0.0") \
                .config("spark.cassandra.connection.host", "cassandra") \
                .config("spark.cassandra.connection.port","9042")\
                .config("spark.cassandra.auth.username", "cassandra") \
                .config("spark.cassandra.auth.password", "cassandra") \
                .getOrCreate()

        spark.sparkContext.setLogLevel("ERROR")
        logger.info('Spark session initialized successfully')
    except Exception as e:
        logger.error(f"Spark session initialization failed. Error: {e}")

    return spark


def get_streaming_dataframe(spark_session, brokers, topic):
    """
    Get a streaming dataframe from Kafka.
    
    :param spark: Initialized Spark session.
    :param brokers: Comma-separated list of Kafka brokers.
    :param topic: Kafka topic to subscribe to.
    :return: Dataframe object or None if there's an error.
    """
    try:
        df = spark_session\
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", brokers) \
            .option("subscribe", topic) \
            .option("delimiter", ",") \
            .option("startingOffsets", "earliest") \
            .load()
        logger.info("Streaming dataframe fetched successfully")
    except Exception as e:
        logger.warning(f"Failed to fetch streaming dataframe. Error: {e}")

    return df


def transform_streaming_data(df):
    """
    Transform the initial dataframe to get the final structure.
    
    :param df: Initial dataframe with raw data.
    :return: Transformed dataframe.
    """
    schema = StructType([
        StructField("userId", LongType(), False),
        StructField("movieId", LongType(), False),
        StructField("rating", LongType(), False),
        StructField("timestamp", StringType(), False)
    ])

    transformed_df = df.selectExpr("CAST(value AS STRING)") \
        .select(from_json(col("value"), schema).alias("data")) \
        .select("data.*")
    return transformed_df

def als_collaborative_filter (df, cols_to_drop):

    """
    This fucnction applies my ALS algorithm in my experimental notebook
    1) it creates and hold put validation approach
    2)it fits an ALS algorithm to the trainin data partition
    3) it makes prediction on the fly on all arriving instances

    :param df: Transormed df with my schema enforced and value unwinded
    :return ALS predictions
    """
    df = df.drop(col(cols_to_drop))

    df_train, df_test = df.randomSplit(weights=[0.7,0.3], seed=100)

    als_model = ALS(
    maxIter = 10, 
    rank = 15,
    regParam = 0.05, 
    userCol= df.userId, 
    itemCol= df.movieId, 
    ratingCol = df.rating, 
    coldStartStrategy="drop")

    model = als_model.fit(df_train)

    df_prediction = model.transform(df_test)

    return df_prediction




def initiate_streaming_to_cassandra(df):
    """"
    Optional function to store on a cassandra table bucket that comes with docker-compose installation
    Start streaming the stream of predictions into the specified cassandra table.
    
    :param df: df with the streaming predictions.
    :param path: S3 bucket path.
    :param checkpoint_location: Checkpoint location for streaming.
    :return: None
    : choosing an optional S3 Sink for streaming my prediction of ML ALS algorithm into a S3 bucket
    """
    logger.info("Initiating streaming process...")
    stream_query = (df.writeStream
                    .format("org.apache.spark.sql.cassandra")
                    .outputMode("append")
                    .options(table = "movie_rec", key_space = "spark_streaming")
                    .start())
    return stream_query.awaitTermination()


def main():
    # app_name = "SparkStructuredStreamingASLModel"
    brokers = "kafka1:19092,kafka2:19093,kafka3:19094"
    topic = "movies_rec"


    spark = initialize_spark_session()
    if spark:
        df = get_streaming_dataframe(spark, brokers, topic)
        if df:
            transformed_df = transform_streaming_data(df)
            df_prediction = als_collaborative_filter(transformed_df, 'timestamp')
            initiate_streaming_to_cassandra(df_prediction)



# Execute the main function if this script is run as the main module
if __name__ == '__main__':
    main()
