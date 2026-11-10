FROM apache/spark:3.5.1-python3

USER root
WORKDIR /app

# Kafka + Hadoop-AWS connectors so Spark can read Kafka and write to SeaweedFS (s3a).
ENV SPARK_KAFKA_VERSION=3.5.1
ENV IVY_JARS="org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262"

COPY spark_stream.py .

# Run an empty job at build time so the jars land in the Ivy cache used at runtime.
RUN touch /tmp/noop.py \
    && /opt/spark/bin/spark-submit \
    --packages "$IVY_JARS" \
    --conf spark.jars.ivy=/opt/ivy \
    /tmp/noop.py

CMD /opt/spark/bin/spark-submit \
    --packages "$IVY_JARS" \
    --conf spark.jars.ivy=/opt/ivy \
    /app/spark_stream.py
