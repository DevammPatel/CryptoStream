FROM bitnami/spark:3.5.1

USER root
WORKDIR /app

# Kafka + Hadoop-AWS connectors so Spark can read Kafka and write to MinIO (s3a).
ENV SPARK_KAFKA_VERSION=3.5.1
ENV IVY_JARS="org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262"

COPY spark_stream.py .

# Pre-resolve the jars at build time so the container starts fast and works offline.
RUN /opt/bitnami/spark/bin/spark-submit --packages "$IVY_JARS" --version || true

CMD /opt/bitnami/spark/bin/spark-submit \
    --packages "$IVY_JARS" \
    --conf spark.jars.ivy=/tmp/.ivy \
    /app/spark_stream.py
