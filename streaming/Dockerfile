FROM apache/spark:3.5.1-python3

USER root
WORKDIR /app

# Kafka + Hadoop-AWS connectors so Spark can read Kafka and write to SeaweedFS (s3a).
ENV SPARK_KAFKA_VERSION=3.5.1
ENV IVY_JARS="org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262"

# Run an empty job at build time so the jars land in the Ivy cache used at runtime.
# It comes before the COPY, so editing the job doesn't download the jars again.
RUN touch /tmp/noop.py \
    && /opt/spark/bin/spark-submit \
    --packages "$IVY_JARS" \
    --conf spark.jars.ivy=/opt/ivy \
    /tmp/noop.py \
    && rm -rf /tmp/noop.py /tmp/spark-*

COPY spark_stream.py .

# Each start copies the jars (about 340 MB) into a /tmp/spark-* folder, which Spark
# deletes when it shuts down. `exec` makes Spark the container's main process, so
# `docker stop` reaches it and it shuts down cleanly; without it the stop signal went
# to the shell, Docker killed Spark after the grace period, and every stop left a
# folder behind. The rm clears any that a crash or an out-of-memory kill leaves.
CMD rm -rf /tmp/spark-* /tmp/blockmgr-* \
    && exec /opt/spark/bin/spark-submit \
    --packages "$IVY_JARS" \
    --conf spark.jars.ivy=/opt/ivy \
    /app/spark_stream.py
