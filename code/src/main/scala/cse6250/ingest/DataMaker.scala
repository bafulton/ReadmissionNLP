package cse6250.ingest

import org.apache.spark.sql.DataFrame

trait DataMaker {
  def process(patients: DataFrame, admissions: DataFrame, notes: DataFrame): DataFrame
}
