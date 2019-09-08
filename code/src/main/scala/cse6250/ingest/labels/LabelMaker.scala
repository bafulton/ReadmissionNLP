package cse6250.ingest.labels

import org.apache.spark.sql.DataFrame


/* Adds labels to a DataFrame */
trait LabelMaker {

  // configure function is optional
  def configure(args: Map[String, Any]): this.type = this

  // label generation function is required
  def generate(data: DataFrame): DataFrame
}
