package cse6250.preprocess.split

import org.apache.spark.sql.DataFrame

trait Splitter {
  def split(data: DataFrame, args: Map[String, Any]): (DataFrame, DataFrame)
}
