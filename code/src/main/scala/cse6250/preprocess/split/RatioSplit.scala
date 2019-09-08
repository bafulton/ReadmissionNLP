package cse6250.preprocess.split

import cse6250.Main.RANDOM_SEED
import org.apache.spark.sql.DataFrame

object RatioSplit extends Splitter {

  def split(data: DataFrame, args: Map[String, Any]): (DataFrame, DataFrame) = {
    val ratio = args.get("ratio").asInstanceOf[Option[Double]].get

    // split into two parts
    val split = data.randomSplit(Array(1f - ratio, ratio), seed = RANDOM_SEED)

    // return the parts as dataframes
    (split(0).toDF, split(1).toDF)
  }
}
