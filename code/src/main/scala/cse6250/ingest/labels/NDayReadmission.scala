package cse6250.ingest.labels

import org.apache.spark.ml.feature.Bucketizer
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{datediff, lag, when}
import org.apache.spark.sql.types.DoubleType


/* Labels based on readmission within n days */
object NDayReadmission extends LabelMaker {

  private var splits: Array[Double] = _

  override def configure(args: Map[String, Any]): this.type = {
    // obtain the class split values from args
    splits = args.get("splits").asInstanceOf[Option[Array[Double]]].get.sorted

    // add negative and positive infinity to the ends
    splits = Array(Double.NegativeInfinity) ++ splits ++ Array(Double.PositiveInfinity)

    this
  }

  def generate(data: DataFrame): DataFrame = {
    import data.sqlContext.implicits._

    // === IDENTIFY UNPLANNED READMISSIONS ============================================================================
    // create a structured window to help with checking for readmissions
    val window = Window
      .partitionBy("subject_id")
      .orderBy("admit_time")

    val readmits_added: DataFrame = data
      // identify all readmissions ("lag" looks at each row's neighboring rows)
      .withColumn("next_admit_time", lag("admit_time", -1).over(window))
      .withColumn("next_admit_type", lag("admit_type", -1).over(window))
      .withColumn("next_admit_days", datediff($"next_admit_time", $"discharge_time"))

      // filter out elective hospitalizations
      .withColumn("readmit_time", when($"next_admit_type".notEqual("ELECTIVE"), $"next_admit_time"))
      .withColumn("readmit_type", when($"next_admit_type".notEqual("ELECTIVE"), $"next_admit_type"))
      .withColumn("readmit_days", when($"next_admit_type".notEqual("ELECTIVE"), $"next_admit_days"))

      // drop intermediate columns (no longer needed)
      .drop("next_admit_time", "next_admit_type", "next_admit_days")

    // === ADD LABELS =================================================================================================
    // bucketizer converts continuous data (eg, days to next admission) and transforms it into categories
    val bucketizer = new Bucketizer()
      .setInputCol("readmit_days")
      .setOutputCol("label_raw")
      .setSplits(splits)

    // if the line doesn't have a readmission, put it in the last bucket
    val nan_label = splits.length - 2
    bucketizer.transform(readmits_added)
      .na.fill(nan_label, Seq("label_raw"))

      // store the label as a double
      .withColumn("label", $"label_raw".cast(DoubleType))
      .drop("label_raw")
  }
}
