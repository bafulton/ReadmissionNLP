package cse6250.ingest

import cse6250.Main.{NUM_CORES, VERBOSE}
import cse6250.fileio.Note
import cse6250.ingest.labels.NDayReadmission
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{lower, udf}


object BinaryReadmitDischargeNotesOnly extends DataMaker {

  def process(patients_raw: DataFrame, admissions_raw: DataFrame, notes_raw: DataFrame): DataFrame = {
    import patients_raw.sqlContext.implicits._

    // === ADMISSIONS =================================================================================================
    if (VERBOSE) println("\nParsing admissions:")
    var start_time = System.currentTimeMillis

    // user-defined function to swap the ones and zeros in the labels
    def reverse: UserDefinedFunction = udf {
      value: Double => {
        1.0 - value
      }
    }

    // one split between classes, at 30 days to readmission
    val splits = Array(30.0)

    // generate the labels
    val admissions: DataFrame = NDayReadmission
      .configure(Map("splits" -> splits))
      .generate(admissions_raw)

      // NDayReadmission labeler gives <=30 days a label of 0; we want to swap that
      .withColumn("label_swapped", reverse($"label"))
      .drop("label")
      .withColumnRenamed("label_swapped", "label")

      // Remove NEWBORN admission codes
      // (a majority are missing their discharge notes; they're likely maintained elsewhere)
      .filter(lower($"admit_type").notEqual("newborn"))

      // drop admissions where the dicharge date is after the death date
      // (no point tring to predict readmission of those who have already died!)
      .filter($"discharge_time" < $"death_time" || $"death_time".isNull)

      .repartition(NUM_CORES * 4)

    if (VERBOSE) {
      println("  %,d admissions records".format(admissions_raw.count))
      println("  %,d admissions (filtering out newborns and dead before discharge) in %,d partitions"
        .format(admissions.count, admissions.rdd.partitions.length))
      println("  %,d of those were unplanned readmissions".format(admissions.filter($"readmit_days".isNotNull).count))
      println("  %,d of those unplanned readmissions were in <30 days".format(admissions.filter($"label" === 1).count))
      println("  Completed in %,.1f seconds".format((System.currentTimeMillis - start_time) / 1000f))
    }

    // === NOTES ======================================================================================================
    if (VERBOSE) println("\nParsing notes:")
    start_time = System.currentTimeMillis

    // Todo: Figure out why parsing the notes.csv file only uses one core (might not matter--only takes 2min)
    // Refer to these:
    // https://stackoverflow.com/questions/42169926/reading-csv-file-in-spark-in-a-distributed-manner
    // https://hackernoon.com/managing-spark-partitions-with-coalesce-and-repartition-4050c57ad5c4

    val notes: DataFrame = notes_raw.as[Note]
      // filter out all notes but discharge summaries
      .filter(lower($"category") === "discharge summary")

      // when there are multiple discharge summaries per admission code, take the latest
      .groupByKey(_.hadm_id)
      /*.reduceGroups((a: Note, b: Note) => if((a.chart_date, b.chart_date) match {
        case (Some(a: Timestamp), Some(b: Timestamp)) => a.after(b)
        case _ => true
      }) a else b)*/

      // when there are multiple discharge summaries per admission code, combine them
      .reduceGroups((a: Note, b: Note) => Note(
        row_id = a.row_id,
        subject_id = a.subject_id,
        hadm_id = a.hadm_id,
        chart_date = a.chart_date,
        chart_time = a.chart_time,
        store_time = a.store_time,
        category = a.category,
        description = a.description,
        cgid = a.cgid,
        iserror = a.iserror,
        text = Option(a.text + " " + b.text)
      ))
      .map(row => row._2)

      // filter out any rows with a value in the ISERROR column
      .filter($"iserror".isNull)

      // remove all information inside brackets ("[** blahblah **]")
      // don't need to do here, as Sunny's preprocessing code removes them instead
      //.withColumn("text", regexp_replace($"text", "\\[\\*\\*[^\\*\\]\\[]*\\*\\*\\]", ""))

      .toDF
      .repartition(NUM_CORES * 4)

    if (VERBOSE) {
      println("  %,d total notes".format(notes_raw.count))
      println("  %,d discharge memos (1 combined note per person) in %,d partitions"
        .format(notes.count, notes.rdd.partitions.length))
      println("  Completed in %,.1f seconds".format((System.currentTimeMillis - start_time) / 1000f))
    }

    // === MERGE & TRIM ===============================================================================================
    if (VERBOSE) println("\nMerging and reducing:")
    start_time = System.currentTimeMillis

    // row_id column creates conflicts--drop it before merging
    val merged = admissions.drop("row_id")
      .join(notes.drop("row_id"), Seq("hadm_id", "subject_id"), "left")

      // drop any rows with null text
      .filter($"text".isNotNull)
      // replace null text with "."
      //.na.fill(".", cols = Seq("text"))

      // Remove all unneeded columns
      .select("subject_id", "hadm_id", "text", "label")

    if (VERBOSE) println("  Completed in %,.1f seconds".format((System.currentTimeMillis - start_time) / 1000f))

    merged
  }
}
