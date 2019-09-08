package cse6250.fileio

import cse6250.Main._
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.functions.{lower, regexp_replace}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, SparkSession}


/* Reads in the MIMIC-III csv files */
object MimicLoader {

  def load(spark: SparkSession, data_dir: String = DATA_DIR): (DataFrame, DataFrame, DataFrame) = {
    import spark.implicits._

    if (VERBOSE) println("\nLoading MIMIC data:")
    val start_time = System.currentTimeMillis

    // load PATIENTS.csv
    if (VERBOSE) println("  PATIENTS.csv")
    val patients: DataFrame = load_patients(spark, data_dir)
      .repartition(NUM_CORES * 4)

    // load ADMISSIONS.csv
    if (VERBOSE) println("  ADMISSIONS.csv")
    val admissions: DataFrame = load_admissions(spark, data_dir)
      .repartition(NUM_CORES * 4)

    // load NOTES.csv
    if (VERBOSE) println("  NOTEEVENTS.csv")
    val notes: DataFrame = load_notes(spark, data_dir)
      // remove newlines and carriage returns
      .withColumn("text", regexp_replace($"text", "[\\r\\n]", " "))
      .repartition(NUM_CORES * 4)

    if (VERBOSE) println("  Completed in %,.1f seconds".format((System.currentTimeMillis - start_time) / 1000f))

    (patients, admissions, notes)
  }

  private def load_patients(spark: SparkSession, data_dir: String = DATA_DIR): DataFrame = {
    // specify the file schema
    val schema = ScalaReflection.schemaFor[Patient].dataType.asInstanceOf[StructType]

    // read the file
    spark.read
      .schema(schema)
      .option("header", value = true)
      .csv(data_dir + "PATIENTS.csv")
      .repartition(NUM_CORES * 4)
  }

  private def load_admissions(spark: SparkSession, data_dir: String = DATA_DIR): DataFrame = {
    // specify the file schema
    val schema = ScalaReflection.schemaFor[Admission].dataType.asInstanceOf[StructType]

    // read the file
    spark.read
      .schema(schema)
      .option("header", value = true)
      .csv(data_dir + "ADMISSIONS.csv")
      .repartition(NUM_CORES * 4)
  }

  private def load_notes(spark: SparkSession, data_dir: String = DATA_DIR): DataFrame = {
    import spark.implicits._

    // specify the file schema
    val schema = ScalaReflection.schemaFor[Note].dataType.asInstanceOf[StructType]

    // read the file
    spark.read
      .schema(schema)
      .option("header", value = true)

      // add escape character to all quotes
      .option("escape", "\"")

      // read multiline text entry as one column
      .option("multiline", value = true)

      // read the file
      .csv(data_dir + "NOTEEVENTS.csv")

      // drop instances where the text is null
      // (for some reason, the reader reads in a whole bunch of null lines)
      .filter($"text".isNotNull)

      // repartition
      .repartition(NUM_CORES * 4)
  }

  /* Generates a sample of 5% of all discharge notes. */
  def generate_notes_sample(spark: SparkSession, data_dir: String = DATA_DIR): DataFrame = {
    import spark.sqlContext.implicits._

    load_notes(spark, data_dir)
      .repartition(NUM_CORES * 4)
      .filter(lower($"category") === "discharge summary")
      .sample(fraction = 0.05, withReplacement = false, seed = RANDOM_SEED)
  }
}
