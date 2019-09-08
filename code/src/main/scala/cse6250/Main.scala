package cse6250

import java.nio.file.{Files, Paths}

import cse6250.Utilities._
import cse6250.fileio._
import cse6250.ingest._
import cse6250.learn._
import cse6250.preprocess._
import cse6250.preprocess.split.RatioSplit
import org.apache.spark.sql.functions.regexp_replace
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.sys.process._


object Main {

  val VERBOSE = true
  val NUM_CORES = 8
  val RANDOM_SEED = 12345
  val DATA_DIR = "../data/"
  val MODEL_DIR = "../models/"
  val GENERATE_NOTES_SAMPLE = false

  // learner settings
  val datamaker: DataMaker = BinaryReadmitDischargeNotesOnly // set the datamaker here
  val preprocessor: Preprocessor = BagOfWords // set the preprocessor here
  var learner: Learner = LRLearnerWithMatching // set the learner here

  def main(args: Array[String]): Unit = {

    // hide log messages
    import org.apache.log4j.{Level, Logger}
    Logger.getRootLogger.setLevel(Level.ERROR)

    // parse arguments
    if (args.length > 0) {
      learner = {
        if (args(0).toLowerCase.equals("lr")) LRLearnerWithMatching
        else if (args(0).toLowerCase.equals("gbt")) GBTLearnerWithMatching
        else if (args(0).toLowerCase.equals("mlp")) MLPLearnerWithMatching
        else if (args(0).toLowerCase.equals("rfr")) RFLearnerWithMatching
        else learner
      }
    }

    for(arg <- args) {
      println(arg)
    }

    // initialize a spark session
    val spark = SparkSession.builder
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    // generate a sample of (human or pandas readable) discharge notes, if desired
    if (GENERATE_NOTES_SAMPLE) {
      // save a csv that's easy for humans to read
      val for_humans = MimicLoader.generate_notes_sample(spark)
      DataHandler.save(for_humans, "NOTEEVENTS_sampleforhumans.csv")

      // make a pandas-readable csv
      val for_pandas = for_humans
        // remove newlines and carriage returns from the text
        .withColumn("text", regexp_replace($"text", "[\\r\\n]", " "))
        // remove all commas from the text
        .withColumn("text", regexp_replace($"text", "[,]", ""))
      DataHandler.save(for_pandas, "NOTEEVENTS_sampleforpandas.csv")
    }

    if (Files.exists(Paths.get(learner.learner_loc))
      && Files.exists(Paths.get(DATA_DIR + "test.parquet"))) {

      // jump straight to testing the learner
      test_learner(spark)

    } else if (Files.exists(Paths.get(DATA_DIR + "train.parquet"))
      && Files.exists(Paths.get(DATA_DIR + "test.parquet"))) {

      // jump straight to training the learner
      train_learner(spark)

    } else if (Files.exists(Paths.get(DATA_DIR + "combined.parquet"))) {

      // jump to preprocessing the data (and then run the learner)
      preprocess_data(spark)

    } else {

      // clean the mimic data, then preprocess and run the learner
      clean_mimic_data(spark)
    }

    // close the spark instance
    spark.close()

    // generate the graphs & charts (calls a Python script)
    println("\nGenerating charts:")
    "python ../analysis/chart_maker/make_charts.py" ! ProcessLogger(stdout.append(_), stderr.append(_))
    print("\n\n")
  }

  /* Combines, cleans, and labelizes the MIMIC files; corresponds to code in the ingest package. */
  def clean_mimic_data(spark: SparkSession): Unit = {

    // read in the MIMIC files
    val (patients, admissions, notes) = MimicLoader.load(spark)

    // clean and combine the data into one dataset
    val combined = datamaker.process(patients, admissions, notes)

    // save the data to file (parquet and csv)
    DataHandler.save(combined, "combined.parquet")
    DataHandler.save(combined, "combined.csv")

    // call the preprocessor function
    // don't pass the dataframe, as reloading from disk runs faster for some reason
    // Todo: Figure out why the code runs faster when the data is reloaded from file.
    preprocess_data(spark)
  }

  /* Overloaded method; loads files from disk and then calls its parent function (see below). */
  def preprocess_data(spark: SparkSession): Unit = {

    // load the data from file
    val combined: DataFrame = DataHandler.load(spark, "combined.parquet")

    // call the parent function
    preprocess_data(spark, combined)
  }

  /* Splits and featurizes the data; corresponds to code in the preprocess package. */
  def preprocess_data(spark: SparkSession, combined: DataFrame): Unit = {

    // split out the train vs. test portions
    val test_ratio: Double = 0.15
    val (raw_train, raw_test) = RatioSplit.split(combined, Map("ratio" -> test_ratio))

    // featurize the data (vectorize/tokenize/etc.)
    val feature_map: DataFrame = preprocessor.fit(raw_train)
    val train = preprocessor.featurize(raw_train, data_type = "training")
    val test = preprocessor.featurize(raw_test)

    // save the vectorized data as parquet files (computer-readable)
    DataHandler.save(train, "train.parquet")
    DataHandler.save(test, "test.parquet")
    DataHandler.save(feature_map, "feature_map.parquet")

    // save the vectorized data as csvs (human-readable)
    DataHandler.save(split_out_vector(train, col_name = "features"), "train.csv")
    DataHandler.save(split_out_vector(test, col_name = "features"), "test.csv")
    DataHandler.save(feature_map, "feature_map.csv")

    // train the learner
    // don't pass the dataframes, as reloading from disk runs faster for some reason
    // Todo: Figure out why the code runs faster when the data is reloaded from file.
    train_learner(spark)
  }

  /* Overloaded method; loads files from disk and then calls its parent function (see below). */
  def train_learner(spark: SparkSession): Unit = {

    // load the data from file
    val train: DataFrame = DataHandler.load(spark, "train.parquet")
    val test: DataFrame = DataHandler.load(spark, "test.parquet")
    val feature_map: DataFrame = DataHandler.load(spark, "feature_map.parquet")

    // call the parent function
    train_learner(spark, train, test, feature_map)
  }

  /* Trains the learner; corresponds to code in the learn package. */
  def train_learner(spark: SparkSession, train: DataFrame, test: DataFrame, feature_map: DataFrame): Unit = {

    // train the learner
    var feature_importance = learner.train(train)

    // save the trained model
    learner.save_learner()

    // save the feature importance dataframe (only if it's not empty)
    if (!feature_importance.rdd.isEmpty() && !feature_map.rdd.isEmpty()) {
      // merge the importance and feature map dfs
      feature_importance = feature_importance
        .join(feature_map, usingColumn = "feature")
        .select("feature", "value", "importance")
    }
    DataHandler.save(feature_importance, "feature_importance.csv")

    // generate test data predictions
    test_learner(learner, test)
  }

  /* Overloaded method; loads trained model from disk and then calls its parent function (see below). */
  def test_learner(spark: SparkSession): Unit = {

    // load the test data from file
    val test: DataFrame = DataHandler.load(spark, "test.parquet")

    // load the trained model
    learner.load_learner()

    // call the parent function
    test_learner(learner, test)
  }

  /* Generates predictions and outputs the results; corresponds to code in the learn package. */
  def test_learner(learner: Learner, test: DataFrame): Unit = {
    // make predictions
    val predictions: DataFrame = learner.predict(test)

    // save the predictions
    DataHandler.save(predictions, "predictions_test.csv")
  }
}
