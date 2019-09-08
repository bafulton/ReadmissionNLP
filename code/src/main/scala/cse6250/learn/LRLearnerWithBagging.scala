package cse6250.learn

import java.io.File

import cse6250.Main._
import cse6250.Utilities._
import cse6250.evaluate._
import cse6250.fileio.DataHandler
import org.apache.spark.ml.classification.{LogisticRegressionModel, LogisticRegression}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

import scala.collection.mutable.ArrayBuffer


object LRLearnerWithBagging extends Learner {
  private val ensemble: ArrayBuffer[LogisticRegressionModel] = ArrayBuffer()
  val learner_loc: String = MODEL_DIR + this.getClass.getSimpleName.replace("$", "")

  def load_learner(): Unit = {
    if (VERBOSE) println("\nLoading trained model:\n  " + learner_loc)
    val start_time = System.currentTimeMillis

    // create a list of the trained models--supports multiple models (bagging)
    val model_dirs = new File(learner_loc).listFiles.filter(_.isDirectory).toList

    // loop over all the trained models in the learner's model directory
    var i = 1
    for (dir <- model_dirs) {
      if (VERBOSE) println("  Model " + i)
      i += 1

      ensemble += LogisticRegressionModel.load(dir.getPath)
    }

    if (VERBOSE) println("  Completed in %,.1f seconds".format((System.currentTimeMillis - start_time) / 1000f))
  }

  def save_learner(): Unit = {
    if (VERBOSE) println("\nSaving trained model:\n  " + learner_loc)
    val start_time = System.currentTimeMillis

    // save the pipelinemodel to disk
    for (i <- ensemble.indices) {
      if (VERBOSE) println("  Model " + (i + 1))
      ensemble(i).save(learner_loc + "/model_" + (i + 1))
    }

    if (VERBOSE) println("  Completed in %,.1f seconds".format((System.currentTimeMillis - start_time) / 1000f))
  }

  def train(data: DataFrame): DataFrame = {
    import data.sqlContext.implicits._

    // === BAGGING ====================================================================================================

    if (VERBOSE) println("\nGenerating the training bags:")
    var start_time = System.currentTimeMillis

    // split the training data into positives & negatives
    val positives = data.filter($"label" === 1)
    val negatives = data.except(positives)

    // bagging settings
    // Todo: Tune the bagging parameters.
    val num_bags = 20
    val num_positives = 400
    val num_negatives = 1000

    // calculate the sampling ratios (max out at 100%)
    val positive_frac = num_positives / positives.count.toDouble
    val negative_frac = num_negatives / negatives.count.toDouble

    // create the bags of sampled data
    val bags: Array[DataFrame] = {
      for (i <- 1 to num_bags) yield {
        println("  --- Bag %,d ---".format(i))
        balance(data, num_positives, num_negatives, replacement = true)
      }
    }.toArray

    if (VERBOSE) {
      println("  ---------------------------------------")
      println("  %,d bags containing ~%,d positive and ~%,d negative items".format(num_bags, num_positives, num_negatives))
      println("  Completed in %,.1f seconds".format((System.currentTimeMillis - start_time) / 1000f))
    }

    // === TRAIN ======================================================================================================

    if (VERBOSE) println("\nTraining the models:")
    start_time = System.currentTimeMillis

    // build the learner
    // Todo: Tune the lr_classifier parameters.
    val lr_classifier = new LogisticRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setRegParam(0.3)

    for (i <- bags.indices) {
      if (VERBOSE) println("  Model %d".format(i + 1))

      // train the learner on this bag of data and add to the learner ensemble
      ensemble += lr_classifier.fit(bags(i))
    }

    if (VERBOSE) println("  Completed in %,.1f seconds".format((System.currentTimeMillis - start_time) / 1000f))

    // === PREDICT ====================================================================================================

    DataHandler.save(predict(data, "training"), "predictions_train.csv")

    // return the feature importance dataframe
    // Todo: Access the relative importance for each learner and then average them?
    data.sqlContext.emptyDataFrame
  }

  def predict(data: DataFrame, data_type: String = "testing"): DataFrame = {

    // === PREDICT ====================================================================================================

    if (VERBOSE) println("\nMaking %s predictions:".format(data_type))
    val start_time = System.currentTimeMillis

    // loop over the bags, getting predictions from each
    var predictions: DataFrame = null
    for (i <- ensemble.indices) {
      if (VERBOSE) println("  Model %d".format(i + 1))

      // make the predictions
      var bag_predictions = ensemble(i)
        .transform(data)
        .select("subject_id", "hadm_id", "label", "prediction", "probability")

      // split the probability vector column into columns
      bag_predictions = split_out_vector(bag_predictions, "probability")

      // add this bag's predictions to the previous predictions
      if (predictions == null) {
        // these are the first predictions
        predictions = bag_predictions
      } else {
        // append these predictions to the previous ones
        predictions = predictions.union(bag_predictions)
      }
    }

    // Todo: Set this code up to allow multiclass problems (remove hardcoded "probability_0", etc.).
    val count = lit(ensemble.length)
    predictions = predictions
      .groupBy("subject_id", "hadm_id", "label")
      .agg(round(sum("prediction") / count, 0), sum("probability_0") / count, sum("probability_1") / count)
      .toDF("subject_id", "hadm_id", "label", "prediction", "probability_0", "probability_1")

    if (VERBOSE) println("  Completed in %,.1f seconds".format((System.currentTimeMillis - start_time) / 1000f))

    // === EVALUATE ===================================================================================================

    Metrics.evaluate(predictions, data_type)

    predictions
  }
}
