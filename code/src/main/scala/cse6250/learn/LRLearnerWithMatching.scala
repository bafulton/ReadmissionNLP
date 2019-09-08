package cse6250.learn

import cse6250.Main._
import cse6250.Utilities._
import cse6250.evaluate.Metrics
import cse6250.fileio.DataHandler
import cse6250.preprocess.split.RatioSplit
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.monotonically_increasing_id


object LRLearnerWithMatching extends Learner {
  private var learner: LogisticRegressionModel = _
  val learner_loc: String = MODEL_DIR + this.getClass.getSimpleName.replace("$", "")

  def load_learner(): Unit = {
    if (VERBOSE) println("\nLoading trained model:\n  " + learner_loc)
    val start_time = System.currentTimeMillis

    learner = LogisticRegressionModel.load(learner_loc)

    if (VERBOSE) println("  Completed in %,.1f seconds".format((System.currentTimeMillis - start_time) / 1000f))
  }

  def save_learner(): Unit = {
    if (VERBOSE) println("\nSaving trained model:\n  " + learner_loc)
    val start_time = System.currentTimeMillis

    learner.save(learner_loc)

    if (VERBOSE) println("  Completed in %,.1f seconds".format((System.currentTimeMillis - start_time) / 1000f))
  }

  def train(data: DataFrame): DataFrame = {
    import data.sqlContext.implicits._

    // === BALANCE ===================================================================================================

    if (VERBOSE) println("\nBalancing the training data:")
    var start_time = System.currentTimeMillis

    // sample the negatives so they're the same size as the positives
    val pos_count = data.filter($"label" === 1).count
    val balanced = balance(data, pos_count, pos_count, replacement = false)

    if (VERBOSE) println("  Completed in %,.1f seconds".format((System.currentTimeMillis - start_time) / 1000f))

    // === TRAIN ======================================================================================================

    if (VERBOSE) println("\nTraining the model:")
    start_time = System.currentTimeMillis

    // train the learner
    learner = new LogisticRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setRegParam(0.3)
      .fit(balanced)

    // get the importance of each feature
    val feature_importance: DataFrame = learner
      .coefficients
      .toArray.toList.toDF("importance")
      .withColumn("feature", monotonically_increasing_id)
      .select("feature", "importance")

    if (VERBOSE) println("  Completed in %,.1f seconds".format((System.currentTimeMillis - start_time) / 1000f))

    // === PREDICT ====================================================================================================

    DataHandler.save(predict(balanced, "training"), "predictions_train.csv")

    // return the feature importance dataframe
    feature_importance
  }

  def predict(data: DataFrame, data_type: String = "testing"): DataFrame = {

    // === PREDICT ====================================================================================================

    println("\nMaking %s predictions:".format(data_type))
    val start_time = System.currentTimeMillis

    // make predictions
    var predictions: DataFrame = learner.transform(data)
      .select("subject_id", "hadm_id", "label", "prediction", "probability")

    // add a separate column for each label's probability
    predictions = split_out_vector(predictions, col_name = "probability")

    println("  Completed in %,.1f seconds".format((System.currentTimeMillis - start_time) / 1000f))

    // === EVALUATE ===================================================================================================

    Metrics.evaluate(predictions, data_type)

    predictions
  }
}
