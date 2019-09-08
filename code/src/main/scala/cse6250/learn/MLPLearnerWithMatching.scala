package cse6250.learn

import cse6250.Main._
import cse6250.Utilities._
import cse6250.evaluate._
import cse6250.fileio.DataHandler
import cse6250.preprocess.split.RatioSplit
import org.apache.spark.ml.classification.{MultilayerPerceptronClassificationModel, MultilayerPerceptronClassifier}
import org.apache.spark.sql.DataFrame


object MLPLearnerWithMatching extends Learner {
  private var learner: MultilayerPerceptronClassificationModel = _
  val learner_loc: String = MODEL_DIR + this.getClass.getSimpleName.replace("$", "")

  def load_learner(): Unit = {
    if (VERBOSE) println("\nLoading trained model:\n  " + learner_loc)
    val start_time = System.currentTimeMillis

    learner = MultilayerPerceptronClassificationModel.load(learner_loc)

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

    // specify layers for the neural network:
    // input layer of size 3000 (features), two intermediate of size 5 and 4
    // and output of size 2 (classes)
    val layers = Array[Int](3000, 5, 4, 2) // Todo: Use the input feature size as a variable, remove hard coding

    // create the trainer and set its parameters
    learner = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(RANDOM_SEED)
      .setMaxIter(100)
        .fit(balanced)

    if (VERBOSE) println("  Completed in %,.1f seconds".format((System.currentTimeMillis - start_time) / 1000f))

    // === PREDICT ====================================================================================================

    DataHandler.save(predict(balanced, "training"), "predictions_train.csv")

    // return a dataframe listing the importance of each feature
    // since MLPs can't easily calculate that, just return an empty dataframe for now
    data.sqlContext.emptyDataFrame
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
