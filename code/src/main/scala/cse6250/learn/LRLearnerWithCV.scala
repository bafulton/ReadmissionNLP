package cse6250.learn

import cse6250.Main._
import cse6250.Utilities._
import cse6250.evaluate._
import cse6250.fileio.DataHandler
import org.apache.spark.ml.classification.{LogisticRegressionModel, LogisticRegression}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.monotonically_increasing_id


object LRLearnerWithCV extends Learner {
  private var learner: CrossValidatorModel = _
  val learner_loc: String = MODEL_DIR + this.getClass.getSimpleName.replace("$", "")

  def load_learner(): Unit = {
    if (VERBOSE) println("\nLoading trained model:\n  " + learner_loc)
    val start_time = System.currentTimeMillis

    learner = CrossValidatorModel.load(learner_loc)

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

    // build the learner
    val lr_classifier = new LogisticRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")

    // put the learner in a pipeline
    val lrn_pipeline: Pipeline = new Pipeline().setStages(Array(lr_classifier))

    // set up a gridsearch parameter space over the gradient boosted trees learner
    val paramGrid = new ParamGridBuilder()

      // regularization parameter
      .addGrid(lr_classifier.regParam, Array(0.3))

      .build()

    // use cross validation to reduce overfitting
    val cv = new CrossValidator()

      // set the gradient boosted trees learner as the estimator
      .setEstimator(lrn_pipeline)

      // default metric for binary classification evaluation is AUROC
      .setEvaluator(new BinaryClassificationEvaluator)

      // 5-fold cross validation
      .setNumFolds(5)

      // set the gridsearch parameter space
      .setEstimatorParamMaps(paramGrid)

      // evaluate up to 2 parameter settings in parallel
      .setParallelism(2)

      // set the random seed (for reproducibility)
      .setSeed(RANDOM_SEED)

    // train the learner
    learner = cv.fit(balanced)

    // get the importance of each feature (use the best model)
    val feature_importance: DataFrame = learner
      .bestModel.asInstanceOf[PipelineModel]
      .stages(0).asInstanceOf[LogisticRegressionModel]
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

    println("  Completed in %,.1f seconds".format((System.currentTimeMillis - start_time) / 1000f))

    // add a separate column for each label's probability
    predictions = split_out_vector(predictions, col_name = "probability")

    // === EVALUATE ===================================================================================================

    Metrics.evaluate(predictions, data_type)

    predictions
  }
}
