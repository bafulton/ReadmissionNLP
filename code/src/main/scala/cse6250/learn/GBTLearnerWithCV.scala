package cse6250.learn

import cse6250.Main._
import cse6250.Utilities._
import cse6250.evaluate._
import cse6250.fileio.DataHandler
import cse6250.preprocess.split.RatioSplit
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.monotonically_increasing_id


object GBTLearnerWithCV extends Learner {
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
    val gbt_classifier: GBTClassifier = new GBTClassifier()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setSeed(RANDOM_SEED)

    // put the learner in a pipeline
    val lrn_pipeline: Pipeline = new Pipeline().setStages(Array(gbt_classifier))

    // set up a gridsearch parameter space over the gradient boosted trees learner
    val paramGrid = new ParamGridBuilder()

      // information gain calculation (default="gini")
      //.addGrid(gbt_classifier.impurity, Array("gini", "entropy"))

      // step size/learning rate (default=0.1)
      //.addGrid(gbt_classifier.stepSize, Array(0.01, 0.1, 0.15, 0.2, 0.5))

      // max depth of each tree (default=5)
      //.addGrid(gbt_classifier.maxDepth, Array(1, 2, 3, 5, 10))

      // minimum number of instances each child must have after a split (default=1)
      //.addGrid(gbt_classifier.minInstancesPerNode, Array(1, 3, 5, 10, 25))

      // maximum number of iterations (default=10)
      //.addGrid(gbt_classifier.maxIter, Array(2, 5, 10, 20))
      .addGrid(gbt_classifier.maxIter, Array(10))

      // number of features to consider for splits at each tree node (default="auto")
      //.addGrid(gbt_classifier.featureSubsetStrategy, Array("all", "sqrt", "log2", "10", "50", "100", "500"))
      .addGrid(gbt_classifier.featureSubsetStrategy, Array("auto"))

      // fraction of the training data to use training each individual tree
      //.addGrid(gbt_classifier.subsamplingRate, Array(0.01, 0.05, 0.1, 0.25, 0.5, 1.0))

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
      .stages(0).asInstanceOf[GBTClassificationModel]
      .featureImportances
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
