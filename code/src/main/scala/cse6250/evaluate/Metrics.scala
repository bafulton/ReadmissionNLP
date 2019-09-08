package cse6250.evaluate

import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame

object Metrics {

  /* Central evaluate function used for all learners--calculates the various metrics. */
  def evaluate(predictions: DataFrame, predictions_type: String = "testing"): Unit = {

    // === DATA MANIPULATION ==========================================================================================
    /*
    The results DataFrame has the following structure:
    root
     |-- subject_id: integer
     |-- hadm_id: integer
     |-- label: double
     |-- prediction: double
     |-- probability_0: double
     |-- ...
     |-- probability_n: double

    In most places, we use the org.apache.spark.ml package instead of org.apache.spark.mllib, as it is newer and
    has better support for dataframes. In this case, the current ml.evaluation.MulticlassClassificationEvaluator
    object does not provide full support for per-label metrics, while mllib.evaluation.MulticlassMetrics does.
    As such, we shall convert our input dataframe into an RDD consisting of a tuple of (prediction, label).

    Additionally, note that this function does not accept any threshold parameters. Instead, each learner has a
    setThresholds() function that should be used to set the label thresholds.
    */

    import predictions.sqlContext.implicits._
    val predictions_rdd: RDD[(Double, Double)] = predictions

      // we just want the (hard) prediction and the label
      .select("prediction", "label")

      // map to a tuple (prediction, label)
      .map(row => (row.getDouble(0), row.getDouble(1)))

      // convert to an RDD
      .rdd

    // === METRICS ====================================================================================================
    // refer to https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers

    println("\nEvaluating %s predictions:".format(predictions_type))
    val start_time = System.currentTimeMillis

    // instantiate the metrics object
    val metrics = new MulticlassMetrics(predictions_rdd)

    // confusion matrix (custom print code--makes it human readable)
    println("  Confusion matrix:")
    var header = "           "
    metrics.labels.foreach(l => header += "%9s".format("pred_%.0f".format(l)))
    println(header)
    var i: Int = 0
    for (row <- metrics.confusionMatrix.rowIter) {
      print("    %7s".format("true_%d".format(i)))
      row.foreachActive { (_: Int, item: Double) =>
        print("%,9.0f".format(item))
      }
      print("\n")
      i += 1
    }

    // accuracy
    println("  Accuracy: %.3f".format(metrics.accuracy))

    // precision
    println("  Precision:")
    metrics.labels.foreach { l =>
      println("    Label %.0f: %.3f".format(l, metrics.precision(l)))
    }
    println("    Overall: %.3f".format(metrics.weightedPrecision))

    // recall
    println("  Recall:")
    metrics.labels.foreach { l =>
      println("    Label %.0f: %.3f".format(l, metrics.recall(l)))
    }
    println("    Overall: %.3f".format(metrics.weightedRecall))

    // f-score
    println("  F-score:")
    metrics.labels.foreach { l =>
      println("    Label %.0f: %.3f".format(l, metrics.fMeasure(l)))
    }
    println("    Overall: %.3f".format(metrics.weightedFMeasure))

    // auroc (using soft labels)
    // Todo: AUROC for all but label_1 are wrong; need to fix.
    println("  AUROC (currently only label_1 works):")
    metrics.labels.foreach { l =>
      // make a binary RDD containing (soft_prob, label) tuples
      val binary_rdd = predictions
        .select("probability_%.0f".format(l), "label")
        .map(row => (
          row.getDouble(0),
          row.getDouble(1)
        ))
        .rdd

      // instantiate a binary metrics object
      val binary_metrics = new BinaryClassificationMetrics(binary_rdd)

      println("    Label %.0f: %.3f".format(l, binary_metrics.areaUnderROC))
    }

    println("  Completed in %,.1f seconds".format((System.currentTimeMillis - start_time) / 1000f))
  }
}
