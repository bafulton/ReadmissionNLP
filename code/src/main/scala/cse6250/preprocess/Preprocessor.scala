package cse6250.preprocess

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame

trait Preprocessor {
  // user-defined processing pipeline
  protected val pipeline: Pipeline

  // pipeline is saved as a pipeline model after it has been fit
  protected var model: PipelineModel

  // trains the pipeline
  def fit(data: DataFrame): DataFrame

  // transforms the data (tokenize, vectorize, etc.)
  def featurize(data: DataFrame, data_type: String = "testing"): DataFrame
}
