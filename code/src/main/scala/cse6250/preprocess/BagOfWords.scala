package cse6250.preprocess

import com.johnsnowlabs.nlp.annotators.{Normalizer, Stemmer, Tokenizer}
import com.johnsnowlabs.nlp.base.{DocumentAssembler, Finisher}
import cse6250.Main.VERBOSE
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, StopWordsRemover}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._


object BagOfWords extends Preprocessor {

  protected var model: PipelineModel = _

  def fit(data: DataFrame): DataFrame = {
    import data.sqlContext.implicits._

    if (VERBOSE) println("\nPreparing the featurizer:")
    val start_time = System.currentTimeMillis

    // train the featurizer pipeline
    model = pipeline.fit(data)

    // return a dataframe mapping feature indices to values (words)
    val feature_map = model
      // the last stage in the model is the CountVectorizer
      .stages.last.asInstanceOf[CountVectorizerModel]

      // get the vocabulary
      .vocabulary

      // convert to a dataframe
      .toList.toDF("value")

      // add the feature id
      .withColumn("feature", monotonically_increasing_id)

      // rearrange columns
      .select("feature", "value")

    if (VERBOSE) println("  Completed in %,.1f seconds".format((System.currentTimeMillis - start_time) / 1000f))

    feature_map
  }

  def featurize(data: DataFrame, data_type: String = "testing"): DataFrame = {
    if (VERBOSE) println("\nFeaturizing the %s data:".format(data_type))
    val start_time = System.currentTimeMillis

    val featurized = model.transform(data)
      // drop unnecessary intermediary columns
      .drop("text", "fin", "rem")

    if (VERBOSE) println("  Completed in %,.1f seconds".format((System.currentTimeMillis - start_time) / 1000f))

    featurized
  }

  protected val pipeline: Pipeline = {
    new Pipeline().setStages(Array(

      new DocumentAssembler().setInputCol("text").setOutputCol("doc"),

      // tokenize (break text into words)
      new Tokenizer().setInputCols("doc").setOutputCol("tok"),

      // normalize (remove punctuation/other non-character values)
      new Normalizer().setInputCols("tok").setOutputCol("norm"),

      // stem (remove the endings from words)
      new Stemmer().setInputCols("norm").setOutputCol("stem"),

      // finish (Finisher is a Spark-NLP function to clean up the output)
      new Finisher().setInputCols("stem").setOutputCols("fin"),

      // remove stop words
      new StopWordsRemover().setInputCol("fin").setOutputCol("rem"),

      // vectorize
      new CountVectorizer().setInputCol("rem").setOutputCol("features").setVocabSize(3000)
    ))
  }
}
