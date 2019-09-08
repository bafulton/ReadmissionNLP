package cse6250.preprocess

import com.johnsnowlabs.nlp.annotators.{Normalizer, Tokenizer}
import com.johnsnowlabs.nlp.base.{DocumentAssembler, Finisher}
import cse6250.Main.VERBOSE
import org.apache.spark.ml.feature._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.monotonically_increasing_id

object TFIDF extends Preprocessor {

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
      .stages(5).asInstanceOf[CountVectorizerModel]

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
      .drop("text", "fin", "rem", "count")

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

      // finish (Finisher is a Spark-NLP function to clean up the output)
      new Finisher().setInputCols("norm").setOutputCols("fin"),

      // remove stop words
      new StopWordsRemover().setInputCol("fin").setOutputCol("rem"),

      // count word frequencies (only keep the 3k most frequent words)
      // Note: Could also use HashingTF(). Can't set its vocab size, though.
      new CountVectorizer().setInputCol("rem").setOutputCol("count").setVocabSize(3000),

      // run TF-IDF
      new IDF().setInputCol("count").setOutputCol("features")
    ))
  }
}
