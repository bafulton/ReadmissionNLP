package cse6250.preprocess

import com.johnsnowlabs.nlp.annotators.{Normalizer, Stemmer, Tokenizer}
import com.johnsnowlabs.nlp.base.{DocumentAssembler, Finisher}
import cse6250.Main.VERBOSE
import org.apache.spark.ml.feature.{StopWordsRemover, Word2Vec}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame


object Word2Vector extends Preprocessor {

  protected var model: PipelineModel = _

  def fit(data: DataFrame): DataFrame = {

    if (VERBOSE) println("\nPreparing the featurizer:")
    val start_time = System.currentTimeMillis

    // train the featurizer pipeline
    model = pipeline.fit(data)

    if (VERBOSE) println("  Completed in %,.1f seconds".format((System.currentTimeMillis - start_time) / 1000f))

    // return an empty feature map (I don't think word2vec supports it)
    // Todo: Confirm word2vec can't provide a feature map.
    data.sqlContext.emptyDataFrame
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

      // word2vec vectorizer
      new Word2Vec().setInputCol("rem").setOutputCol("features")
    ))
  }
}
