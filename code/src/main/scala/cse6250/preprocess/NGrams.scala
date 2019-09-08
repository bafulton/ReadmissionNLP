package cse6250.preprocess

import com.johnsnowlabs.nlp.annotators.{Normalizer, Stemmer, Tokenizer}
import com.johnsnowlabs.nlp.base.{DocumentAssembler, Finisher}
import cse6250.Main.VERBOSE
import org.apache.spark.ml.feature._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._


object NGrams extends Preprocessor {

  protected var model: PipelineModel = _

  def fit(data: DataFrame): DataFrame = {
    import data.sqlContext.implicits._

    if (VERBOSE) println("\nPreparing the featurizer:")
    val start_time = System.currentTimeMillis

    // train the featurizer pipeline
    model = pipeline.fit(data)

//    model.transform(data).show(3,false)

    // return a dataframe mapping feature indices to values (words)
    /*val feature_map = model
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

    feature_map*/

    // return a dataframe listing the importance of each feature
    // since VectorAssembler can't easily calculate that, just return an empty dataframe for now
    data.sqlContext.emptyDataFrame
  }

  def featurize(data: DataFrame, data_type: String = "testing"): DataFrame = {
    if (VERBOSE) println("\nFeaturizing the %s data:".format(data_type))
    val start_time = System.currentTimeMillis

    val featurized = model.transform(data)
      // drop unnecessary intermediary columns
      .drop("text", "fin", "rem", "1_grams", "2_grams", "1_counts", "2_counts")

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

      // n-grams for n = 1 and 2
      new NGram().setN(1).setInputCol("rem").setOutputCol("1_grams"),
      new NGram().setN(2).setInputCol("rem").setOutputCol("2_grams"),

      // vectorize
      new CountVectorizer().setInputCol("1_grams").setOutputCol("1_counts").setVocabSize(3000),
      new CountVectorizer().setInputCol("2_grams").setOutputCol("2_counts").setVocabSize(1000),

      // Assemble the count vectors
      new VectorAssembler().setInputCols(Array("1_counts","2_counts")).setOutputCol("features")
    ))
  }
}
