package cse6250.preprocess

import com.johnsnowlabs.nlp.annotators.{Normalizer, Stemmer, Tokenizer}
import com.johnsnowlabs.nlp.base.{DocumentAssembler, Finisher}
import cse6250.Main.VERBOSE
import org.apache.spark.ml.feature._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._


object SectionWiseBOW extends Preprocessor {

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
//      .drop("text", "fin", "rem", "1_grams", "2_grams", "3_grams", "1_counts", "2_counts", "3_counts")
      .drop("text","discharge_condition","discharge_disposition","discharge_diagnosis","discharge_medication","discharge_instructions")

    if (VERBOSE) println("  Completed in %,.1f seconds".format((System.currentTimeMillis - start_time) / 1000f))

    featurized
  }

  protected val pipeline: Pipeline = {
    new Pipeline().setStages(Array(

      new DocumentAssembler().setInputCol("text").setOutputCol("doc1"),
      new DocumentAssembler().setInputCol("discharge_condition").setOutputCol("doc2"),
      new DocumentAssembler().setInputCol("discharge_disposition").setOutputCol("doc3"),
      new DocumentAssembler().setInputCol("discharge_diagnosis").setOutputCol("doc4"),
      new DocumentAssembler().setInputCol("discharge_medication").setOutputCol("doc5"),
      new DocumentAssembler().setInputCol("discharge_instructions").setOutputCol("doc6"),

      // tokenize (break text into words)
      new Tokenizer().setInputCols("doc1").setOutputCol("tok1"),
      new Tokenizer().setInputCols("doc2").setOutputCol("tok2"),
      new Tokenizer().setInputCols("doc3").setOutputCol("tok3"),
      new Tokenizer().setInputCols("doc4").setOutputCol("tok4"),
      new Tokenizer().setInputCols("doc5").setOutputCol("tok5"),
      new Tokenizer().setInputCols("doc6").setOutputCol("tok6"),

      // normalize (remove punctuation/other non-character values)
      new Normalizer().setInputCols("tok1").setOutputCol("norm1"),
      new Normalizer().setInputCols("tok2").setOutputCol("norm2"),
      new Normalizer().setInputCols("tok3").setOutputCol("norm3"),
      new Normalizer().setInputCols("tok4").setOutputCol("norm4"),
      new Normalizer().setInputCols("tok5").setOutputCol("norm5"),
      new Normalizer().setInputCols("tok6").setOutputCol("norm6"),

      // stem (remove the endings from words)
      new Stemmer().setInputCols("norm1").setOutputCol("stem1"),
      new Stemmer().setInputCols("norm2").setOutputCol("stem2"),
      new Stemmer().setInputCols("norm3").setOutputCol("stem3"),
      new Stemmer().setInputCols("norm4").setOutputCol("stem4"),
      new Stemmer().setInputCols("norm5").setOutputCol("stem5"),
      new Stemmer().setInputCols("norm6").setOutputCol("stem6"),

//      // finish (Finisher is a Spark-NLP function to clean up the output)
//      new Finisher().setInputCols("stem1").setOutputCols("fin1"),
//      new Finisher().setInputCols("stem2").setOutputCols("fin2"),
//      new Finisher().setInputCols("stem3").setOutputCols("fin3"),
//      new Finisher().setInputCols("stem4").setOutputCols("fin4"),
//      new Finisher().setInputCols("stem5").setOutputCols("fin5"),
//      new Finisher().setInputCols("stem6").setOutputCols("fin6"),

      // remove stop words
      new StopWordsRemover().setInputCol("stem1").setOutputCol("rem1"),
      new StopWordsRemover().setInputCol("stem2").setOutputCol("rem2"),
      new StopWordsRemover().setInputCol("stem3").setOutputCol("rem3"),
      new StopWordsRemover().setInputCol("stem4").setOutputCol("rem4"),
      new StopWordsRemover().setInputCol("stem5").setOutputCol("rem5"),
      new StopWordsRemover().setInputCol("stem6").setOutputCol("rem6"),

      // n-grams for n = 1 and 2
      new NGram().setN(1).setInputCol("rem1").setOutputCol("1_grams1"),
      new NGram().setN(2).setInputCol("rem1").setOutputCol("2_grams1"),

      new NGram().setN(1).setInputCol("rem2").setOutputCol("1_grams2"),
      new NGram().setN(2).setInputCol("rem2").setOutputCol("2_grams2"),

      new NGram().setN(1).setInputCol("rem3").setOutputCol("1_grams3"),
      new NGram().setN(2).setInputCol("rem3").setOutputCol("2_grams3"),

      new NGram().setN(1).setInputCol("rem4").setOutputCol("1_grams4"),
      new NGram().setN(2).setInputCol("rem4").setOutputCol("2_grams4"),

      new NGram().setN(1).setInputCol("rem5").setOutputCol("1_grams5"),
      new NGram().setN(2).setInputCol("rem5").setOutputCol("2_grams5"),

      new NGram().setN(1).setInputCol("rem6").setOutputCol("1_grams6"),
      new NGram().setN(2).setInputCol("rem6").setOutputCol("2_grams6"),

      // vectorize
      new CountVectorizer().setInputCol("1_grams1").setOutputCol("1_counts1").setVocabSize(2000),
      new CountVectorizer().setInputCol("2_grams1").setOutputCol("2_counts1").setVocabSize(1000),

      new CountVectorizer().setInputCol("1_grams2").setOutputCol("1_counts2").setVocabSize(2000),
      new CountVectorizer().setInputCol("2_grams2").setOutputCol("2_counts2").setVocabSize(1000),

      new CountVectorizer().setInputCol("1_grams3").setOutputCol("1_counts3").setVocabSize(2000),
      new CountVectorizer().setInputCol("2_grams3").setOutputCol("2_counts3").setVocabSize(1000),

      new CountVectorizer().setInputCol("1_grams4").setOutputCol("1_counts4").setVocabSize(2000),
      new CountVectorizer().setInputCol("2_grams4").setOutputCol("2_counts4").setVocabSize(1000),

      new CountVectorizer().setInputCol("1_grams5").setOutputCol("1_counts5").setVocabSize(2000),
      new CountVectorizer().setInputCol("2_grams5").setOutputCol("2_counts5").setVocabSize(1000),

      new CountVectorizer().setInputCol("1_grams6").setOutputCol("1_counts6").setVocabSize(2000),
      new CountVectorizer().setInputCol("2_grams6").setOutputCol("2_counts6").setVocabSize(1000),

      // Assemble the count vectors
      new VectorAssembler().setInputCols(Array("1_counts1","2_counts1","1_counts2","2_counts2","1_counts3","2_counts3","1_counts4","2_counts4","1_counts5","2_counts5","1_counts6","2_counts6")).setOutputCol("features")
    ))
  }
}
