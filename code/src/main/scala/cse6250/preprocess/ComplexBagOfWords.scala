package cse6250.preprocess

import com.johnsnowlabs.nlp.annotator.SentimentDetector
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.annotators.spell.symmetric.SymmetricDeleteApproach
import com.johnsnowlabs.nlp.annotators.{Normalizer, Tokenizer}
import com.johnsnowlabs.nlp.base.{DocumentAssembler, Finisher}
import cse6250.Main.VERBOSE
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, StopWordsRemover}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

object ComplexBagOfWords extends Preprocessor {

  private val DICT_LOC = "../dictionaries/"

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
      //.drop("text", "fin", "rem")

    featurized.show(50)

    if (VERBOSE) println("  Completed in %,.1f seconds".format((System.currentTimeMillis - start_time) / 1000f))

    featurized
  }

  protected val pipeline: Pipeline = {
    new Pipeline().setStages(Array(

      new DocumentAssembler().setInputCol("text").setOutputCol("document"),

      new SentenceDetector().setInputCols("document").setOutputCol("sentence"),

      new Tokenizer().setInputCols("sentence").setOutputCol("token"),

      // normalize (remove punctuation/other non-character values)
      new Normalizer().setInputCols("token").setOutputCol("normal"),

      // spellcheck (can use either the Norvig-Sweeting approach or the Symmetric Delete approach;
      // see https://nlp.johnsnowlabs.com/components.html for details)
      // medical dictionary courtesy of: https://github.com/glutanimate/wordlist-medicalterms-en
      // english dictionary courtesy of: https://github.com/dwyl/english-words
      new SymmetricDeleteApproach().setInputCols("normal").setOutputCol("spell")
        .setDictionary(DICT_LOC + "combined_dict.txt"),

      new SentimentDetector().setInputCols(Array("spell", "sentence")).setOutputCol("sentiment"),

      // stem (remove the endings from words)
      //new Stemmer().setInputCols("spell").setOutputCol("stem"),

      // lemmatize (convert words to their base concepts)
      // Todo: Add lemmatizer (need a corpus of medical concepts first...)

      // finish (Finisher is a Spark-NLP function to clean up the output)
      new Finisher().setInputCols("sentiment").setOutputCols("finish"),

      // remove stop words (use the default list for now)
      /*val stop_words = Array("the", "and", "to", "of", "was", "with", "a", "on", "in", "for", "name", "is",
        "patient", "s", "he", "at", "as", "or", "one", "she", "his", "her", "am", "were", "you", "pt", "pm", "by",
        "be", "had", "your", "this", "date", "from", "there", "an", "that", "p", "are", "have", "has", "h", "but",
        "o", "namepattern", "which", "every", "also")*/
      new StopWordsRemover().setInputCol("finish").setOutputCol("removed"),

      // vectorize (only keep the 1000 most frequent words)
      new CountVectorizer().setInputCol("removed").setOutputCol("features").setVocabSize(1000)
    ))
  }
}
