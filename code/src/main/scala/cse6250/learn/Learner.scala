package cse6250.learn

import cse6250.Main.{NUM_CORES, RANDOM_SEED, VERBOSE}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.rand


trait Learner extends Serializable {

  // save/load models to/from disk
  val learner_loc: String
  def load_learner(): Unit
  def save_learner(): Unit

  // balance the positive/negative items
  def balance(data: DataFrame, num_pos: Long, num_neg: Long, replacement: Boolean = true): DataFrame = {
    import data.sqlContext.implicits._

    // split the training data into positives & negatives
    val positives = data.filter($"label" === 1)
    val negatives = data.except(positives)

    // sample the negatives so they're the same size as the positives
    val pos_sample = positives.sample(withReplacement = replacement, num_pos / positives.count.toFloat, RANDOM_SEED)
    val neg_sample = negatives.sample(withReplacement = replacement, num_neg / negatives.count.toFloat, RANDOM_SEED)

    // merge positive and negative samples and shuffle
    val balanced = pos_sample.union(neg_sample)
      .orderBy(rand())
      .repartition(NUM_CORES * 4)

    if (VERBOSE) {
      println("  %,d positive training items (chosen from %,d)".format(pos_sample.count, positives.count))
      println("  %,d negative training items (chosen from %,d)".format(neg_sample.count, negatives.count))
      println("  %,d total training items".format(balanced.count))
    }

    balanced
  }

  def train(data: DataFrame): DataFrame

  def predict(data: DataFrame, data_type: String = "testing"): DataFrame
}
