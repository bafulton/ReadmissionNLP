package cse6250

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{Column, DataFrame}

package object Utilities {

  /* Breaks a column containing a vector into n columns, one for each item in the vector. */
  def split_out_vector(data: DataFrame, col_name: String): DataFrame = {

    //if (VERBOSE) println("\nConverting vector to columns:")
    val start_time = System.currentTimeMillis

    // Code courtesy of: https://stackoverflow.com/questions/49911608/
    // scala-spark-split-vector-column-into-separate-columns-in-a-spark-dataframe

    // user-defined function to convert a Vector to an array
    def to_array: UserDefinedFunction = udf {
      x: Vector => x.toArray
    }

    // convert Vector column to array column
    var data_ret: DataFrame = data
      .withColumn(col_name, to_array(col(col_name)))

    // create an array of the new column names
    val num_cols: Int = data.select(col_name).first.getAs[Vector](0).size
    val col_names: Array[String] = (for (i <- 0 until num_cols) yield "%s_%d".format(col_name, i)).toArray

    // turn the array column into an array of columns
    val sqlExpr: Array[Column] = col_names.zipWithIndex.map{
      case (alias, idx) => col(col_name).getItem(idx).as(alias)
    }

    // merge the new columns into the original dataframe
    data_ret = data_ret.select(col("*") +: sqlExpr :_*)

    //if (VERBOSE) println("  Completed in %,.1f seconds".format((System.currentTimeMillis - start_time) / 1000f))

    data_ret.drop(col_name)
  }
}
