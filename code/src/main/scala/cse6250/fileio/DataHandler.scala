package cse6250.fileio

import java.io.File
import java.nio.file.{Files, Paths, StandardCopyOption}

import cse6250.Main.{DATA_DIR, NUM_CORES, VERBOSE}
import org.apache.commons.io.FileUtils
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}


/* Saves and loads the merged, cleaned dataset */
object DataHandler {

  def load(spark: SparkSession, file_name: String, data_dir: String = DATA_DIR): DataFrame = {
    if (VERBOSE) println("\nLoading data:\n  " + file_name)
    val start_time = System.currentTimeMillis

    val data: DataFrame =
      if (!Files.exists(Paths.get(data_dir + file_name))) {
        if (VERBOSE) println("  Note: %s is empty".format(file_name))

        // return an empty dataframe
        spark.sqlContext.emptyDataFrame

      } else if (file_name.contains(".csv")) {
        // read a csv
        spark.read
          .option("header", value = true)
          .csv(data_dir + file_name)

      } else {
        // read a parquet file
        spark.read
          .option("header", value = true)
          .parquet(data_dir + file_name)

      }.repartition(NUM_CORES * 4)

    if (VERBOSE) println("  Completed in %,.1f seconds".format((System.currentTimeMillis - start_time) / 1000f))

    data
  }

  def save(data: DataFrame, file_name: String, data_dir: String = DATA_DIR): Unit = {

    if (VERBOSE) println("\nSaving data:\n  " + file_name)
    val start_time = System.currentTimeMillis

    val tempFiles = Files.createTempDirectory(Paths.get(DATA_DIR), "tempFolder").toString

    if (data.rdd.isEmpty()) {
      if (VERBOSE) println("  Note: %s is empty".format(file_name))

      // delete any files from previous runs to avoid confusion
      FileUtils.deleteQuietly(new File(data_dir + file_name))

    } else {
      if (file_name.contains(".csv")) {
        // save to a csv
        data.repartition(1).write
          .mode(SaveMode.Overwrite)
          .option("header", value = true)
          .csv(tempFiles)

      } else if (file_name.contains(".parquet")) {
        // save to a parquet file
        data.repartition(1).write
          .mode(SaveMode.Overwrite)
          .option("header", value = true)
          .parquet(tempFiles)
      }

      // rename the file and move it to the correct place
      val file_path: String = Files.list(Paths.get(tempFiles))
        .toArray
        .filter(_.toString.endsWith(if (file_name.contains(".csv")) ".csv" else ".parquet"))(0)
        .toString

      // move the file to the correct folder
      val from_file = Paths.get(file_path)
      val to_file = Paths.get(data_dir + file_name)
      Files.move(from_file, to_file, StandardCopyOption.REPLACE_EXISTING)

      // remove the folder
      FileUtils.deleteDirectory(from_file.toFile.getParentFile)
    }

    if (VERBOSE) println("  Completed in %,.1f seconds".format((System.currentTimeMillis - start_time) / 1000f))
  }
}
