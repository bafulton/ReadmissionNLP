lazy val projectName = "CSE6250-NLP"
lazy val projectVersion = "0.1.0"
lazy val projectScalaVersion = "2.11.12"
lazy val sparkVersion = "2.3.2"
lazy val sparkNLPVersion = "1.6.3"

name := projectName
version := projectVersion
scalaVersion := projectScalaVersion

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "com.johnsnowlabs.nlp" %% "spark-nlp" % sparkNLPVersion
)
