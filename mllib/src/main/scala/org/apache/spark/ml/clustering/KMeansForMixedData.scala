/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.clustering

import scala.collection.mutable

import org.apache.spark.SparkException
import org.apache.spark.annotation.{Experimental, Since}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.linalg.{SparseMatrix, Vector, VectorUDT}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util._
import org.apache.spark.mllib.clustering.{KMeans => MLlibKMeans, KMeansModel => MLlibKMeansModel}
import org.apache.spark.mllib.linalg.{DenseMatrix, Vector => OldVector, Vectors => OldVectors}
import org.apache.spark.mllib.linalg.VectorImplicits._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.{IntegerType, StructType}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.VersionUtils.majorVersion




/**
 * Common params for KMeansForMixedData and KMeansForMixedDataModel
 */
private[clustering] trait KMeansForMixedDataParams
  extends Params with HasMaxIter with HasFeaturesCol
  with HasSeed with HasPredictionCol with HasTol {

  /**
   * The number of clusters to create (k). Must be &gt; 1. Note that it is possible for fewer than
   * k clusters to be returned, for example, if there are fewer than k distinct points to cluster.
   * Default: 2.
   * @group param
   */
  @Since("1.5.0")
  final val k = new IntParam(this, "k", "The number of clusters to create. " +
    "Must be > 1.", ParamValidators.gt(1))

  /** @group getParam */
  @Since("1.5.0")
  def getK: Int = $(k)

  /** @group param */
  final val Interval = new IntParam(this, "interval", "The number of intervals to " +
    "descretize numeric feature", ParamValidators.gt(1))

  /** @group getParam */
  def getInterval: Int = $(Interval)

  /** @group param */
  final val occurrences = new Param[Map[String, Array[(Double, Long)]]](this, "occurrence",
    "The occurrence of an exact value of a feature.")

  /** @group getParam */
  def getOccurrences: Map[String, Array[(Double, Long)]] = $(occurrences)

  /** @group param */
  final val inputQualitativeCols = new Param[Array[String]](this, "qualitative column names",
    "qualitative column names")

  /** @group param */
  final val inputQuantitativeCols = new Param[Array[String]](this, "quantitative column names",
    "quantitative column names")

  /** @group setParam */
  def setInputQualitativeCols(values: Array[String]): this.type = set(inputQualitativeCols, values)

  /** @group setParam */
  def setInputQuantitativeCols(values: Array[String]): this.type =
    set(inputQuantitativeCols, values)

  /**
   * Param for the initialization algorithm. Now only have random initialization.
   * @group expertParam
   */
  @Since("1.5.0")
  final val initMode = new Param[String](this, "initMode", "The initialization algorithm. " +
    "Supported options: 'random'.",
    (value: String) => MLlibKMeans.validateInitMode(value))

  /** @group expertGetParam */
  @Since("1.5.0")
  def getInitMode: String = $(initMode)

  /**
   * Param for the number of steps for the k-means|| initialization mode. This is an advanced
   * setting -- the default of 2 is almost always enough. Must be &gt; 0. Default: 2.
   * @group expertParam
   */
  @Since("1.5.0")
  final val initSteps = new IntParam(this, "initSteps", "The number of steps for k-means|| " +
    "initialization mode. Must be > 0.", ParamValidators.gt(0))

  /** @group expertGetParam */
  @Since("1.5.0")
  def getInitSteps: Int = $(initSteps)

  /**
   * Validates and transforms the input schema.
   * @param schema input schema
   * @return output schema
   */
  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, $(featuresCol), new VectorUDT)
    SchemaUtils.appendColumn(schema, $(predictionCol), IntegerType)
  }
}

class KMeansForMixedDataModel extends Model[KMeansForMixedDataModel]
  with KMeansForMixedDataParams {
  /**
   * An immutable unique ID for the object and its derivatives.
   */
  override val uid: String = Identifiable.randomUID("kMeansForMixedDataModel")
  override def copy(extra: ParamMap): KMeansForMixedDataModel = defaultCopy(extra)

  /**
   * Transforms the input dataset.
   */
  override def transform(dataset: Dataset[_]): DataFrame = dataset.toDF()

  /**
   * :: DeveloperApi ::
   *
   * Check transform validity and derive the output schema from the input schema.
   *
   * We check validity for interactions between parameters during `transformSchema` and
   * raise an exception if any parameter value is invalid. Parameter value checks which
   * do not depend on other parameters are handled by `Param.validate()`.
   *
   * Typical implementation should first conduct verification on schema change and parameter
   * validity, including complex parameter interaction checks.
   */
  override def transformSchema(schema: StructType): StructType = schema


}

/**
 * Created by junlang on 7/27/17.
 * KMeans for mixed data
 */
class KMeansForMixedData extends Estimator[KMeansForMixedDataModel]
  with KMeansForMixedDataParams with DefaultParamsWritable {
  /**
   * An immutable unique ID for the object and its derivatives.
   */
  override val uid: String = Identifiable.randomUID("kMeansForMixedData")

  /** @group setParam */
  @Since("1.5.0")
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  /**
   * TODO: return type to be KMeansModel. And make it similar to KMeans fit().
   * @param dataset
   */
  def fit(dataset: Dataset[_]): KMeansForMixedDataModel = {
    // TODO: transformSchema

    val handlePersistence = dataset.rdd.getStorageLevel == StorageLevel.NONE
    val instances: RDD[OldVector] = dataset.select(col($(featuresCol))).rdd.map {
      case Row(point: Vector) => OldVectors.fromML(point)
    }

    if (handlePersistence) {
      instances.persist(StorageLevel.MEMORY_AND_DISK)
    }
    val instr = Instrumentation.create(this, instances)
    instr.logParams(featuresCol, predictionCol, k, initMode, initSteps, maxIter, seed, tol)

    val (coOccurrences, significances) = coOccurrencesAndSignificances(dataset)

    val featureIndexes = getFeatureIndexes(dataset, $(inputQualitativeCols))
    //
    var acc = 0
    val indices = for (ele <- featureIndexes ) yield { acc = acc + ele.length; acc }


    new KMeansForMixedDataModel()
  }

  /**
   * Creates a copy of this instance with the same UID and some extra params.
   * Subclasses should implement this method and set the return type properly.
   * See `defaultCopy()`.
   */
  override def copy(extra: ParamMap): KMeansForMixedData = defaultCopy(extra)

  /**
   * :: DeveloperApi ::
   *
   * Check transform validity and derive the output schema from the input schema.
   *
   * We check validity for interactions between parameters during `transformSchema` and
   * raise an exception if any parameter value is invalid. Parameter value checks which
   * do not depend on other parameters are handled by `Param.validate()`.
   *
   * Typical implementation should first conduct verification on schema change and parameter
   * validity, including complex parameter interaction checks.
   */
  override def transformSchema(schema: StructType): StructType = schema

  /**
   * For convenience, compute the co-occurrence and significances here.
   */
  def coOccurrencesAndSignificances(
      dataset: Dataset[_]
      ): (Array[SparseMatrix], Array[Double]) = {
    val columns = $(inputQualitativeCols) ++ $(inputQuantitativeCols)
    val m = columns.length
    val qualiColumns = $(inputQualitativeCols)
    val qualiLength = qualiColumns.length
    val quantiColumns = $(inputQuantitativeCols)

    // Step1: for each qualitative feature and discretized quantitative feautre,
    // calculate co-occurrence of each value-pair.
    // TODO: For now, not considering multi-valued features

    // 2 dimension array. The first dimension is the columns.
    // the second is a listMap of (x, y) -> d(x, y) for every x != y of this column.
//    val coOccurrence = Array.fill(columns.length)(mutable.ListMap.empty[(Double, Double), Double])

    // use matrix to save storage and fast access to values
    val coOccurrence: Array[SparseMatrix] = Array.ofDim(qualiLength)
    val sigs = Array.fill(quantiColumns.length)(0.0)

    for (i <- columns.indices) {
      val coli = columns(i)
      val values = if (qualiLength > i) {
        occurrenceOf(qualiColumns(i))
      } else {
        dataset.select(coli).rdd.map(row =>
          (row.getDouble(0), 1)).
          reduceByKey(_ + _).
          collect().
          map(value => (value._1, value._2.toLong))
        // .groupBy(coli).count().collect().map(row => row.getDouble(0) -> row.getLong(1))
      }

      // mutable.Buffer.empty[(Int, Int, Double)], it is a coordinate list
      val coo_i = Array.ofDim[(Int, Int, Double)](values.length * values.length)

      // TODO: use other collection rather than ListMap to improve performance
      var conditionalProbas = mutable.ListMap.empty[Double, Array[(Double, Double)]]

      // a map likes (vi -> (vj, p_ij))
      values.foreach(v => conditionalProbas += (v._1 -> Array.empty[(Double, Double)]))

      val rest = if (i == 0) {
        columns.drop(i)
      } else {
        columns.take(i) ++ columns.drop(i + 1)
      }
      for(j <- rest.indices) {
        val colj = rest(j)
        val temp = dataset.groupBy(coli, colj).count().rdd
          .map(row => {
            val vi = row.getDouble(0)
            val vj = row.getDouble(1)
            val count = row.getLong(2)
            val p_ij = count.toDouble / values.find(_._1 == vi).get._2.toDouble
            (vi, (vj, p_ij))
          })
        temp.collect().foreach(row =>
          conditionalProbas += (row._1 -> (conditionalProbas(row._1) :+ row._2)))

        // now, we cal calculate the d_ij(x,y) for every x and y in column i
        var matrixIndex = 0
        for (x <- 0 until values.length) {
          val vx = values(x)
          for (y <- 0 until values.length) {
            val vy = values(y)
            if (vx._1.toInt <= vy._1.toInt) {
              coo_i(matrixIndex) = (vx._1.toInt, vy._1.toInt, 0.0)
            }
            else {
              // d_j(x, y)
              val d_jx = conditionalProbas(vx._1)
              val d_jy = conditionalProbas(vy._1)

              if (d_jx.length != d_jy.length) {
                throw new Exception("The length of array `conditionalProbas` should be same")
              }
              var d_jxy = 0.0
              for (k <- d_jx.indices) {
                d_jxy += Math.max(d_jx(k)._2, d_jy(k)._2)
              }
              coo_i(matrixIndex) = (vx._1.toInt, vy._1.toInt, d_jxy / (m - 1))
            }
            matrixIndex += 1
            // add d_j(x,y) to the array of d(x,y),
            // and directly divided by m-1 to avoid another loop.
            // coo_i += ((vx._1, vy._1) -> d_jxy / (m - 1))
            // coo_i.append((vx._1.toInt, vy._1.toInt, d_jxy / (m - 1)))
//            coOccurrence(i).up
          }
        }
      }
      if(i < qualiLength) {
        coOccurrence(i) = SparseMatrix.fromCOO(values.length, values.length, coo_i)
      }
      else {
        // for each numeric column, compute the significance
        // using the co-occurrence of numeric features.
        sigs(i) = coo_i.reduce((x, y) =>
          (0, 0, x._3 + y._3))._3 / ($(Interval) * ($(Interval) - 1) / 2)
      }

    }

    val ss = dataset.sparkSession
    // the co-occurrence of numeric features is not needed
    val coOccurrenceMatrix = ss.sparkContext.broadcast(
      coOccurrence.dropRight($(inputQuantitativeCols).length))


    val significances = ss.sparkContext.broadcast(sigs)
    (coOccurrenceMatrix.value, significances.value)
  }

  def occurrenceOf(col: String): Array[(Double, Long)] = $(occurrences)(col)

  /**
   * Each array of double is the values originals, of which the index is the value of indexed.
   * For example, array(a,b,c) means the index of a is 0, of b is 1, and of c is 2.
   * @param data The input dataset
   * @param features The array of names of features
   * @return array of features of their values zipped with index
   */
  def getFeatureIndexes(data: Dataset[_], features: Array[String]): Array[Array[(String, Int)]] = {
    val schema = data.schema
    features.map(f => {
      schema(f).metadata.getMetadata("ml_attrs").getStringArray("vals").zipWithIndex
    })
  }
}
