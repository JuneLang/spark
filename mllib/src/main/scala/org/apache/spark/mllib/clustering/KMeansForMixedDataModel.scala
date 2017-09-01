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

package org.apache.spark.mllib.clustering

import scala.collection.JavaConverters._

import org.json4s._
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._

import org.apache.spark.SparkContext
import org.apache.spark.annotation.Since
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.mllib.linalg.{SparseMatrix, Vector}
import org.apache.spark.mllib.pmml.PMMLExportable
import org.apache.spark.mllib.util.{Loader, Saveable}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SparkSession}

/**
 * A clustering model for K-means. Each point belongs to the cluster with the closest center.
 */
@Since("0.8.0")
class KMeansForMixedDataModel @Since("1.1.0")
  (@Since("1.0.0") val clusterCenters: Array[VectorWithVector],
   val coOccurrences: Array[SparseMatrix],
   val significances: Array[Double],
   val indices: Array[Int])
  extends Saveable with Serializable with PMMLExportable {

  private val clusterCentersWithVector =
    if (clusterCenters == null) null else clusterCenters.map(new VectorWithVector(_))
  private implicit val coo = coOccurrences
  private implicit val sig = significances
  private implicit val ind = indices

  /**
   * Total number of clusters.
   */
  @Since("0.8.0")
  def k: Int = clusterCentersWithVector.length

  /**
   * Returns the cluster index that a given point belongs to.
   */
  def predict(point: VectorWithVector): Int = {
    KMeansForMixedData.findClosest(clusterCentersWithVector, point)._1
  }

  /**
   * Maps given points to their cluster indices.
   */
  @Since("1.0.0")
  def predict(points: RDD[VectorWithVector]): RDD[Int] = {
    val bcCenters = points.context.broadcast(clusterCentersWithVector)
    points.map(p => KMeansForMixedData.findClosest(bcCenters.value, p)._1)
  }

  /**
   * Returns the cluster index that a given point belongs to.
   */
  def predict(point: Vector): Int = {
    val p = VectorWithVector.create(point, significances, indices.last + 1)
    predict(p)
  }
// scala cant recognize the data type in rdd.
//  def predict(points: RDD[Vector]): RDD[Int] = {
//    val ps = points.map(vec => VectorWithVector.create(vec, significances, indices.last + 1))
//    predict(ps)
//  }

  /**
   * Maps given points to their cluster indices.
   */
  @Since("1.0.0")
  def predict(points: JavaRDD[Vector]): JavaRDD[java.lang.Integer] =
    predict(points.rdd).toJavaRDD().asInstanceOf[JavaRDD[java.lang.Integer]]

  /**
   * Return the K-means cost (sum of squared distances of points to their nearest center) for this
   * model on the given data.
   */
  def computeCost(data: RDD[VectorWithVector]): Double = {
    val bcCenters = data.context.broadcast(clusterCentersWithVector)
    val cost = data
      .map(p => KMeansForMixedData.pointCost(bcCenters.value, p)).sum()
    bcCenters.destroy(blocking = false)
    cost
  }


  @Since("1.4.0")
  override def save(sc: SparkContext, path: String): Unit = {
    KMeansForMixedDataModel.SaveLoadV1_0.save(sc, this, path)
  }

  override protected def formatVersion: String = "1.0"
}

//scalastyle:off
@Since("1.4.0")
object KMeansForMixedDataModel extends Loader[KMeansModel] {

  @Since("1.4.0")
  override def load(sc: SparkContext, path: String): KMeansModel = {
    KMeansModel.SaveLoadV1_0.load(sc, path)
  }

  private case class Cluster(id: Int, point: Vector)

  private object Cluster {
    def apply(r: Row): Cluster = {
      Cluster(r.getInt(0), r.getAs[Vector](1))
    }
  }

  private[clustering]
  object SaveLoadV1_0 {

    private val thisFormatVersion = "1.0"

    private[clustering]
    val thisClassName = "org.apache.spark.mllib.clustering.KMeansForMixedDataModel"

    def save(sc: SparkContext, model: KMeansForMixedDataModel, path: String): Unit = {
      val spark = SparkSession.builder().sparkContext(sc).getOrCreate()
      val metadata = compact(render(
        ("class" -> thisClassName) ~ ("version" -> thisFormatVersion) ~ ("k" -> model.k)))
      sc.parallelize(Seq(metadata), 1).saveAsTextFile(Loader.metadataPath(path))
      val dataRDD = sc.parallelize(model.clusterCentersWithVector.zipWithIndex).map { case (p, id) =>
        Cluster(id, p.vector)
      }
      spark.createDataFrame(dataRDD).write.parquet(Loader.dataPath(path))
    }

//    def load(sc: SparkContext, path: String): KMeansModel = {
//      implicit val formats = DefaultFormats
//      val spark = SparkSession.builder().sparkContext(sc).getOrCreate()
//      val (className, formatVersion, metadata) = Loader.loadMetadata(sc, path)
//      assert(className == thisClassName)
//      assert(formatVersion == thisFormatVersion)
//      val k = (metadata \ "k").extract[Int]
//      val centroids = spark.read.parquet(Loader.dataPath(path))
//      Loader.checkSchema[Cluster](centroids.schema)
//      val localCentroids = centroids.rdd.map(Cluster.apply).collect()
//      assert(k == localCentroids.length)
//      new KMeansForMixedDataModel(localCentroids.sortBy(_.id).map(_.point))
//    }
  }
}
