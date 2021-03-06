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

import scala.language.implicitConversions

import org.apache.spark.annotation.Since
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.internal.Logging
import org.apache.spark.ml.{linalg => newlinalg}
import org.apache.spark.ml.clustering.{KMeansForMixedData => NewKMeans}
import org.apache.spark.ml.util.Instrumentation
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.BLAS.{axpy, scal}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.Utils
import org.apache.spark.util.random.XORShiftRandom



/**
 * dfawer
 * sdfasdfsdf
 */
@Since("2.1.1")
class KMeansForMixedData private (
    val coOccurrences: Array[SparseMatrix],
    val significances: Array[Double],
    val indices: Array[Int],
    private var k: Int,
    private var maxIterations: Int,
    private var initializationMode: String,
    private var initializationSteps: Int,
    private var epsilon: Double,
    private var seed: Long) extends Serializable with Logging  {

  /**
   * Constructs a KMeans instance with default parameters: {k: 2, maxIterations: 20,
   * initializationMode: "k-means||", initializationSteps: 2, epsilon: 1e-4, seed: random}.
   */
  @Since("0.8.0")
  def this(coOccurrences: Array[SparseMatrix], significances: Array[Double],
    indices: Array[Int]) = this(coOccurrences, significances, indices,
    2, 20, KMeans.RANDOM, 2, 1e-4, Utils.random.nextLong())

  def getCoOccurrences: Array[SparseMatrix] = this.coOccurrences
//  def setCoOccurrences(coo: Array[SparseMatrix]): this.type = {
//    this.coOccurrences = coo
//    this
//  }

  def getSignificances: Array[Double] = significances
//  def setSignificances(sigs: Array[Double]): this.type = {
//    significances = sigs
//    this
//  }

  def getIndices: Array[Int] = indices
//  def setIndices(inds: Array[Int]): this.type = {
//    indices = inds
//    this
//  }

  private implicit val coo = coOccurrences
  private implicit val sig = significances
  private implicit val ind = indices
  /**
   * Number of clusters to create (k).
   *
   * @note It is possible for fewer than k clusters to
   * be returned, for example, if there are fewer than k distinct points to cluster.
   */
  @Since("1.4.0")
  def getK: Int = k

  /**
   * Set the number of clusters to create (k).
   *
   * @note It is possible for fewer than k clusters to
   * be returned, for example, if there are fewer than k distinct points to cluster. Default: 2.
   */
  @Since("0.8.0")
  def setK(k: Int): this.type = {
    require(k > 0,
      s"Number of clusters must be positive but got ${k}")
    this.k = k
    this
  }

  /**
   * Maximum number of iterations allowed.
   */
  @Since("1.4.0")
  def getMaxIterations: Int = maxIterations

  /**
   * Set maximum number of iterations allowed. Default: 20.
   */
  @Since("0.8.0")
  def setMaxIterations(maxIterations: Int): this.type = {
    require(maxIterations >= 0,
      s"Maximum of iterations must be nonnegative but got ${maxIterations}")
    this.maxIterations = maxIterations
    this
  }

  /**
   * The initialization algorithm. This can be either "random" or "k-means||".
   */
  @Since("1.4.0")
  def getInitializationMode: String = initializationMode

  /**
   * Set the initialization algorithm. This can be either "random" to choose random points as
   * initial cluster centers, or "k-means||" to use a parallel variant of k-means++
   * (Bahmani et al., Scalable K-Means++, VLDB 2012). Default: k-means||.
   */
  @Since("0.8.0")
  def setInitializationMode(initializationMode: String): this.type = {
    KMeans.validateInitMode(initializationMode)
    this.initializationMode = initializationMode
    this
  }

  /**
   * Number of steps for the k-means|| initialization mode
   */
  @Since("1.4.0")
  def getInitializationSteps: Int = initializationSteps

  /**
   * Set the number of steps for the k-means|| initialization mode. This is an advanced
   * setting -- the default of 2 is almost always enough. Default: 2.
   */
  @Since("0.8.0")
  def setInitializationSteps(initializationSteps: Int): this.type = {
    require(initializationSteps > 0,
      s"Number of initialization steps must be positive but got ${initializationSteps}")
    this.initializationSteps = initializationSteps
    this
  }

  /**
   * The distance threshold within which we've consider centers to have converged.
   */
  @Since("1.4.0")
  def getEpsilon: Double = epsilon

  /**
   * Set the distance threshold within which we've consider centers to have converged.
   * If all centers move less than this Euclidean distance, we stop iterating one run.
   */
  @Since("0.8.0")
  def setEpsilon(epsilon: Double): this.type = {
    require(epsilon >= 0,
      s"Distance threshold must be nonnegative but got ${epsilon}")
    this.epsilon = epsilon
    this
  }

  /**
   * The random seed for cluster initialization.
   */
  @Since("1.4.0")
  def getSeed: Long = seed

  /**
   * Set the random seed for cluster initialization.
   */
  @Since("1.4.0")
  def setSeed(seed: Long): this.type = {
    this.seed = seed
    this
  }

  // Initial cluster centers can be provided as a KMeansModel object rather than using the
  // random or k-means|| initializationMode
  private var initialModel: Option[KMeansForMixedDataModel] = None

  /**
   * Set the initial starting point, bypassing the random initialization or k-means||
   * The condition model.k == this.k must be met, failure results
   * in an IllegalArgumentException.
   */
  @Since("1.4.0")
  def setInitialModel(model: KMeansForMixedDataModel): this.type = {
    require(model.k == k, "mismatched cluster count")
    initialModel = Some(model)
    this
  }

  def run(data: RDD[Vector]): KMeansForMixedDataModel = {
    if (data.getStorageLevel == StorageLevel.NONE) {
      logWarning("The input data is not directly cached, which may hurt performance if its"
        + " parent RDDs are also uncached.")
    }
    val cut = indices.last + 1
    val newData: RDD[VectorWithVector] = data.map(vec => {
      VectorWithVector.create(vec, significances, cut)
    })
    newData.persist()
    val model = runAlgorithm(newData)
    newData.unpersist()

    // Warn at the end of the run as well, for increased visibility.
    if (data.getStorageLevel == StorageLevel.NONE) {
      logWarning("The input data was not directly cached, which may hurt performance if its"
        + " parent RDDs are also uncached.")
    }
    model
  }

  /**
   * TODO Unit-> KMeansForMixedDataModel
   * @param data input rdd
   * ,instr: Option[Instrumentation[NewKMeans
   */
  def runAlgorithm (
      data: RDD[VectorWithVector]): KMeansForMixedDataModel = {

    val sc = data.sparkContext

    val initStartTime = System.nanoTime()
    // initialization of k centers randomly

    val centers: Array[VectorWithVector] = initialModel match {
      case Some(kMeansCenters) =>
        kMeansCenters.clusterCenters
      case None =>
        if (initializationMode == KMeans.RANDOM) {
          initRandom(data)
        } else {
          throw new Exception(s"Mode $initializationMode is not supported yet.")
        }
    }

//    // scalastyle:off
//    centers.foreach(vwv => println(vwv.vector.toArray.mkString(",")))

    val initTimeInSeconds = (System.nanoTime() - initStartTime) / 1e9
    logInfo(f"Initialization with $initializationMode took $initTimeInSeconds%.3f seconds.")

    var converged = false
    var cost = 0.0
    var iteration = 0

    val iterationStartTime = System.nanoTime()

    // Execute iterations of Lloyd's algorithm until converged
    while (iteration < maxIterations && !converged) {
      val costAccum = sc.doubleAccumulator
      val bcCenters = sc.broadcast(centers)

      // Find the new centers
      val newCenters = data.zipWithIndex().mapPartitions { points =>
        val thisCenters = bcCenters.value
        val dims = thisCenters.head.size

        val sums = Array.fill(thisCenters.length)(Vectors.zeros(dims))
        val counts = Array.fill(thisCenters.length)(0L)
        val members = Array.fill(thisCenters.length)(Array.empty[Long])

        val st = System.nanoTime()
        points.foreach { case(point, index) =>
          val (bestCenter, cost) = KMeansForMixedData.findClosest(thisCenters, point)
          costAccum.add(cost)
          val sum = sums(bestCenter)
          axpy(1.0, point.vector, sum)
          counts(bestCenter) += 1
        }
        val computeTime = (System.nanoTime() - st) / 1e9
        logInfo(f"compute time took $computeTime%.3f seconds.")

        counts.indices.filter(counts(_) > 0).map(j => (j, (sums(j), counts(j)))).iterator
      }.reduceByKey { case ((sum1, count1), (sum2, count2)) =>
        axpy(1.0, sum2, sum1)
        (sum1, count1 + count2)
      }.mapValues { case (sum, count) =>
        scal(1.0 / count, sum)
        val cut = indices.last + 1
        val qualis = sum.slice(0, cut)
        val quantis = sum.slice(cut, sum.size)
//        new VectorWithVector(qualis, quantis)
        VectorWithVector.create(qualis, quantis, significances)
      }.collectAsMap()

      bcCenters.destroy(blocking = false)

      // update de cluster centers and costs
      converged = true
      newCenters.foreach {case (j, newCenter) =>
        val sqQualidist = KMeansForMixedData.sqQualiDistance(newCenter.qualiVector,
                                                              centers(j).qualiVector)
        if(converged &&
            KMeansForMixedData.fastSquaredDistance(newCenter, centers(j),
                                                    sqQualidist) > epsilon * epsilon) {
          converged = false
        }
          centers(j) = newCenter
      }

      cost = costAccum.value
      iteration += 1
    }

    val iterationTimeInSeconds = (System.nanoTime() - iterationStartTime) / 1e9
    logInfo(f"Iterations took $iterationTimeInSeconds%.3f seconds.")
    if (iteration == maxIterations) {
      logInfo(s"KMeans reached the max number of iterations: $maxIterations.")
    } else {
      logInfo(s"KMeans converged in $iteration iterations.")
    }

    logInfo(s"The cost is $cost.")

    new KMeansForMixedDataModel(centers, coOccurrences, significances, indices)
  }

  /**
   * Initialize a set of cluster centers at random.
   */
  private def initRandom(data: RDD[VectorWithVector]): Array[VectorWithVector] = {
    // Select without replacement; may still produce duplicates if the data has < k distinct
    // points, so deduplicate the centroids to match the behavior of k-means|| in the same situation
    data.takeSample(false, k, new XORShiftRandom(this.seed).nextInt()).distinct
  }
}

object KMeansForMixedData {

  private[spark] def validateInitMode(initMode: String): Boolean = {
    initMode match {
      case KMeans.RANDOM => true
      case KMeans.K_MEANS_PARALLEL => true
      case _ => false
    }
  }

  /**
   * Returns the index of the closest center to the given point, as well as the squared distance.
   */
  private[mllib] def findClosest(
                                  centers: TraversableOnce[VectorWithVector],
                                  point: VectorWithVector)
                                (implicit coOccurences: Array[SparseMatrix],
                                 significances: Array[Double],
                                 indices: Array[Int]): (Int, Double) = {
    var bestDistance = Double.PositiveInfinity
    var bestIndex = 0
    var i = 0
    centers.foreach { center =>
      val (lowerBoundOfSqDist, sqQualiDist) = fastCompareDistance(center, point)
      if (lowerBoundOfSqDist < bestDistance) {
        val distance = fastSquaredDistance(center, point, sqQualiDist)
        if (distance < bestDistance) {
          bestDistance = distance
          bestIndex = i
        }
      }
      i += 1
    }
    (bestIndex, bestDistance)
  }

  def fastCompareDistance(
                           v1: VectorWithVector,
                           v2: VectorWithVector)
                         (implicit coOccurrences: Array[SparseMatrix],
                          indices: Array[Int]): (Double, Double) = {
    val quald = sqQualiDistance(v1.qualiVector, v2.qualiVector)
    var norm = v1.norm - v2.norm
    norm = norm * norm
    (quald + norm, quald)
  }

  private def fastSquaredDistance(
                                   v1: VectorWithVector,
                                   v2: VectorWithVector,
                                   sqQualiDist: Double)
                                 (implicit significances: Array[Double]): Double = {
    val sqQuantiDist = MLUtils.fastSquaredDistance(v1.quantiVectorWithSig, v1.norm,
                                                    v2.quantiVectorWithSig, v2.norm)
    sqQualiDist + sqQuantiDist
  }

  /**
   * The squared distance between 2 vectors.
   * This function is suitable for multi-value features.
   * @param v1
   * @param v2
   * @param coOccurrences
   * @param indices
   * @return
   */
  def sqQualiDistance(v1: Vector, v2: Vector)
                     (implicit coOccurrences: Array[SparseMatrix],
                      indices: Array[Int]): Double = {
    // how to get co-occurrence?
    // need to put these functions in class, not object
    var sum = 0.0
    var ind = 0
    //    val size = v1.size

    //    var vv1 = 0
    //    var vv2 = 0
    //    var dist = 0.0
    //    for (i <- 0 until size) {
    //      // TODO: compute distance
    //      if (v1(i) == 1.0) {
    //        vv1 = i
    //      }
    //      if (v2(i) == 1.0) {
    //        vv2 = i
    //      }
    //      // if we get the index of values for both 2 vectors
    //      if (vv1 * vv2 != 0) {
    //        // when indexes are different
    //        if (vv1 != vv2) {
    //          val (vv1_index, vv2_index) = findValuesIndex(indices, vv1, vv2, ind)
    //          dist = if (vv1_index < vv2_index) {
    //            coOccurrences(ind)(vv1_index, vv2_index)
    //          }
    //          else {
    //            coOccurrences(ind)(vv2_index, vv1_index)
    //          }
    //        }
    //        else { // when indexed are same, means values are same
    //          dist = 0
    //        }
    //
    //        ind += 1
    //        vv1 = 0
    //        vv2 = 0
    //      }
    //      val dist = coOccurrences(index)(v1(i), v2(i))
    //      sum += dist * dist
    //    }

    while(ind < indices.length) {
      // the indexed values are (begin, end]
      val endOfFeature = indices(ind)
      val beginOfFeature = if (ind == 0) -1 else indices(ind - 1)
      // the matrix for current feature
      val matrix = coOccurrences(ind)
      var dist = 0.0
      // loop for v1
      for (i <- (beginOfFeature + 1) until endOfFeature) {
        // loop for v2
        for (j <- (i + 1) to endOfFeature) {
          val coeff1 = v1(i) * v2(j)
          val coeff2 = v1(j) * v2(i)
          dist += Math.abs(coeff1 - coeff2) *
            matrix(i - beginOfFeature - 1, j - beginOfFeature - 1)
        }
      }
      sum += dist * dist
      ind += 1
    }
    sum
  }

  /**
   * Returns the cost of a given point against the given cluster centers.
   */
  private[mllib] def pointCost(centers: TraversableOnce[VectorWithVector],
                                point: VectorWithVector)
                              (implicit coOccurrences: Array[SparseMatrix],
                               significances: Array[Double],
                               indices: Array[Int]): Double =
    findClosest(centers, point)._2
}

/**
 * A vector with a vector of qualitative features, a vector of quantitative feature;
 * the norm of quantitative vector, and the indices of qualitative features.
 * The qualiVector is composed by the vectors generated by One-hot-encoder of each feature.
 *
 */
private[clustering]
class VectorWithVector(
    val qualiVector: Vector,
    val quantiVector: Vector,
    val quantiVectorWithSig: Vector,
    val norm: Double) extends Serializable {
  val vector: Vector = Vectors.dense(qualiVector.toArray ++ quantiVector.toArray).compressed

  def size: Int = qualiVector.size + quantiVector.size

  def this(vector: VectorWithVector) = this(vector.qualiVector, vector.quantiVector,
                                             vector.quantiVectorWithSig, vector.norm)

//  def this(array: Array[Double]) = this(Vectors.dense(array))

  /** Converts the vector to a dense vector. */
  def toDense: VectorWithNorm = new VectorWithNorm(Vectors.dense(vector.toArray), norm)
}

object VectorWithVector {
  def create(qualiVector: Vector,
             quantiVector: Vector, significance: Array[Double]): VectorWithVector = {
    require(quantiVector.size == significance.length, "Not every quantitative feature " +
      s"has a significance. vec:${qualiVector.size} ${quantiVector.size}, " +
      s"sig: ${significance.length}")
    val values = quantiVector.toArray.zip(significance).map{ case (x, a) => (a * x).toDouble }
    val quantiVectorWithSig = Vectors.dense(values)
    new VectorWithVector(qualiVector, quantiVector,
                          quantiVectorWithSig, Vectors.norm(quantiVectorWithSig, 2.0))
  }

  def create(vector: Vector,
             significances: Array[Double],
             cut: Int): VectorWithVector = {
    val qualiVec = vector.slice(0, cut)
    val quantiVec = vector.slice(cut, vector.size)
    VectorWithVector.create(qualiVec, quantiVec, significances)
  }
}

private[spark] object MatrixImplicits{
  implicit def mlMatrixArrayToMllibMatrixArray(m: Array[newlinalg.Matrix]): Array[Matrix] = {
    m map Matrices.fromML
  }

  implicit def mllibMatrixArrayToMlMatrixArray(m: Array[Matrix]): Array[newlinalg.Matrix] = {
    m map(_.asML)
  }

  implicit def mlSMAToMllibSMA(m: Array[newlinalg.SparseMatrix]): Array[SparseMatrix] = {
    m.map(Matrices.fromML(_).asInstanceOf[SparseMatrix])
  }
}
