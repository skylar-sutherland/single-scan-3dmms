/* Copyright 2021 Massachusetts Institute of Technology

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

package fitting.experiments

import java.io.File

import fitting.StandardFitScript
import scalismo.color.RGBA
import scalismo.faces.io.{MoMoIO, RenderParameterIO}
import scalismo.faces.momo.MoMo
import scalismo.faces.sampling.face.MoMoRenderer
import scalismo.utils.Random
import breeze.linalg._

import scala.reflect.io.Path

/* Adapted from the Basel Face Registration Pipeline
https://github.com/unibas-gravis/basel-face-pipeline/blob/master/src/main/scala/fitting/experiments/RecognitionMultiPIE.scala
That project was released via the Apache License 2.0 by the University of Basel, and uses the same license file as was
provided with this project. */

object RecognitionMultiPIE {
  def fit(model: MoMo, modelName: String)(implicit rand: Random): Unit = {
    val targetsPath = Path("pipeline-data/") + "/recognition"
    val outPath = targetsPath + "/results/" + modelName + "/"
    val files = new File(targetsPath + "/originals/").listFiles.filter(_.getName.endsWith(".png"))
    val listTarget = files.map(p => p.getName.substring(0, p.getName.length - 4)).toList
    for (targetName <- listTarget) {
      val outPathTarget = outPath + targetName + "/"
      if (!Path(outPathTarget).exists) {
        Path(outPathTarget).createDirectory(failIfExists = false)
        val renderer = MoMoRenderer(model, RGBA.BlackTransparent).cached(5)
        val targetFn = targetsPath + "/originals/" +  targetName + ".png"
        val targetLM = targetsPath + "/landmarks/" + targetName + "_face0.tlms"
        StandardFitScript.fit(targetFn, targetLM, outPathTarget, renderer)
      }
    }
  }

  case class Fit (id: String, pose: String, coeffs: IndexedSeq[Double])
  case class Match (id: String, similarity: Double)

  def cosineAngle(aa: IndexedSeq[Double], bb: IndexedSeq[Double]): Double = {
    val a = DenseVector(aa.toArray)
    val b = DenseVector(bb.toArray)
    (a dot b) / (norm(a) * norm(b))
  }

  val listOfPoses: IndexedSeq[String] = IndexedSeq("051", "140", "130", "080")

  def queriesWithSimilarities(modelName: String): IndexedSeq[IndexedSeq[(String, IndexedSeq[Match])]] = {
    val resultPath = Path("pipeline-data/") + "/recognition/results/" + modelName + "/"
    val files = new File(resultPath).listFiles.filter(_.isDirectory).toIndexedSeq.sortBy(_.getAbsoluteFile)
    val allFits = for (f <- files) yield {
      val name = f.getName
      val id = name.substring(0, 3)
      val pose = name.substring(10, 13)
      val rps = RenderParameterIO.read(new File(resultPath + name + "/fit-best.rps")).get
      val coeffs = rps.momo.color ++ rps.momo.shape
      Fit(id, pose, coeffs)
    }

    val gallery = allFits.filter(fit => fit.pose == "051")
    val output = for (queryPose <- listOfPoses) yield {
      val queriesInExperiment = allFits.filter(fit => fit.pose == queryPose)
      for (query <- queriesInExperiment) yield {
        val similaritiesForQuery = for (subject <- gallery) yield {
          Match(subject.id, cosineAngle(query.coeffs, subject.coeffs))
        }
        (query.id, similaritiesForQuery)
      }
    }
    output
  }

  def evaluate(modelName: String): Unit = {
    val correctMatchesPerExperiment = for (experiment <- queriesWithSimilarities(modelName)) yield {
      val correctMatches = for ((query_id, similarities) <- experiment) yield {
        val bestMatch = similarities.maxBy(m => m.similarity)
        if (bestMatch.id == query_id) 1.0 else 0.0
      }
      correctMatches.sum / experiment.length
    }
    println(modelName + correctMatchesPerExperiment)
  }

  def evaluateKDE(modelNames: IndexedSeq[String]): Unit = {
    val similaritiesPerModel = for (name <- modelNames) yield queriesWithSimilarities(name)
    val numImgs = similaritiesPerModel(0)(0).length
    val correctMatchesPerExperiment = for (m <- similaritiesPerModel) yield for (experiment <- m) yield {
      for ((query_id, similarities) <- experiment) yield {
        val bestMatch = similarities.maxBy(m => m.similarity)
        (bestMatch.similarity, if (bestMatch.id == query_id) 1.0 else 0.0)
      }
    }
    for (p <- listOfPoses.zipWithIndex) {
      val filtered = for (model <- correctMatchesPerExperiment) yield model(p._2)
      val zeros = IndexedSeq.fill(numImgs)((0.0, 0.0))
      val results = filtered.foldLeft(zeros)((a,b) => for ((aa, bb) <- a.zip(b)) yield if (aa._1 > bb._1) aa else bb)
      println(results.map(_._2).sum / numImgs)
    }
  }
}

// Requires all 9 face models based on the BFM mean (RGB, XYZ, and combined; standard, correlated, and symmetric).
object FaceRecognitionFromBFMMean extends App {
  scalismo.initialize(); val seed = 1987L; implicit val rnd: Random = scalismo.utils.Random(seed)
  val modelNames = for (s1 <- IndexedSeq("standard", "correlated", "symmetric"); s2 <- IndexedSeq("rgb", "xyz", "combined")) yield s1 + "_" + s2
  for (name <- modelNames) {
    val model = MoMoIO.read(new File(Path("scan/") + "/bfm19/" + name + ".h5")).get
    RecognitionMultiPIE.fit(model, name)
    RecognitionMultiPIE.evaluate(name)
  }
}

// Requires all the face models based on individual face scans.
object FaceRecognitionFromSingleFaceScans extends App {
  scalismo.initialize(); val seed = 1987L; implicit val rnd: Random = scalismo.utils.Random(seed)
  val files = new File(Path("pipeline-data/") + "/data/bfm_scans/").listFiles.toIndexedSeq.filter(_.getName.endsWith(".ply"))
  for (fp <- files) {
    val name = fp.getName.split('.')(0)
    val model = MoMoIO.read(new File(Path("scan/") + "bfmscan_symmetric_multimodels/" + name + ".h5")).get
    RecognitionMultiPIE.fit(model, name)
    RecognitionMultiPIE.evaluate(name)
  }
}

// Requires that the registration experiment (BuildRegistrationModels) has been performed.
object FaceRecognitionFrom10ScanPCA extends App {
  scalismo.initialize(); val seed = 1987L; implicit val rnd: Random = scalismo.utils.Random(seed)
  val model = MoMoIO.read(new File(Path("pipeline-data/") + "/registration/models/bfm09GTfromScans.h5")).get
  val name = "pca_10_scans"
  RecognitionMultiPIE.fit(model, name)
  RecognitionMultiPIE.evaluate(name)
}

// Requires that FaceRecognitionFromSingleFaceScans has been run.
object FaceRecognitionFrom10ScanKDE extends App {
  scalismo.initialize(); val seed = 1987L; implicit val rnd: Random = scalismo.utils.Random(seed)
  val files = new File(Path("pipeline-data/") + "/data/bfm_scans/").listFiles.toIndexedSeq.filter(_.getName.endsWith(".ply"))
  RecognitionMultiPIE.evaluateKDE(for (fp <- files) yield fp.getName.split('.')(0))
}