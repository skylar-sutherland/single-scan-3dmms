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
import scalismo.faces.io.MoMoIO
import scalismo.faces.sampling.face.MoMoRenderer
import scalismo.utils.Random

import scala.reflect.io.Path

/* Adapted from the Basel Face Registration Pipeline
https://github.com/unibas-gravis/basel-face-pipeline/blob/master/src/main/scala/fitting/experiments/QualitativeLFW.scala
That project was released via the Apache License 2.0 by the University of Basel, and uses the same license file as was
provided with this project. */

object QualitativeLFW {
  def fit(inputName: String, isFace: Boolean, modelPaths: IndexedSeq[String], modelNames: IndexedSeq[String])
         (implicit random: Random): Unit = {
    val targetsPath =  Path("pipeline-data/") + "/qualitative_lfw/lfw_selection/" + inputName + "/"

    val files = new File(targetsPath).listFiles.filter(_.getName.endsWith(".png"))
    val listTarget = for (p <- files.toIndexedSeq) yield p.getName.substring(0, p.getName.length - 4)
    listTarget.foreach{println(_)}

    val modelsWithPaths = for ((modelPath, modelName) <- modelPaths.zip(modelNames).par) yield {(
      MoMoIO.read(new File(Path("scan/") + modelPath)).get,
      Path("pipeline-data/") + "/qualitative_lfw/lfw_results" + inputName + "/" + modelName + "/"
    )}

    for (targetName <- listTarget.par; (model, outPath) <- modelsWithPaths) {
      val outPathTarget = outPath + targetName + "/"
      Path(outPathTarget).createDirectory(failIfExists = false)
      val targetFn = targetsPath + targetName + ".png"
      val targetLM = targetsPath + targetName + ".tlms"
      val renderer = MoMoRenderer(model, RGBA.BlackTransparent).cached(5)
      StandardFitScript.fit(targetFn, targetLM, outPathTarget, renderer, isFace = isFace)
    }
  }
}

// Requires all 9 face models based on the BFM mean (RGB, XYZ, and combined; standard, correlated, and symmetric).
object BFMMeanFaceLFW extends App {
  scalismo.initialize(); val seed = 1987L; implicit val rnd: Random = scalismo.utils.Random(seed)
  val modelNames = for (s1 <- IndexedSeq("standard", "correlated", "symmetric"); s2 <- IndexedSeq("rgb", "xyz", "combined")) yield s1 + "_" + s2
  QualitativeLFW.fit("faces", isFace = true, modelNames, for (name <- modelNames) yield "/bfm19/" + name + ".h5")
}

// Requires all the face models based on face scans.
object ScanFaceLFW extends App {
  scalismo.initialize(); val seed = 1987L; implicit val rnd: Random = scalismo.utils.Random(seed)
  val files = new File(Path("pipeline-data/") + "/data/bfm_scans/").listFiles.toIndexedSeq.filter(_.getName.endsWith(".ply"))
  val modelNames = for (fp <- files) yield fp.getName.split('.')(0)
  val modelPaths = for (name <- modelNames) yield "bfmscan_symmetric_multimodels/" + name + ".h5"
  QualitativeLFW.fit("faces", isFace = true, modelNames, modelPaths)
}

// Requires all 4 models based on the bird.
object BirdLFW extends App {
  scalismo.initialize(); val seed = 1987L; implicit val rnd: Random = scalismo.utils.Random(seed)
  val modelNames = for (s1 <- IndexedSeq("standard", "symmetric"); s2 <- IndexedSeq("xyz", "combined")) yield s1 + "_" + s2
  QualitativeLFW.fit("birds", isFace = false, modelNames, for (name <- modelNames) yield "/birds/bird_" + name + ".h5")
}