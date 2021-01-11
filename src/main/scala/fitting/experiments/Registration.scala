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
import java.net.URI

import breeze.linalg.{DenseVector, max}
import fitting.RegistrationFitScript
import onemesh.{CombinedModelHelpers, CreateCombinedModel}
import scalismo.color.RGBA
import scalismo.faces.io.{MeshIO, MoMoIO, PixelImageIO}
import scalismo.faces.mesh.ColorNormalMesh3D
import scalismo.faces.momo.MoMo
import scalismo.faces.parameters.{MoMoInstance, RenderParameter}
import scalismo.faces.sampling.face.MoMoRenderer
import scalismo.geometry.{EuclideanVector3D, Landmark, Point, Point3D, _3D}
import scalismo.mesh.{MeshOperations, SurfacePointProperty, TriangleMesh3D, VertexColorMesh3D}
import scalismo.utils.Random
import scalismo.faces.io.GravisArrayIO
import scalismo.io.StatismoIO
import scalismo.registration.{LandmarkRegistration, RigidTransformation, ScalingTransformation}

import scala.reflect.io.Path

/* Includes code adapted from A Closest Point Proposal for MCMC-based Probabilistic Surface Registration
https://github.com/unibas-gravis/icp-proposal/blob/master/src/main/scala/apps/bfm/AlignShapes.scala
https://github.com/unibas-gravis/icp-proposal/blob/master/src/main/scala/apps/util/AlignmentTransforms.scala
https://github.com/unibas-gravis/icp-proposal/blob/master/src/main/scala/apps/util/FileUtils.scala
That project was released via the Apache License 2.0 by the University of Basel, and uses the same license file as was
provided with this project. */

object RegistrationHelper {
  def getFiles(targetsPath: String): Array[String] = {
    val files = new File(targetsPath).listFiles.filter(_.getName.endsWith(".ply"))
    for (p <- files) yield p.getName.substring(0, p.getName.length - 4)
  }

  def fitModel(model: MoMo, outPath: String, albedoConsistency: Boolean)(implicit rnd: Random): Unit = {
    val targetsPath = Path("pipeline-data/") + "/registration/aligned_meshes/"
    for (targetName <- getFiles(targetsPath).toList.par) {
      val outPathTarget = outPath + targetName + "/"
      if (!Path(outPathTarget).exists) {
        Path(outPathTarget).createDirectory(failIfExists = false)
        val renderer = MoMoRenderer(model, RGBA.BlackTransparent).cached(5)
        val targetFn = targetsPath + targetName + ".ply"
        RegistrationFitScript.fit(targetFn, outPathTarget, renderer, albedoConsistency)
      }
    }
  }
}

object Crop extends App {
  scalismo.initialize()
  val mask = GravisArrayIO.read[Int](new File(Path("pipeline-data/") + "/data/masks/face12.mask")).get
  val mask2 = GravisArrayIO.read[Int](new File(Path("pipeline-data/") + "/data/masks/face12_nomouth.mask")).get
  val outpath = Path("pipeline-data/") + "/registration/face12mask"
  val inpath = Path("pipeline-data/") + "/data/bfm_scans"
  for (m <- RegistrationHelper.getFiles(inpath).toList) {
    println(m)
    val meshVCM = MeshIO.read(new File(inpath + m + ".ply")).get
    val mesh = meshVCM.shape
    val meshCropped2 = MeshOperations(mesh).maskPoints(p => mask(p.id) == 1).transformedMesh
    val meshCropped = MeshOperations(meshCropped2).maskPoints(p => mask2(p.id) == 1).transformedMesh
    val scaledMesh = meshCropped.transform(p => (p.toVector/1000.0).toPoint)
    val colorCropped2 = meshVCM.vertexColorMesh3D.get.color.pointData.zipWithIndex.filter(r => mask(r._2) == 1).map(_._1)
    val colorCropped = colorCropped2.zipWithIndex.filter(r => mask2(r._2) == 1).map(_._1)
    val newMesh = VertexColorMesh3D(scaledMesh, SurfacePointProperty(scaledMesh.triangulation, colorCropped.map(_.clamped)))
    MeshIO.write(newMesh, new File(outpath + m + ".ply"))
  }
}

// Requires that Crop has been run.
object Align extends App {
  def getBasename(file: File): String = {
    val name = file.getName
    val dot = name.lastIndexOf('.')
    if (dot > 0) name.substring(0, dot) else name
  }
  def computeTransform(lm1: Seq[Landmark[_3D]], lm2: Seq[Landmark[_3D]], center: Point[_3D]): RigidTransformation[_3D] = {
    val commonLmNames = lm1.map(_.id) intersect lm2.map(_.id)
    val landmarksPairs = commonLmNames.map(name => (lm1.find(_.id == name).get.point, lm2.find(_.id == name).get.point))
    LandmarkRegistration.rigid3DLandmarkRegistration(landmarksPairs, center)
  }

  scalismo.initialize(); val seed = 1024L; implicit val random: Random = Random(seed)
  val initialMeshesPath = new File(Path("pipeline-data/") + "/registration/face12mask")
  val alignedMeshesPath = new File(Path("pipeline-data/") + "/registration/aligned_meshes/")
  val model = StatismoIO.readStatismoMeshModel(new File(Path("scan/") + "bfm19/standard_combinedmodel.h5"), "shape").get
  val momo = MoMoIO.read(new File(Path("scan/") + "bfm19/standard_combinedmodel.h5")).get
  val modelLMs = momo.landmarks.toIndexedSeq.map(_._2)
  val scalingTransform = ScalingTransformation[_3D](1 / 1000.0)
  val fromBFM = CreateCombinedModel.bfm17Mean()

  for (f <- initialMeshesPath.listFiles) {
    val basename = getBasename(f)
    val meshVCM =  scalismo.faces.io.MeshIO.read(f).get
    val initialLMs = CombinedModelHelpers.transferLandmarks(fromBFM._2, meshVCM.shape, fromBFM._1.shape)
    val lms = initialLMs.toSeq.map(_._2.transform(scalingTransform))
    val alignmentTransform = computeTransform(lms, modelLMs, Point3D(0, 0, 0))
    val alignedMesh = meshVCM.shape.transform(scalingTransform).transform(alignmentTransform)
    scalismo.faces.io.MeshIO.write(meshVCM.copy(shape = alignedMesh), new File(alignedMeshesPath, basename + ".ply"))
  }
}

// Requires that Align and RenderMultiSpaceModelLvl3Standard have been run.
object Registration extends App {
  scalismo.initialize(); val seed = 1986L; implicit val rnd: Random = scalismo.utils.Random(seed)
  val standard_lvl3 = MoMoIO.read(new File(Path("scan/") + "/bfm19_lvl3/lvl3_standard_combinedmodel.h5")).get
  RegistrationHelper.fitModel(standard_lvl3, Path("pipeline-data/") + "/registration/registered/", albedoConsistency = true)
}

// Requires that Align and RenderMultiSpaceModelLvl3Standard have been run.
object RegistrationNoAlbedoConsistency extends App {
  scalismo.initialize(); val seed = 1986L; implicit val rnd: Random = scalismo.utils.Random(seed)
  val standard_lvl3 = MoMoIO.read(new File(Path("scan/") + "/bfm19_lvl3/lvl3_standard_combinedmodel.h5")).get
  RegistrationHelper.fitModel(standard_lvl3, Path("pipeline-data/") + "/registration/registered_no_albedo/", albedoConsistency = false)
}

// Requires that Registration and RegistrationNoAlbedoConsistency have been run.
object EvaluateShapeRegistration extends App {
  scalismo.initialize()
  val seed = 1986L
  implicit val rnd: Random = scalismo.utils.Random(seed)

  for (base <- IndexedSeq("registered_no_albedo","registered")) {
    println(base)
    val targetsPath = Path("pipeline-data/") + "/registration/aligned_meshes/"
    val outPath = Path("pipeline-data/") + "/registration/" + base + "/"
    for (targetName <- RegistrationHelper.getFiles(targetsPath).toList.par) {
      try {
        val targetMesh = MeshIO.read(new File(outPath + targetName + "/target.ply")).get.shape.pointSet
        val fitMesh = MeshIO.read(new File(outPath + targetName + "/fitter-best.ply")).get.shape.pointSet
        val distances = for (p <- fitMesh.points.toIndexedSeq) yield (p - targetMesh.findClosestPoint(p).point).norm
        val dist = DenseVector(distances.toArray)
        println(targetName, breeze.stats.meanAndVariance(dist), max(dist))
      } catch {
        case _: Throwable => println("not found" + targetName)
      }
    }
  }
}

// Requires that Registration and RegistrationNoAlbedoConsistency have been run.
object BuildRegistrationModels extends App {
  scalismo.initialize()
  val seed = 1986L
  implicit val rnd: Random = scalismo.utils.Random(seed)

  // from Basel Face Pipeline
  /** Extracts per vertex color for the registered mesh from the input mesh. The correspondence for each
    * vertex is sought after along the normal.
    *
    * @param registeredMesh Registration result without color.
    * @param colorMesh Input mesh with color.
    * @return Registeration result with color. */
  def extractVertexColor(registeredMesh: TriangleMesh3D, colorMesh: ColorNormalMesh3D): SurfacePointProperty[RGBA] = {
    val pointsReg = registeredMesh.pointSet.points.toIndexedSeq
    val normalsReg = registeredMesh.vertexNormals.pointData
    val shapeFromColorMesh = colorMesh.shape
    val meshOperations = shapeFromColorMesh.operations
    val colors: IndexedSeq[RGBA] = for ((point, normal) <- pointsReg.zip(normalsReg)) yield {
      val intersections = meshOperations.getIntersectionPointsOnSurface(point, normal)
      val sortedIntersections = (for (i <- intersections) yield {
        val pointI = shapeFromColorMesh.position(i._1, i._2)
        (i, (pointI - point).norm)
      }).sortWith((a, b) => a._2 < b._2)
      try {
        val i = sortedIntersections.head._1
        colorMesh.color(i._1, i._2)
      } catch {
        case _: Throwable => RGBA.Black
      }
    }
    new SurfacePointProperty[RGBA](registeredMesh.triangulation, colors)
  }

  val models = for (base <- IndexedSeq("registered_no_albedo","registered")) yield {
    println(base)
    val targetsPath = Path("pipeline-data/") + "/registration/aligned_meshes/"
    val outPath = Path("pipeline-data/") + "/registration/" + base +"/"
    val meshes = for (targetName <- RegistrationHelper.getFiles(targetsPath).toIndexedSeq) yield {
      val targetMesh = MeshIO.read(new File(outPath + targetName + "/target.ply")).get.colorNormalMesh3D.get
      val fitMesh = MeshIO.read(new File(outPath + targetName + "/fitter-best.ply")).get.shape
      val fitMeshTranslated = fitMesh.transform(p => p + EuclideanVector3D(0, 0, 1000))
      val vertexColor = extractVertexColor(fitMesh, targetMesh)
      VertexColorMesh3D(fitMeshTranslated, vertexColor)
    }
    MoMo.buildFromRegisteredSamples(meshes(0).shape, meshes, meshes, 0.0, 0.0)
  }

  val bfm09GT = {
    val targetsPath = Path("pipeline-data/") + "/registration/face12mask/"
    val meshes = for (targetName <- RegistrationHelper.getFiles(targetsPath).toIndexedSeq) yield {
      MeshIO.read(new File(targetsPath + targetName + ".ply")).get.vertexColorMesh3D.get
    }
    val tmpMomo = MoMo.buildFromRegisteredSamples(meshes(0).shape, meshes, meshes, 0.0, 0.0)
    val fromBFM = CreateCombinedModel.bfm17Mean()
    tmpMomo.withLandmarks(CombinedModelHelpers.transferLandmarks(fromBFM._2, tmpMomo.mean.shape, fromBFM._1.shape))
  }

  for (m <- (models ++ IndexedSeq(bfm09GT)).zip(IndexedSeq("registered_no_albedo", "registered", "bfm09GTfromScans")).par) {
    CombinedModelHelpers.uglyWrite(m._1, Path("pipeline-data/") + "/registration/models/" + m._2 +".h5")
    val renderer = MoMoRenderer(m._1)
    val stdRP = RenderParameter.defaultSquare.fitToImageSize(2048, 2048)
    val img = renderer.renderImage(stdRP.withMoMo(MoMoInstance(IndexedSeq(0.0), IndexedSeq(0.0), IndexedSeq(0.0), new URI(""))))
    PixelImageIO.write(img, new File(Path("pipeline-data/") + "/registration/models/" + m._2 +".png"))
    val imgAp = renderer.renderImage(stdRP.withMoMo(MoMoInstance(IndexedSeq(0.0), IndexedSeq(3.0), IndexedSeq(0.0), new URI(""))))
    PixelImageIO.write(imgAp, new File(Path("pipeline-data/") + "/registration/models/" + m._2 +"+3A.png"))
    val imgAm = renderer.renderImage(stdRP.withMoMo(MoMoInstance(IndexedSeq(0.0), IndexedSeq(-3.0), IndexedSeq(0.0), new URI(""))))
    PixelImageIO.write(imgAm, new File(Path("pipeline-data/") + "/registration/models/" + m._2 +"-3A.png"))
    val imgSp = renderer.renderImage(stdRP.withMoMo(MoMoInstance(IndexedSeq(3.0), IndexedSeq(0.0), IndexedSeq(0.0), new URI(""))))
    PixelImageIO.write(imgSp, new File(Path("pipeline-data/") + "/registration/models/" + m._2 +"+3S.png"))
    val imgSm = renderer.renderImage(stdRP.withMoMo(MoMoInstance(IndexedSeq(-3.0), IndexedSeq(0.0), IndexedSeq(0.0), new URI(""))))
    PixelImageIO.write(imgSm, new File(Path("pipeline-data/") + "/registration/models/" + m._2 +"-3S.png"))
  }
}