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

package onemesh

import java.io.File

import scalismo.color.RGBA
import scalismo.common.UnstructuredPointsDomain.Create.CreateUnstructuredPointsDomain3D
import scalismo.common._
import scalismo.faces.io.{MoMoIO, PixelImageIO}
import scalismo.faces.momo.{MoMo, PancakeDLRGP}
import scalismo.faces.parameters._
import scalismo.geometry._
import scalismo.mesh.{SurfacePointProperty, TriangleMesh3D, VertexColorMesh3D}
import scalismo.numerics.Sampler
import scalismo.statisticalmodel._
import scalismo.utils.Random

import scala.reflect.io.Path

object CombinedModelHelpers {
  type UPD3D = UnstructuredPointsDomain[_3D]

  /* A modification of RenderParameter.default with image width and height divided by 2. */
  val lowResRenderParameter: RenderParameter = RenderParameter(
    pose = Pose.away1m,
    view = ViewParameter.neutral,
    camera = Camera.for35mmFilm(50.0),
    environmentMap = SphericalHarmonicsLight.frontal.withNumberOfBands(2),
    directionalLight = DirectionalLight.off,
    momo = MoMoInstance.empty,
    imageSize = ImageSize(360, 240),
    colorTransform = ColorTransform.neutral
  )

  /* A modification of RenderParameter.defaultSquare with image width and height divied by 2. */
  val lowResSquareRenderParameter: RenderParameter = RenderParameter(
    pose = Pose.away1m,
    view = ViewParameter.neutral,
    camera = Camera(
      focalLength = 7.5,
      principalPoint = Point2D.origin,
      sensorSize = EuclideanVector(2.0, 2.0),
      near = 10,
      far = 1000e3,
      orthographic = false),
    environmentMap = SphericalHarmonicsLight.frontal.withNumberOfBands(2),
    directionalLight = DirectionalLight.off,
    momo = MoMoInstance.empty,
    imageSize = ImageSize(256, 256),
    colorTransform = ColorTransform.neutral
  )

  /* A version of RenderParameter.default with ambient white. */
  val ambientWhite: RenderParameter = RenderParameter(
    pose = Pose.away1m,
    view = ViewParameter.neutral,
    camera = Camera.for35mmFilm(50.0),
    environmentMap = SphericalHarmonicsLight.ambientWhite,
    directionalLight = DirectionalLight.off,
    momo = MoMoInstance.empty,
    imageSize = ImageSize(720, 480),//ImageSize(3600, 2400),
    colorTransform = ColorTransform.neutral
  )

  /* Save a VCM as an image. */
  def renderVCM(VCM: VertexColorMesh3D, fpath: String, color: RGBA = RGBA.Black): Unit = {
    PixelImageIO.write(ParametricRenderer.renderParameterVertexColorMesh(ambientWhite, VCM, color), new File(fpath))
  }

  /* Render a VCM from the side. */
  def renderFromSide(VCM: VertexColorMesh3D, yaw: Double, fpath: String, color: RGBA = RGBA.Black): Unit = {
    val rp = ambientWhite.copy(pose = ambientWhite.pose.withYaw(yaw))
    PixelImageIO.write(ParametricRenderer.renderParameterVertexColorMesh(rp, VCM, color), new File(fpath))
  }

  /* Map a landmark from one UPD3D to another. */
  def transferLandmarks(lm: (String, Landmark[_3D]), newUPD: UPD3D, oldUPD: UPD3D): (String, Landmark[_3D]) = {
    val id = oldUPD.findClosestPoint(lm._2.point)
    val newPoint = Landmark[_3D](lm._1, newUPD.point(id.id))
    (lm._1, newPoint.copy(uncertainty = None))
  }

  /* Map a landmark from one mesh to another. */
  def transferLandmarks(lms: Map[String, Landmark[_3D]], newMesh: TriangleMesh3D,
                        oldMesh: TriangleMesh3D): Map[String, Landmark[_3D]] = {
    val newUPD = newMesh.pointSet
    val oldUPD = oldMesh.pointSet
    for (l <- lms) yield transferLandmarks(l, newUPD, oldUPD)
  }

  /* Write, then read, then write again---terrible, but necessary for the landmarks to work. */
  def uglyWrite(momo: MoMo, fpath: String): Unit = {
    val outfile = new File(fpath)
    MoMoIO.write(momo, outfile)
    val momoLoaded = MoMoIO.read(outfile).get
    val lms = transferLandmarks(momoLoaded.landmarks, momoLoaded.mean.shape, momoLoaded.referenceMesh)
    MoMoIO.write(momoLoaded.withLandmarks(lms), outfile)
  }

  def getReference(plyName: String, lmName: String): (VertexColorMesh3D, Map[String, Landmark[_3D]]) = {
    val path = Path("pipeline-data/") + "/data/references/"
    val reference = scalismo.faces.io.MeshIO.read(new File(path + plyName)).get.vertexColorMesh3D.get
    val lms = scalismo.io.LandmarkIO.readLandmarksJson[_3D](new File(path + lmName)).get
    (reference, (for (lm <- lms) yield (lm.id, lm)).toMap)
  }

  def renderSmoothModel(reference: VertexColorMesh3D, gpColor: GaussianProcess[_3D, EuclideanVector[_3D]],
                        getSampler: VertexColorMesh3D => (Sampler[_3D], UPD3D),
                        folderName: String, numSamples: Int, numComponents: Int = 199)(implicit rand: Random): Unit = {
    val (colorSampler, colorSet) = getSampler(reference)
    val colorPancake = CombinedModel.generateColorPancake(reference, gpColor, colorSampler, colorSet, numComponents)

    println(s"Writing to files...")
    for (i <- 0 until numSamples) {
      val VCM = CombinedModel.pancakeToVCM(colorPancake, reference.shape)
      renderVCM(VCM, Path("scan/") + "/" + folderName + s"/test$i.png")
    }
  }

  def renderShapeModel(reference: VertexColorMesh3D, gpShape: GaussianProcess[_3D, EuclideanVector[_3D]],
                       folderName: String, numSamples: Int, numComponents: Int = 199)(implicit rand: Random): Unit = {
    val (positionSampler, positionSet) = CombinedModel.generateXYZSampler(reference)
    val shapePancake = CombinedModel.generateShapePancake(reference, gpShape, positionSampler, positionSet, numComponents)

    println(s"Writing to files...")
    for (i <- 0 until numSamples) {
      val pointset = UnstructuredPointsDomain[_3D](shapePancake.gpModel.sample().data)
      val VCM = VertexColorMesh3D(TriangleMesh3D(pointset, reference.shape.triangulation), reference.color)
      renderVCM(VCM, Path("scan/") + "/" + folderName + s"/test$i.png")
    }
  }

  def renderFullModel(fromBFM: (VertexColorMesh3D, Map[String, Landmark[_3D]]),
                      gpColor: GaussianProcess[_3D, EuclideanVector[_3D]],
                      getColorSampler: VertexColorMesh3D => (Sampler[_3D], UPD3D),
                      gpShape: GaussianProcess[_3D, EuclideanVector[_3D]],
                      folderName: String, modelName: String, numSamples: Int, numComponents: Int = 199)(implicit rand: Random): Unit = {
    val (reference, lmOnMean) = fromBFM
    val (colorSampler, colorSet) = getColorSampler(reference)
    val (positionSampler, positionSet) = CombinedModel.generateXYZSampler(reference)
    val colorPancake = CombinedModel.generateColorPancake(reference, gpColor, colorSampler, colorSet, numComponents)
    val shapePancake = CombinedModel.generateShapePancake(reference, gpShape, positionSampler, positionSet, numComponents)
    val momo = MoMo(reference.shape, shapePancake, colorPancake).withLandmarks(lmOnMean)

    println(s"Writing to files...")
    for (i <- 0 until numSamples) {
      renderVCM(momo.sample(), Path("scan/") + "/" + folderName + s"/test$i.png")
    }
    println("Saving model...")
    uglyWrite(momo, Path("scan/") + "/" + modelName + ".h5")
  }

  def buildMultiSpaceVCMs(reference: VertexColorMesh3D,
                          gpColorRGB: GaussianProcess[_3D, EuclideanVector[_3D]],
                          gpColorXYZ: GaussianProcess[_3D, EuclideanVector[_3D]],
                          numSamples:Int, numComponents: Int)(implicit rand: Random): IndexedSeq[VertexColorMesh3D] = {
    val refShape = reference.shape

    println("Beginning RGB phase...")
    val (colorSampler, colorSet) = CombinedModel.generateRGBSampler(reference)
    val RGBColorPancake = CombinedModel.generateColorPancake(reference, gpColorRGB, colorSampler, colorSet, numComponents)
    println("RGB phase complete, beginning XYZ phase...")
    val (positionSampler, positionSet) = CombinedModel.generateXYZSampler(reference)
    val XYZColorPancake = CombinedModel.generateColorPancake(reference, gpColorXYZ, positionSampler, positionSet, numComponents)
    println("XYZ phase complete.")

    println("Generating samples...")
    for (_ <- 0 until numSamples) yield {
      val RGBColorStruct = RGBColorPancake.gpModel.sample().data.toIndexedSeq
      val XYZColorStruct = XYZColorPancake.gpModel.sample().data.toIndexedSeq
      val colorComponent = for (x <- RGBColorStruct.zip(XYZColorStruct)) yield ((x._1 + x._2)/2).toRGBA
      VertexColorMesh3D(refShape, SurfacePointProperty[RGBA](refShape.triangulation, colorComponent))
    }
  }

  // Does NOT include landmarks!
  def buildMultiSpaceModel(reference: VertexColorMesh3D,
                           gpShape: GaussianProcess[_3D, EuclideanVector[_3D]],
                           samples: Array[VertexColorMesh3D], numComponents: Int)
                          (implicit rand: Random): (MoMo, PancakeDLRGP[_3D, UPD3D, Point[_3D]]) = {
    val refShape = reference.shape
    val meshes = samples

    println("Creating shape pancake...")
    val (positionSampler, positionSet) = CombinedModel.generateXYZSampler(reference) // slightly inefficient
    val shapePancake = CombinedModel.generateShapePancake(reference, gpShape, positionSampler, positionSet, numComponents)
    println("Building model from samples...")
    val colorLearnedModel = MoMo.buildFromRegisteredSamples(refShape, meshes, meshes, 0.0, 0.0)
    (MoMo(refShape, shapePancake, colorLearnedModel.color), shapePancake)
  }

  def renderMultiSpaceModel(fromBFM: (VertexColorMesh3D, Map[String, Landmark[_3D]]),
                            gpColorRGB: GaussianProcess[_3D, EuclideanVector[_3D]],
                            gpColorXYZ: GaussianProcess[_3D, EuclideanVector[_3D]],
                            gpShape: GaussianProcess[_3D, EuclideanVector[_3D]],
                            imageName: String, outputName: String, numSamples: Int,
                            numComponents: Int = 199)(implicit rand: Random): Unit = {
    val meshes = buildMultiSpaceVCMs(fromBFM._1, gpColorRGB, gpColorXYZ, numSamples, numComponents).toArray
    val (momoNoLMs, shapePancake) = buildMultiSpaceModel(fromBFM._1, gpShape, meshes, numComponents)
    for ((vcm, i) <- meshes.zipWithIndex) {
      val newVCM = VertexColorMesh3D(TriangleMesh3D(shapePancake.sample().data, vcm.shape.triangulation), vcm.color)
      renderVCM(newVCM, Path("scan/") + "/" + imageName + s"/test$i.png")
    }

    println("Saving model...")
    uglyWrite(momoNoLMs.withLandmarks(fromBFM._2), Path("scan/") + "/" + outputName + ".h5")
  }

  // Does NOT include landmarks!
  def createMultiSpaceModel(reference: VertexColorMesh3D,
                            gpColorRGB: GaussianProcess[_3D, EuclideanVector[_3D]],
                            gpColorXYZ: GaussianProcess[_3D, EuclideanVector[_3D]],
                            gpShape: GaussianProcess[_3D, EuclideanVector[_3D]],
                            numSamples: Int, numComponents: Int = 199)(implicit rand: Random): MoMo = {
    val meshes = buildMultiSpaceVCMs(reference, gpColorRGB, gpColorXYZ, numSamples, numComponents).toArray
    buildMultiSpaceModel(reference, gpShape, meshes, numComponents)._1
  }

  def createStandardModel(reference: VertexColorMesh3D)(implicit rand: Random): MoMo = {
    createMultiSpaceModel(
      reference, CombinedModel.RGBStandardGaussianProcess(), CombinedModel.XYZStandardGaussianProcess(),
      CombinedModel.shapeGaussianProcess(), 200
    )
  }

  def renderMultiSpaceWithDifferentMeans(fromBFM: (VertexColorMesh3D, Map[String, Landmark[_3D]]), plyPaths: String,
                                         gpColorRGB: GaussianProcess[_3D, EuclideanVector[_3D]],
                                         gpColorXYZ: GaussianProcess[_3D, EuclideanVector[_3D]],
                                         gpShape: GaussianProcess[_3D, EuclideanVector[_3D]], numSamples: Int,
                                         modelFolder: String, numComponents: Int = 199)(implicit rand: Random): Unit = {
    for (fp <- new File(plyPaths).listFiles.toIndexedSeq.filter(_.getName.endsWith(".ply"))) {
      println("Starting " + fp.getName)
      val reference = scalismo.faces.io.MeshIO.read(fp).get.vertexColorMesh3D.get
      val prefix = Path("scan/") + "/" + modelFolder + "/" + fp.getName.split('.')(0)
      val lms = transferLandmarks(fromBFM._2, reference.shape, fromBFM._1.shape)

      val meshes = buildMultiSpaceVCMs(reference, gpColorRGB, gpColorXYZ, numSamples, numComponents).toArray
      val (momoNoLMs, shapePancake) = buildMultiSpaceModel(fromBFM._1, gpShape, meshes, numComponents)
      for ((vcm, i) <- meshes.zipWithIndex) {
        val newVCM = VertexColorMesh3D(TriangleMesh3D(shapePancake.sample().data, vcm.shape.triangulation), vcm.color)
        renderVCM(newVCM, prefix + s"/vcm$i.png")
      }

      println("momo done, saving model...")
      uglyWrite(momoNoLMs.withLandmarks(lms), prefix + ".h5")
    }
  }
}