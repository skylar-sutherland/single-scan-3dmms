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
import scalismo.faces.io.MoMoIO
import scalismo.geometry.{Landmark, _3D}
import scalismo.mesh.VertexColorMesh3D
import scalismo.utils.Random

import scala.reflect.io.Path

object CreateCombinedModel {
  def bfmMean(): (VertexColorMesh3D, Map[String, Landmark[_3D]]) = {
    CombinedModelHelpers.getReference(
      "model2019_bfm_nomouth_noexpression_lvl1_mean.ply", "model2019_bfm_nomouth_noexpression_lvl1_landmarks.json"
    )
  }

  def bfmLvl3Mean(): (VertexColorMesh3D, Map[String, Landmark[_3D]]) = {
    CombinedModelHelpers.getReference(
      "model2019_bfm_nomouth_noexpression_mean.ply", "model2019_bfm_nomouth_noexpression_landmarks.json"
    )
  }

  def bfm17Mean(): (VertexColorMesh3D, Map[String, Landmark[_3D]]) = {
    val bfm17 = MoMoIO.read(new File(Path("pipeline-data/") + "/data/model2017-1_face12_nomouth.h5")).get
    val reference = bfm17.neutralModel.mean
    val lmOnMean = CombinedModelHelpers.transferLandmarks(bfm17.landmarks, reference.shape, bfm17.neutralModel.referenceMesh)
    (reference, lmOnMean)
  }

  def birdMean(): (VertexColorMesh3D, Map[String, Landmark[_3D]]) = {
    CombinedModelHelpers.getReference("bird_reference_gradient.ply", "bird_landmarks.json")
  }
}

/* Generate 200 sampled images from the RGB-based albedo deformation model and the reference shape. */
object RenderRGBModelStandardSmooth extends App {
  scalismo.initialize(); val seed = 1986L; implicit val rnd: Random = scalismo.utils.Random(seed)
  CombinedModelHelpers.renderSmoothModel(
    CreateCombinedModel.bfmMean()._1, CombinedModel.RGBStandardGaussianProcess(),
    CombinedModel.generateRGBSampler, "bfm19_gp_renders/smoothstandard_rgbimgs", 200
  )
}

/* Generate 200 sampled images from the XYZ-based albedo deformation model and the reference shape. */
object RenderXYZModelStandardSmooth extends App {
  scalismo.initialize(); val seed = 1986L; implicit val rnd: Random = scalismo.utils.Random(seed)
  CombinedModelHelpers.renderSmoothModel(
    CreateCombinedModel.bfmMean()._1, CombinedModel.XYZStandardGaussianProcess(),
    CombinedModel.generateXYZSampler, "bfm19_gp_renders/smoothstandard_xyzimgs", 200
  )
}

/* Generate 200 sampled images from the RGB-based albedo deformation model with color-channel correlations and the
reference shape. */
object RenderRGBModelCorrelatedSmooth extends App {
  scalismo.initialize(); val seed = 1986L; implicit val rnd: Random = scalismo.utils.Random(seed)
  CombinedModelHelpers.renderSmoothModel(
    CreateCombinedModel.bfmMean()._1, CombinedModel.RGBCorrelatedGaussianProcess(),
    CombinedModel.generateRGBSampler, "bfm19_gp_renders/smoothcorrelated_rgbimgs", 200
  )
}

/* Generate 200 sampled images from the XYZ-based albedo deformation model with color-channel correlations and the
reference shape. */
object RenderXYZModelCorrelatedSmooth extends App {
  scalismo.initialize(); val seed = 1986L; implicit val rnd: Random = scalismo.utils.Random(seed)
  CombinedModelHelpers.renderSmoothModel(
    CreateCombinedModel.bfmMean()._1, CombinedModel.XYZCorrelatedGaussianProcess(),
    CombinedModel.generateXYZSampler, "bfm19_gp_renders/smoothcorrelated_xyzimgs", 200
  )
}

/* Generate 200 sampled images from the symmetric XYZ-based albedo deformation model and the reference shape. */
object RenderXYZModelSymmetricSmooth extends App {
  scalismo.initialize(); val seed = 1986L; implicit val rnd: Random = scalismo.utils.Random(seed)
  CombinedModelHelpers.renderSmoothModel(
    CreateCombinedModel.bfmMean()._1, CombinedModel.XYZSymmetricGaussianProcess(),
    CombinedModel.generateXYZSampler, "bfm19_gp_renders/smoothsymmetric_xyzimgs", 200
  )
}

/* Generate 200 sampled images from the standard shape deformation model and the reference albedo. */
object RenderShapeDeformation extends App {
  scalismo.initialize(); val seed = 1986L; implicit val rnd: Random = scalismo.utils.Random(seed)
  CombinedModelHelpers.renderShapeModel(
    CreateCombinedModel.bfmMean()._1, CombinedModel.shapeGaussianProcess(),
    "bfm19_gp_renders/shapeimgs", 200
  )
}

/* Generate 200 sampled images from the symmetric shape deformation model and the reference albedo. */
object RenderSymmetricShapeDeformation extends App {
  scalismo.initialize(); val seed = 1986L; implicit val rnd: Random = scalismo.utils.Random(seed)
  CombinedModelHelpers.renderShapeModel(
    CreateCombinedModel.bfmMean()._1, CombinedModel.symmetricShapeGaussianProcess(),
    "bfm19_gp_renders/symmetric_shapeimgs", 200
  )
}

/* Generate 200 sampled images from the RGB-based albedo deformation model and the standard shape deformation model, and
save the model as a MoMo to a file. */
object RenderRGBModelStandard extends App {
  scalismo.initialize(); val seed = 1986L; implicit val rnd: Random = scalismo.utils.Random(seed)
  CombinedModelHelpers.renderFullModel(
    CreateCombinedModel.bfmMean(), CombinedModel.RGBStandardGaussianProcess(),
    CombinedModel.generateRGBSampler, CombinedModel.shapeGaussianProcess(),
    "bfm19/standard_rgbimgs", "bfm19/standard_rgbmodel", 200
  )
}

/* Generate 200 sampled images from the RGB-based albedo deformation model with color-channel correlations and the
standard shape deformation model, and save the model as a MoMo to a file. */
object RenderRGBModelCorrelated extends App {
  scalismo.initialize(); val seed = 1986L; implicit val rnd: Random = scalismo.utils.Random(seed)
  CombinedModelHelpers.renderFullModel(
    CreateCombinedModel.bfmMean(), CombinedModel.RGBCorrelatedGaussianProcess(),
    CombinedModel.generateRGBSampler, CombinedModel.shapeGaussianProcess(),
    "bfm19/correlated_rgbimgs", "bfm19/correlated_rgbmodel", 200
  )
}

/* Generate 200 sampled images from the RGB-based albedo deformation model with color-channel correlations and the
symmetric shape deformation model, and save the model as a MoMo to a file. */
object RenderRGBModelSymmetric extends App {
  scalismo.initialize(); val seed = 1986L; implicit val rnd: Random = scalismo.utils.Random(seed)
  CombinedModelHelpers.renderFullModel(
    CreateCombinedModel.bfmMean(), CombinedModel.RGBCorrelatedGaussianProcess(),
    CombinedModel.generateRGBSampler, CombinedModel.symmetricShapeGaussianProcess(),
    "bfm19/symmetric_rgbimgs", "bfm19/symmetric_rgbmodel", 200
  )
}

/* Generate 200 sampled images from the XYZ-based albedo deformation model and the standard shape deformation model, and
save the model as a MoMo to a file. */
object RenderXYZModelStandard extends App {
  scalismo.initialize(); val seed = 1986L; implicit val rnd: Random = scalismo.utils.Random(seed)
  CombinedModelHelpers.renderFullModel(
    CreateCombinedModel.bfmMean(), CombinedModel.XYZStandardGaussianProcess(),
    CombinedModel.generateXYZSampler, CombinedModel.shapeGaussianProcess(),
    "bfm19/standard_xyzimgs", "bfm19/standard_xyzmodel", 200
  )
}

/* Generate 200 sampled images from the XYZ-based albedo deformation model with color-channel correlations and the
standard shape deformation model, and save the model as a MoMo to a file. */
object RenderXYZModelCorrelated extends App {
  scalismo.initialize(); val seed = 1986L; implicit val rnd: Random = scalismo.utils.Random(seed)
  CombinedModelHelpers.renderFullModel(
    CreateCombinedModel.bfmMean(), CombinedModel.XYZCorrelatedGaussianProcess(),
    CombinedModel.generateXYZSampler, CombinedModel.shapeGaussianProcess(),
    "bfm19/correlated_xyzimgs", "bfm19/correlated_xyzmodel", 200
  )
}

/* Generate 200 sampled images from the symmetric XYZ-based albedo deformation model and the symmetric shape deformation
model, and save the model as a MoMo to a file. */
object RenderXYZModelSymmetric extends App {
  scalismo.initialize(); val seed = 1986L; implicit val rnd: Random = scalismo.utils.Random(seed)
  CombinedModelHelpers.renderFullModel(
    CreateCombinedModel.bfmMean(), CombinedModel.XYZSymmetricGaussianProcess(),
    CombinedModel.generateXYZSampler, CombinedModel.symmetricShapeGaussianProcess(),
    "bfm19/symmetric_xyzimgs", "bfm19/symmetric_xyzmodel", 200
  )
}

/* Generate 200 sampled images from the XYZ-based albedo deformation model and the standard shape deformation model with
the bird as reference, and save the model as a MoMo to a file. */
object RenderXYZModelBirdStandard extends App {
  scalismo.initialize(); val seed = 1986L; implicit val rnd: Random = scalismo.utils.Random(seed)
  CombinedModelHelpers.renderFullModel(
    CreateCombinedModel.birdMean(), CombinedModel.XYZStandardGaussianProcess(),
    CombinedModel.generateXYZSampler, CombinedModel.shapeGaussianProcess(), "birds/bird_standard_xyzimgs",
    "birds/bird_standard_xyzmodel", 200
  )
}

/* Generate 200 sampled images from the symmetric XYZ-based albedo deformation model and the symmetric shape deformation
model with the bird as reference, and save the model as a MoMo to a file. */
object RenderXYZModelBirdSymmetric extends App {
  scalismo.initialize(); val seed = 1986L; implicit val rnd: Random = scalismo.utils.Random(seed)
  CombinedModelHelpers.renderFullModel(
    CreateCombinedModel.birdMean(), CombinedModel.XYZSymmetricGaussianProcess(),
    CombinedModel.generateXYZSampler, CombinedModel.symmetricShapeGaussianProcess(), "birds/bird_symmetric_xyzimgs",
    "birds/bird_symmetric_xyzmodel", 200
  )
}

/* Generate 200 sampled images and corresponding VCMs by averaging the changes proposed by the RGB and XYZ albedo
deformation models, and fit a Gaussian process model to them; then combine the resulting albedo model with the standard
shape deformation model, and save the model as a MoMo to a file. */
object RenderMultiSpaceModelStandard extends App {
  scalismo.initialize(); val seed = 1986L; implicit val rnd: Random = scalismo.utils.Random(seed)
  CombinedModelHelpers.renderMultiSpaceModel(
    CreateCombinedModel.bfmMean(), CombinedModel.RGBStandardGaussianProcess(), CombinedModel.XYZStandardGaussianProcess(),
    CombinedModel.shapeGaussianProcess(), "bfm19/standard_multiimgs", "bfm19/standard_combinedmodel", 200
  )
}

/* Generate 200 sampled images and corresponding VCMs by averaging the changes proposed by the RGB and XYZ albedo
deformation models with color-channel correlations, and fit a Gaussian process model to them; then combine the resulting
albedo model with the standard shape deformation model, and save the model as a MoMo to a file. */
object RenderMultiSpaceModelCorrelated extends App {
  scalismo.initialize(); val seed = 1986L; implicit val rnd: Random = scalismo.utils.Random(seed)
  CombinedModelHelpers.renderMultiSpaceModel(
    CreateCombinedModel.bfmMean(), CombinedModel.RGBCorrelatedGaussianProcess(), CombinedModel.XYZCorrelatedGaussianProcess(),
    CombinedModel.shapeGaussianProcess(), "bfm19/correlated_multiimgs","bfm19/correlated_combinedmodel", 200
  )
}

/* Generate 200 sampled images and corresponding VCMs by averaging the changes proposed by the symmetric RGB and XYZ
albedo deformation models, and fit a Gaussian process model to them; then combine the resulting albedo model with the
symmetric shape deformation model, and save the model as a MoMo to a file. */
object RenderMultiSpaceModelSymmetric extends App {
  scalismo.initialize(); val seed = 1986L; implicit val rnd: Random = scalismo.utils.Random(seed)
  CombinedModelHelpers.renderMultiSpaceModel(
    CreateCombinedModel.bfmMean(), CombinedModel.RGBCorrelatedGaussianProcess(), CombinedModel.XYZSymmetricGaussianProcess(),
    CombinedModel.symmetricShapeGaussianProcess(), "bfm19/symmetric_multiimgs", "bfm19/symmetric_combinedmodel", 200
  )
}

/* Generate 200 sampled images and corresponding VCMs by averaging the changes proposed by the RGB and XYZ albedo
deformation models, with the bird as reference, and fit a Gaussian process model to them; then combine the resulting
albedo model with the standard shape deformation model, and save the model as a MoMo to a file. */
object RenderMultiSpaceModelBirdStandard extends App {
  scalismo.initialize(); val seed = 1987L; implicit val rnd: Random = scalismo.utils.Random(seed)
  CombinedModelHelpers.renderMultiSpaceModel(
    CreateCombinedModel.birdMean(), CombinedModel.RGBStandardGaussianProcess(), CombinedModel.XYZStandardGaussianProcess(),
    CombinedModel.shapeGaussianProcess(), "birds/bird_standard_multiimgs", "birds/bird_standard_combinedmodel", 200
  )
}

/* Generate 200 sampled images and corresponding VCMs by averaging the changes proposed by the symmetric RGB and XYZ
albedo deformation models, with the bird as reference, and fit a Gaussian process model to them; then combine the
resulting albedo model with the symmetric shape deformation model, and save the model as a MoMo to a file. */
object RenderMultiSpaceModelBirdSymmetric extends App {
  scalismo.initialize(); val seed = 1987L; implicit val rnd: Random = scalismo.utils.Random(seed)
  CombinedModelHelpers.renderMultiSpaceModel(
    CreateCombinedModel.birdMean(), CombinedModel.RGBCorrelatedGaussianProcess(), CombinedModel.XYZSymmetricGaussianProcess(),
    CombinedModel.symmetricShapeGaussianProcess(), "birds/bird_symmetric_multiimgs",
    "birds/bird_symmetric_combinedmodel", 200
  )
}

/* Generate 200 sampled images and corresponding VCMs by averaging the changes proposed by the symmetric RGB and XYZ
albedo deformation models, with the various face scans as reference, and fit a Gaussian process model to them; then
combine the resulting albedo models with the symmetric shape deformation model, and save the models as MoMos.  Requires
that Align (in Registration.scala) has been run. */
object RenderMultiSpaceSymmetricWithDifferentMeans extends App {
  scalismo.initialize(); val seed = 1986L; implicit val rnd: Random = scalismo.utils.Random(seed)
  CombinedModelHelpers.renderMultiSpaceWithDifferentMeans(
    CreateCombinedModel.bfm17Mean(), Path("pipeline-data/") + "/registration/aligned_meshes/",
    CombinedModel.RGBCorrelatedGaussianProcess(), CombinedModel.XYZSymmetricGaussianProcess(),
    CombinedModel.symmetricShapeGaussianProcess(), 200,"bfmscan_symmetric_multimodels"
  )
}

/* Generate 200 sampled images and corresponding VCMs by averaging the changes proposed by the RGB and XYZ albedo
deformation models, with the high-resolution BFM mean as reference, and fit a Gaussian process model to them; then
combine the resulting albedo model with the standard shape deformation model, and save the model as a MoMo to a file. */
object RenderMultiSpaceModelLvl3Standard extends App {
  scalismo.initialize(); val seed = 1986L; implicit val rnd: Random = scalismo.utils.Random(seed)
  CombinedModelHelpers.renderMultiSpaceModel(
    CreateCombinedModel.bfmLvl3Mean(), CombinedModel.RGBStandardGaussianProcess(), CombinedModel.XYZStandardGaussianProcess(),
    CombinedModel.shapeGaussianProcess(), "bfm19_lvl3/lvl3_standard_multiimgs",
    "bfm19_lvl3/lvl3_standard_combinedmodel", 200
  )
}

// Requires all 4 bird models.
object SampleFromBirdModels extends App {
  scalismo.initialize(); val seed = 1986L; implicit val rnd: Random = scalismo.utils.Random(seed)
  val bird_start = Path("scan/") + "/bird_samples/"
  val bird_reference_vcm = CreateCombinedModel.birdMean()._1
  CombinedModelHelpers.renderVCM(bird_reference_vcm, bird_start + "/front/reference.png", RGBA.White)
  CombinedModelHelpers.renderFromSide(bird_reference_vcm, math.Pi / 2, bird_start + "/side/reference.png", RGBA.White)
  val birdStandard = MoMoIO.read(new File(Path("scan/") + "/birds/bird_standard_combinedmodel.h5")).get
  val birdSymmetric = MoMoIO.read(new File(Path("scan/") + "/birds/bird_symmetric_combinedmodel.h5")).get
  val birdStandardXYZ = MoMoIO.read(new File(Path("scan/") + "/birds/bird_standard_xyzmodel.h5")).get
  val birdSymmetricXYZ = MoMoIO.read(new File(Path("scan/") + "/birds/bird_symmetric_xyzmodel.h5")).get
  for (i <- 0 until 10) {
    val bird_standard_vcm = birdStandard.neutralModel.sample()
    val bird_symmetric_vcm = birdSymmetric.neutralModel.sample()
    val bird_standard_xyz_vcm = birdStandardXYZ.neutralModel.sample()
    val bird_symmetric_xyz_vcm = birdSymmetricXYZ.neutralModel.sample()
    CombinedModelHelpers.renderVCM(bird_standard_vcm, bird_start + s"/front/standard_sample_$i.png", RGBA.White)
    CombinedModelHelpers.renderVCM(bird_symmetric_vcm, bird_start + s"/front/symmetric_sample_$i.png", RGBA.White)
    CombinedModelHelpers.renderFromSide(bird_standard_vcm, math.Pi / 2, bird_start + s"/side/standard_sample_$i.png", RGBA.White)
    CombinedModelHelpers.renderFromSide(bird_symmetric_vcm, math.Pi / 2, bird_start + s"/side/symmetric_sample_$i.png", RGBA.White)
    CombinedModelHelpers.renderVCM(bird_standard_xyz_vcm, bird_start + s"/front/standard_xyz_sample_$i.png", RGBA.White)
    CombinedModelHelpers.renderVCM(bird_symmetric_xyz_vcm, bird_start + s"/front/symmetric_xyz_sample_$i.png", RGBA.White)
    CombinedModelHelpers.renderFromSide(bird_standard_xyz_vcm, math.Pi / 2, bird_start + s"/side/standard_xyz_sample_$i.png", RGBA.White)
    CombinedModelHelpers.renderFromSide(bird_symmetric_xyz_vcm, math.Pi / 2, bird_start + s"/side/symmetric_xyz_sample_$i.png", RGBA.White)
  }
}