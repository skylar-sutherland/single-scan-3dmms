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

package fitting

import java.io.File

import breeze.linalg.min
import onemesh.CombinedModelHelpers
import scalismo.color.{RGB, RGBA}
import scalismo.faces.deluminate.SphericalHarmonicsOptimizer
import scalismo.faces.image.PixelImage
import scalismo.faces.io.{PixelImageIO, RenderParameterIO, TLMSLandmarksIO}
import scalismo.faces.mesh.MeshSurfaceSampling
import scalismo.faces.parameters.RenderParameter
import scalismo.faces.sampling.face.evaluators.PixelEvaluators._
import scalismo.faces.sampling.face.evaluators.PointEvaluators.IsotropicGaussianPointEvaluator
import scalismo.faces.sampling.face.evaluators.PriorEvaluators.{GaussianShapePrior, GaussianTexturePrior}
import scalismo.faces.sampling.face.evaluators._
import scalismo.faces.sampling.face.loggers._
import scalismo.faces.sampling.face.proposals.ImageCenteredProposal.implicits._
import scalismo.faces.sampling.face.proposals.ParameterProposals.implicits._
import scalismo.faces.sampling.face.proposals.SphericalHarmonicsLightProposals._
import scalismo.faces.sampling.face.proposals._
import scalismo.faces.sampling.face.{MoMoRenderer, ParametricLandmarksRenderer, ParametricModel}
import scalismo.geometry.{EuclideanVector2D, EuclideanVector3D, _2D}
import scalismo.sampling.algorithms.MetropolisHastings
import scalismo.sampling.evaluators.ProductEvaluator
import scalismo.sampling.loggers.ChainStateLogger.implicits._
import scalismo.sampling.loggers.ChainStateLoggerContainer.implicits._
import scalismo.sampling.loggers.BestSampleLogger
import scalismo.sampling.proposals.MixtureProposal.implicits._
import scalismo.sampling.proposals.{MetropolisFilterProposal, MixtureProposal}
import scalismo.sampling.{ProposalGenerator, TransitionProbability}
import scalismo.utils.Random

/* Adapted from the Basel Face Registration Pipeline
https://github.com/unibas-gravis/basel-face-pipeline/blob/master/src/main/scala/fitting/StandardFitScript.scala
That project was released via the Apache License 2.0 by the University of Basel, and uses the same license file as was
provided with this project.

This fitscript with its evaluators and the proposal distribution follows closely the proposed setting of:

Markov Chain Monte Carlo for Automated Face Image Analysis
Sandro Sch�nborn, Bernhard Egger, Andreas Morel-Forster and Thomas Vetter
International Journal of Computer Vision 123(2), 160-183 , June 2017
DOI: http://dx.doi.org/10.1007/s11263-016-0967-5

To understand the concepts behind the fitscript and the underlying methods there is a tutorial on:
http://gravis.dmi.unibas.ch/pmm/ */

object StandardFitScript {
  // Mixture proposal format:
  // a *: x + b *: y + ... means draw from x with probability a, draw from y with probability b, etc.
  // C, I, F mean coarse, intermediate, fine; C, F, and HF also mean the same
  // translation is in plane orthogonal to depth axis
  // scaling proposal it changes the focal length
  // it might make sense to do two stages: first a higher probability of coarse, then a lower probability
  // rotation rotates in the mesh, in mesh coordinates
  // maybe do yaw, pitch and roll simultaneously
  // nonTranslationPoseProposal tries to fix the issue that rotation, distance, and focal length change modify everything
  // translation is introduced so that the landmarks stay fixed; remember that these are mesh landmarks, not image ones

  def translationProposal(implicit rnd: Random): ProposalGenerator[RenderParameter] with TransitionProbability[RenderParameter] = {
    val translationC = GaussianTranslationProposal(EuclideanVector2D(300f, 300f)).toParameterProposal
    val translationF = GaussianTranslationProposal(EuclideanVector2D(50f, 50f)).toParameterProposal
    val translationHF = GaussianTranslationProposal(EuclideanVector2D(10f, 10f)).toParameterProposal
    MixtureProposal(0.2 *: translationC + 0.2 *: translationF + 0.6 *: translationHF)
  }

  def nonTranslationPoseProposal(implicit rnd: Random): ProposalGenerator[RenderParameter] with TransitionProbability[RenderParameter] = {
    val yawProposalC = GaussianRotationProposal(EuclideanVector3D.unitY, 0.75f)
    val yawProposalI = GaussianRotationProposal(EuclideanVector3D.unitY, 0.10f)
    val yawProposalF = GaussianRotationProposal(EuclideanVector3D.unitY, 0.01f)
    val rotationYaw = MixtureProposal(0.1 *: yawProposalC + 0.4 *: yawProposalI + 0.5 *: yawProposalF)

    val pitchProposalC = GaussianRotationProposal(EuclideanVector3D.unitX, 0.75f)
    val pitchProposalI = GaussianRotationProposal(EuclideanVector3D.unitX, 0.10f)
    val pitchProposalF = GaussianRotationProposal(EuclideanVector3D.unitX, 0.01f)
    val rotationPitch = MixtureProposal(0.1 *: pitchProposalC + 0.4 *: pitchProposalI + 0.5 *: pitchProposalF)

    val rollProposalC = GaussianRotationProposal(EuclideanVector3D.unitZ, 0.75f)
    val rollProposalI = GaussianRotationProposal(EuclideanVector3D.unitZ, 0.10f)
    val rollProposalF = GaussianRotationProposal(EuclideanVector3D.unitZ, 0.01f)
    val rotationRoll = MixtureProposal(0.1 *: rollProposalC + 0.4 *: rollProposalI + 0.5 *: rollProposalF)

    val rotationProposal = MixtureProposal(0.5 *: rotationYaw + 0.3 *: rotationPitch + 0.2 *: rotationRoll).toParameterProposal

    val distanceProposalC = GaussianDistanceProposal(500f, compensateScaling = true).toParameterProposal
    val distanceProposalF = GaussianDistanceProposal(50f, compensateScaling = true).toParameterProposal
    val distanceProposalHF = GaussianDistanceProposal(5f, compensateScaling = true).toParameterProposal
    val distanceProposal = MixtureProposal(0.2 *: distanceProposalC + 0.6 *: distanceProposalF + 0.2 *: distanceProposalHF)

    val scalingProposalC = GaussianScalingProposal(0.15f).toParameterProposal
    val scalingProposalF = GaussianScalingProposal(0.05f).toParameterProposal
    val scalingProposalHF = GaussianScalingProposal(0.01f).toParameterProposal
    val scalingProposal = MixtureProposal(0.2 *: scalingProposalC + 0.6 *: scalingProposalF + 0.2 *: scalingProposalHF)

    MixtureProposal(rotationProposal + distanceProposal + scalingProposal)
  }

  /* Collection of all pose related proposals */
  def defaultPoseProposal(lmRenderer: ParametricLandmarksRenderer)(implicit rnd: Random):
  ProposalGenerator[RenderParameter] with TransitionProbability[RenderParameter] = {
    val poseMovingNoTransProposal = nonTranslationPoseProposal
    val centerREyeProposal = poseMovingNoTransProposal.centeredAt("right.eye.corner_outer", lmRenderer).get
    val centerLEyeProposal = poseMovingNoTransProposal.centeredAt("left.eye.corner_outer", lmRenderer).get
    val centerRLipsProposal = poseMovingNoTransProposal.centeredAt("right.lips.corner", lmRenderer).get
    val centerLLipsProposal = poseMovingNoTransProposal.centeredAt("left.lips.corner", lmRenderer).get
    MixtureProposal(centerREyeProposal + centerLEyeProposal + centerRLipsProposal + centerLLipsProposal + 0.2 *: translationProposal)
  }

  def nonFacePoseProposal(implicit rnd: Random): ProposalGenerator[RenderParameter] with TransitionProbability[RenderParameter] = {
    MixtureProposal(0.8 *: nonTranslationPoseProposal + 0.2 *: translationProposal)
  }

  // Random perturbation based illumination proposals
  def perturbIllumationProposal(implicit rnd: Random): ProposalGenerator[RenderParameter] with TransitionProbability[RenderParameter] = {
    val lightSHPert = SHLightPerturbationProposal(0.001f, fixIntensity = true)
    val lightSHIntensity = SHLightIntensityProposal(0.1f)
    val lightSHBandMixter = SHLightBandEnergyMixer(0.1f)
    val lightSHSpatial = SHLightSpatialPerturbation(0.05f)
    val lightSHColor = SHLightColorProposal(0.01f)
    MixtureProposal(lightSHSpatial + lightSHBandMixter + lightSHIntensity + lightSHPert + lightSHColor).toParameterProposal
  }

  /* Collection of all illumination related proposals */
  def defaultIlluminationProposal(modelRenderer: ParametricModel, target: PixelImage[RGBA])(implicit rnd: Random):
  ProposalGenerator[RenderParameter] with TransitionProbability[RenderParameter] = {
    // this uses optimization to directly estimate the illumination model
    val shOpt = SphericalHarmonicsOptimizer(modelRenderer, target)
    val shOptimizerProposal = SHLightSolverProposal(shOpt, MeshSurfaceSampling.sampleUniformlyOnSurface(100))
    MixtureProposal((5f / 6f) *: perturbIllumationProposal + (1f / 6f) *: shOptimizerProposal)
  }

  /* Collection of all statistical model (shape, texture) related proposals */
  def neutralMorphableModelProposal(implicit rnd: Random): ProposalGenerator[RenderParameter] with TransitionProbability[RenderParameter] = {
    // once again, C, F, HF are coarse, intermediate and fine
    // shapeScaleProposal and textureScaleProposal scale all the parameters in the shape and color DLRGPs respecitvely
    // shapeC/F/HF and textureC/F/HF add Gaussian noise to all parameters simultaneously in the shape and color DLRGPs respectively

    val shapeC = GaussianMoMoShapeProposal(0.2f)
    val shapeF = GaussianMoMoShapeProposal(0.1f)
    val shapeHF = GaussianMoMoShapeProposal(0.025f)
    val shapeScaleProposal = GaussianMoMoShapeCaricatureProposal(0.2f)
    val shapeProposal = MixtureProposal(0.1f *: shapeC + 0.5f *: shapeF + 0.2f *: shapeHF + 0.2f *: shapeScaleProposal).toParameterProposal

    val textureC = GaussianMoMoColorProposal(0.2f)
    val textureF = GaussianMoMoColorProposal(0.1f)
    val textureHF = GaussianMoMoColorProposal(0.025f)
    val textureScale = GaussianMoMoColorCaricatureProposal(0.2f)
    val textureProposal = MixtureProposal(0.1f *: textureC + 0.5f *: textureF + 0.2 *: textureHF + 0.2f *: textureScale).toParameterProposal

    MixtureProposal(shapeProposal + textureProposal)
  }

  /* Collection of all color transform proposals */
  def defaultColorProposal(implicit rnd: Random): ProposalGenerator[RenderParameter] with TransitionProbability[RenderParameter] = {
    // this represents the color space of the camera, i.e. if the camera has a bias towards colors
    val colorC = GaussianColorProposal(RGB(0.01f, 0.01f, 0.01f), 0.01f, RGB(1e-4f, 1e-4f, 1e-4f))
    val colorF = GaussianColorProposal(RGB(0.001f, 0.001f, 0.001f), 0.01f, RGB(1e-4f, 1e-4f, 1e-4f))
    val colorHF = GaussianColorProposal(RGB(0.0005f, 0.0005f, 0.0005f), 0.01f, RGB(1e-4f, 1e-4f, 1e-4f))
    MixtureProposal(0.2f *: colorC + 0.6f *: colorF + 0.2f *: colorHF).toParameterProposal
  }

  def neutralMorphableModelProposalShape(implicit rnd: Random): ProposalGenerator[RenderParameter] with TransitionProbability[RenderParameter] = {
    val shapeC = GaussianMoMoShapeProposal(0.2f)
    val shapeF = GaussianMoMoShapeProposal(0.1f)
    val shapeHF = GaussianMoMoShapeProposal(0.025f)
    val shapeScaleProposal = GaussianMoMoShapeCaricatureProposal(0.2f)
    val shapeProposal = MixtureProposal(0.1f *: shapeC + 0.5f *: shapeF + 0.2f *: shapeHF + 0.2f *: shapeScaleProposal).toParameterProposal
    MixtureProposal(shapeProposal)
  }

  def fit(targetFn : String, initFn: String, outputDir: String, modelRenderer: MoMoRenderer, isFace: Boolean = true)
         (implicit rnd: Random): RenderParameter = {
    // note, does not use an expression model
    val target = PixelImageIO.read[RGBA](new File(targetFn)).get
    val targetLM = TLMSLandmarksIO.read2D(new File(initFn)).get.filter(lm => lm.visible)
    PixelImageIO.write(target, new File(s"$outputDir/target.png")).get
    // initialize with default render parameters, i.e. all the parameters that are used to generate the image
    // i.e. pose, distance, illumination etc. plus shape and color; everything that has proposals generated for it
    val init: RenderParameter = CombinedModelHelpers.lowResSquareRenderParameter.fitToImageSize(target.width, target.height)
    val sdev = 0.043f

    // This is effectively the logpdf of the generative functions
    // foreground and background are treated differently
    /* Foreground Evaluator */
    val pixEval = IsotropicGaussianPixelEvaluator(sdev)
    /* Background Evaluator */
    val histBGEval = HistogramRGB.fromImageRGBA(target, 25)
    /* Pixel Evaluator */
    val imgEval = IndependentPixelEvaluator(pixEval, histBGEval) // combine pixEval and histBGEval
    /* Prior Evaluator */
    val priorEval = ProductEvaluator(GaussianShapePrior(0, 1), GaussianTexturePrior(0, 1)) // prior for the model (shape/color)
    /* Image Evaluator */
    val allEval = ImageRendererEvaluator(modelRenderer, imgEval.toDistributionEvaluator(target)) // combine imgEval and priorEval
    /* Landmarks Evaluator */
    // these are just for landmarks, to take into account that given landmarks won't be perfectly accurate
    // since we are doing amortized inference instead of landmarks this can be skipped
    val sdevLM = if (isFace) 4.0 else 2.0  //lm click uncertainty in pixels, should be related to image/face size
    val pointEval = IsotropicGaussianPointEvaluator[_2D](sdevLM)
    val landmarksEval = LandmarkPointEvaluator(targetLM, pointEval, modelRenderer)

    // keep track of best sample
    val bestFileLogger = ParametersFileBestLogger(allEval, new File(s"$outputDir/fit-best.rps"))
    val bestSampleLogger = BestSampleLogger(allEval)
    //val parametersLogger = ParametersFileLogger(new File(s"$outputDir/"), "mc-")
    val fitLogger = bestFileLogger :+ bestSampleLogger

    // pose proposal
    val totalPose = if (isFace) defaultPoseProposal(modelRenderer) else nonFacePoseProposal
    //light proposals
    val lightProposal = if (isFace) defaultIlluminationProposal(modelRenderer, target) else perturbIllumationProposal
    //color proposals
    val colorProposal = defaultColorProposal
    //momo proposals
    val momoProposal = neutralMorphableModelProposal
    // full proposal filtered by the landmark and prior Evaluator
    // MixtureProposal normalizes; so effectively the probabilities are 1/7, 1/7, 3/7, 2/7
    val propmix = MixtureProposal(totalPose + colorProposal + 3f *: momoProposal + 2f *: lightProposal)
    // this is a hack to make it faster
    // evaluate cheapest likelihood first
    // 1) if the prior is low, skip rendering
    // 2) then if the landmarks are low, only do the landmarks first, since that's only rendering a few points
    // 3) then we render
    val proposal = MetropolisFilterProposal(MetropolisFilterProposal(propmix, landmarksEval), priorEval)

    //image chain
    val imageFitter = MetropolisHastings(proposal, allEval)
    val poseFitter = MetropolisHastings(totalPose, landmarksEval)
    println("everything setup. starting fitter ...")

    val nShape = min(50, modelRenderer.model.neutralModel.shape.rank) // note that you can change 50 to change the max number of pcs
    val nColor = min(50, modelRenderer.model.neutralModel.color.rank) // see above
    val initDefault: RenderParameter = CombinedModelHelpers.lowResSquareRenderParameter.fitToImageSize(target.width, target.height)
    val initFixed = initDefault.withMoMo(init.momo.withNumberOfCoefficients(nShape, nColor, 5))

    //landmark chain for initialisation
    // this chain only renders the landmarks and just tries to get the pose parameters good
    // it saves and renders the result of the pose estimation for debugging
    val initLMSamples: IndexedSeq[RenderParameter] = poseFitter.iterator(initFixed).take(5000).toIndexedSeq
    val lmScores = initLMSamples.map(rps => (landmarksEval.logValue(rps), rps))
    val bestLM = lmScores.maxBy(_._1)._2
    RenderParameterIO.write(bestLM, new File(s"$outputDir/fitter-lminit.rps")).get
    val imgLM = modelRenderer.renderImage(bestLM)
    PixelImageIO.write(imgLM, new File(s"$outputDir/fitter-lminit.png")).get

    // shape-only fitting; we skip this for faces
    val bestLMShape = if (isFace) bestLM else {
      val shapemix = MixtureProposal(totalPose + 3f *: neutralMorphableModelProposalShape)
      val proposalShape = MetropolisFilterProposal(MetropolisFilterProposal(shapemix, landmarksEval), priorEval)
      val landmarkFitter = MetropolisHastings(proposalShape, landmarksEval)
      val fitsamplesLM = landmarkFitter.iterator(bestLM).take(5000).toIndexedSeq
      val lmScoresShape = fitsamplesLM.map(rps => (landmarksEval.logValue(rps), rps))
      val bestLMShape_tmp = lmScoresShape.maxBy(_._1)._2
      val imgBestS = modelRenderer.renderImage(bestLMShape_tmp)
      PixelImageIO.write(imgBestS, new File(s"$outputDir/fitter-best-shape.png")).get
      bestLMShape_tmp
    }

    // image chain, fitting
    // .take(5000) is default, .take(1000) for faster performance, and .take(10000) for better fitting
    println("Starting full fitting...")
    val fitsamples = imageFitter.iterator(bestLMShape).loggedWith(fitLogger).take(5000).toIndexedSeq
    val best = bestSampleLogger.currentBestSample().get
    val imgBest = modelRenderer.renderImage(best)
    PixelImageIO.write(imgBest, new File(s"$outputDir/fitter-best.png")).get
    best
  }
}