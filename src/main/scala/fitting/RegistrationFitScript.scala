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

import breeze.stats.distributions.Gaussian
import scalismo.color.RGBA
import scalismo.faces.io.{MeshIO, PixelImageIO, RenderParameterIO}
import scalismo.faces.mesh.ColorNormalMesh3D
import scalismo.faces.parameters.{ParametricRenderer, RenderParameter, SphericalHarmonicsLight}
import scalismo.faces.sampling.face.evaluators.PixelEvaluators._
import scalismo.faces.sampling.face.evaluators.PriorEvaluators.{GaussianShapePrior, GaussianTexturePrior}
import scalismo.faces.sampling.face.evaluators.{ImageRendererEvaluator, _}
import scalismo.faces.sampling.face.loggers._
import scalismo.faces.sampling.face.proposals.ParameterProposals.implicits._
import scalismo.faces.sampling.face.proposals._
import scalismo.faces.sampling.face.MoMoRenderer
import scalismo.sampling.algorithms.MetropolisHastings
import scalismo.sampling.evaluators.ProductEvaluator
import scalismo.sampling.loggers.BestSampleLogger
import scalismo.sampling.loggers.ChainStateLogger.implicits._
import scalismo.sampling.loggers.ChainStateLoggerContainer.implicits._
import scalismo.sampling.proposals.MixtureProposal.implicits._
import scalismo.sampling.proposals.{MetropolisFilterProposal, MixtureProposal}
import scalismo.sampling.{DistributionEvaluator, ProposalGenerator, TransitionProbability}
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

case class ShapeRenderEvaluator (modelRenderer: MoMoRenderer, targetMesh: ColorNormalMesh3D, sdev: Double)
  extends DistributionEvaluator[RenderParameter] {

  val dist: Gaussian = breeze.stats.distributions.Gaussian(0.0, sdev)
  override def logValue(rps: RenderParameter): Double = {
    val mesh = modelRenderer.renderMesh(rps)
    val distances = mesh.shape.pointSet.points.toIndexedSeq.map(p =>{
      val closest = targetMesh.shape.pointSet.findClosestPoint(p)
      (p - closest.point.toVector).toVector.norm
    })
    val avg = distances.sum/mesh.shape.pointSet.numberOfPoints
    dist.logPdf(avg)
  }
  override def toString: String = "MeshRendererEvaluator"
}

object RegistrationFitScript {
  /* Collection of all statistical model (shape, texture) related proposals */
  def neutralMorphableModelProposal(implicit rnd: Random):
  ProposalGenerator[RenderParameter] with TransitionProbability[RenderParameter] = {

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

    MixtureProposal(shapeProposal + textureProposal )
  }

  def fit(targetMeshFn: String, outputDir: String, modelRenderer: MoMoRenderer, albedoConsistency: Boolean)(implicit rnd: Random): Unit = {
    val nShape = modelRenderer.model.neutralModel.shape.rank
    val nColor = modelRenderer.model.neutralModel.color.rank
    val defaultInit = RenderParameter.defaultSquare.withEnvironmentMap(SphericalHarmonicsLight.ambientWhite)
    val init = defaultInit.fitToImageSize(512,512).withMoMo(defaultInit.momo.withNumberOfCoefficients(nShape, nColor, 5))

    val targetMeshRead = MeshIO.read(new File(targetMeshFn)).get.colorNormalMesh3D.get
    val target = ParametricRenderer.renderParameterMesh(init, targetMeshRead,RGBA.Black)
    val targetMesh: ColorNormalMesh3D = targetMeshRead.transform(init.modelViewTransform)
    PixelImageIO.write(target, new File(s"$outputDir/target.png")).get
    MeshIO.write(targetMesh, new File(s"$outputDir/target.ply")).get

    val targetMean = modelRenderer.renderImage(init)
    val targetMeshMean = modelRenderer.renderMesh(init)
    PixelImageIO.write(targetMean, new File(s"$outputDir/init.png")).get
    MeshIO.write(targetMeshMean, new File(s"$outputDir/init.ply")).get

    /* Prior Evaluator */
    val priorEval = ProductEvaluator(GaussianShapePrior(0, 1), GaussianTexturePrior(0, 1)) // prior for the model (shape/color)
    /* Shape Evaluator */
    val shapeEval = ShapeRenderEvaluator(modelRenderer, targetMesh, 0.1)

    // keep track of best sample
    val bestFileLogger = ParametersFileBestLogger(shapeEval, new File(s"$outputDir/fit-best.rps"))
    val bestSampleLogger = BestSampleLogger(shapeEval)
    val fitLogger = bestFileLogger :+ bestSampleLogger

    // Metropolis logger
    val printLogger = PrintLogger[RenderParameter](Console.out, "").verbose
    val mhLogger = printLogger

    //momo proposals
    val momoProposal = neutralMorphableModelProposal

    val proposal = if (albedoConsistency) {
      /* Foreground Evaluator */
      val pixEval = IsotropicGaussianPixelEvaluator(0.043f) // the parameter is the amount of pixel noise (standard deviation)
      /* Background Evaluator */
      val histBGEval = HistogramRGB.fromImageRGBA(target, 25)
      /* Pixel Evaluator */
      val imgEval = IndependentPixelEvaluator(pixEval, histBGEval) // combine pixEval and histBGEval
      /* Image Evaluator */
      val imgRenderEval = ImageRendererEvaluator(modelRenderer, imgEval.toDistributionEvaluator(target)) // combine imgEval and priorEval
      MetropolisFilterProposal(MetropolisFilterProposal(momoProposal, priorEval), imgRenderEval)
    } else MetropolisFilterProposal(momoProposal, priorEval)

    //image chain
    val imageFitter = MetropolisHastings(proposal, shapeEval)
    println("everything setup. starting fitter ...")

    val fitsamples = imageFitter.iterator(init, mhLogger).loggedWith(fitLogger).take(10000).toIndexedSeq
    val best = bestSampleLogger.currentBestSample().get
    val imgBest = modelRenderer.renderImage(best)
    PixelImageIO.write(imgBest, new File(s"$outputDir/fitter-best.png")).get
    RenderParameterIO.write(best, new File(s"$outputDir/fitter-best.rps")).get
    MeshIO.write(modelRenderer.renderMesh(best), new File(s"$outputDir/fitter-best.ply")).get
  }
}