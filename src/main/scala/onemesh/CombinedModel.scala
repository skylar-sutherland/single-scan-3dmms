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

import breeze.linalg.DenseMatrix
import breeze.stats.distributions.Uniform
import scalismo.color.{RGB, RGBA}
import scalismo.common.UnstructuredPointsDomain.Create.CreateUnstructuredPointsDomain3D
import scalismo.common._
import scalismo.faces.momo.PancakeDLRGP
import scalismo.geometry._
import scalismo.kernels.{DiagonalKernel, GaussianKernel, MatrixValuedPDKernel, PDKernel}
import scalismo.mesh.{SurfacePointProperty, TriangleMesh3D, VertexColorMesh3D}
import scalismo.numerics.{RandomMeshSampler3D, Sampler}
import scalismo.statisticalmodel._
import scalismo.utils.Random

/* This file contains the code for our paper's core novel contribution.  The following summarizes its contents:
* Hyperparameters                     An object containing the hyperparameters and simple associated code for our
*                                     Gaussian process kernels and our color-correlation and symmetry heuristics.
* buildColorDLRGP                     A function for converting discrete low-rank Gaussian processes defined over 3D
*                                     vectors into discrete low-rank Gaussian process albedo models.
* MeshColorSampler3D                  A helper class for defining samplers of the colors of mesh vertices.
* IsotropicDiagonalKernelCoupled      A class for building a matrix-valued positive-definite kernel from a scalar-valued
*                                     positive-definite kernel that (when used in a Gaussian process) correlates the
*                                     different dimensions of the kernel's outputs.
* RGBStandardGaussian Process         The standard color space-based Gaussian process albedo model.
* XYZStandardGaussian Process         The standard physical space-based Gaussian process albedo model.
* RGBCorrelatedGaussian Process       The color channel-correlated color space-based Gaussian process albedo model.
* XYZCorrelatedGaussian Process       The color channel-correlated physical space-based Gaussian process albedo model.
* generateRGBSampler                  Creates a sampler for the albedos of the vertices of a mesh.
* generateXYZSampler                  Creates a sampler for the positions of the vertices of a mesh.
* shapeGaussianProcess                The standard Gaussian process shape model.
* ColorSymmetricMatrixValuedKernel    A class for adding bilateral symmetry to Gaussian process albedo models.
* ShapeSymmetricMatrixValuedKernel    A class for adding bilateral symmetry to Gaussian process shape models.
* XYZSymmetricGaussianProcess         The symmetric physical space-based Gaussian process albedo model.
* symmetricShapeGaussianProcess       The symmetric Gaussian process shape model.
* generateColorPancake                Convert a continuous full-rank Gaussian process albedo model to a discrete
*                                     low-rank Gaussian process model.
* generateShapePancake                Convert a continuous full-rank Gaussian process shape model to a discrete low-rank
*                                     Gaussian process model.
* pancakeToVCM                        Generate mesh samples directly from shape and albedo discrete low-rank Gaussian
*                                     process models. */

object CombinedModel {
  type UPD3D = UnstructuredPointsDomain[_3D]

  object Hyperparameters {
    /* The scalar versions of the albedo deformation models for RGB and XYZ space, respectively.  These are defined as
    scalar-valued positive-semidefinite kernels defined over 3D space.  Change these to tune the albedo deformation
    models' parameters. */
    def scalarRGBKernel(): PDKernel[_3D] = GaussianKernel[_3D](0.15) * 0.015
    def scalarXYZKernel(): PDKernel[_3D] = {
      (GaussianKernel[_3D](500) * 0.02) +
        (GaussianKernel[_3D](20) * 0.01) +
        (GaussianKernel[_3D](2) * 0.01)
    }

    /* The shape deformation model's scalar kernel.  This is defined as a Gaussian process over 3D space. */
    def shapeScalarValuedKernel(): PDKernel[_3D] = {
      (GaussianKernel[_3D](100) * 7) +
        (GaussianKernel[_3D](50) * 5) +
        (GaussianKernel[_3D](10) * 3)
    }

    /* The denominators for color-correlated RGB-space and XYZ-space albedo deformation model; change this to change
    the degree of color-channel correlation.  A higher value makes the different color channels will be more strongly
    correlated.  These must be positive. */
    val RGBDenominator: Double = 20.0
    val XYZDenominator: Double = 16.0

    /* The degree of symmetry; change this to increase or decrease the correlation between opposite sides of the face.
    Must range from 0 to 0.5.  A higher value makes opposite sides of the face more strongly correlated. */
    val symmetryStrength: Double = 0.35

    /* How many points to sample from when discretizing the Gaussian processes.  This is a technical parameter and
    should generally not be modified.  */
    val samplerNumberOfPoints: Int = 300
  }

  /** Converts a deformation model (DLRGP for EuclideanVector[_3D]) to a point distribution model (DLRGP for Point[_3D]).
    * @param model DLRGP EuclideanVector[_3D] model
    * @param reference Reference used to map the deformation model to a point model.
    * @return DLRGP Point[_3D] model */
  def buildColorDLRGP(model: DiscreteLowRankGaussianProcess[_3D, UPD3D, EuclideanVector[_3D]],
                      reference: VertexColorMesh3D): DiscreteLowRankGaussianProcess[_3D, UPD3D, RGB] = {
    def vectorFToColorF(pf: DiscreteField[_3D, UPD3D, EuclideanVector[_3D]],
                        f: (EuclideanVector[_3D], PointId) => RGB): DiscreteField[_3D, UPD3D, RGB] = {
      new DiscreteField[_3D, UPD3D, RGB](pf.domain, (for (p <- pf.valuesWithIds) yield f(p._1, p._2)).toIndexedSeq)
    }

    val newKLBasis = for (b <- model.klBasis) yield {
      DiscreteLowRankGaussianProcess.Eigenpair[_3D, UPD3D, RGB](
        b.eigenvalue, vectorFToColorF(b.eigenfunction, (v: EuclideanVector[_3D], _) => RGB(v.x, v.y, v.z))
      )
    }
    val newMeanField = vectorFToColorF(model.mean, (v: EuclideanVector[_3D], i: PointId) => {
      val oldColor = reference.color.pointData.toIndexedSeq(i.id)
      RGB(oldColor.r + v.x, oldColor.g + v.y, oldColor.b + v.z)
    })

    DiscreteLowRankGaussianProcess[_3D, UPD3D, RGB](newMeanField, newKLBasis)
  }

  /* Samples points given an IndexedSeq listing them. */
  case class MeshColorSampler3D(colors: IndexedSeq[Point[_3D]], numberOfPoints: Int)(implicit rand: Random) extends Sampler[_3D] {
    // Colors are in [0, 1]^3, a set of measure 1, so I think this is correct:
    val volumeOfSampleRegion: Double = 1.0
    def sample(): IndexedSeq[(Point[_3D], Double)] = {
      val distrDim1 = Uniform(0, colors.length)(rand.breezeRandBasis)
      for (_ <- 0 until numberOfPoints) yield (colors(distrDim1.draw().toInt), 1.0)
    }
  }

  /* Builds a matrix-valued kernel that is a linear combination of a diagonal kernel and the all-ones matrix. */
  case class IsotropicDiagonalKernelCoupled[D <: Dim: NDSpace](kernel: PDKernel[D], denom: Double, outputDim: Int) extends DiagonalKernel[D] {
    val scaleMatrix: DenseMatrix[Double] = (DenseMatrix.eye[Double](outputDim) + (denom - 1.0))/denom
    def k(x: Point[D], y: Point[D]): DenseMatrix[Double] = scaleMatrix * kernel(x, y)
    override def domain: Domain[D] = kernel.domain
  }

  /* The color deformation models for RGB and XYZ space, respectively. */
  def RGBStandardGaussianProcess(): GaussianProcess[_3D, EuclideanVector[_3D]] = {
    GaussianProcess[_3D, EuclideanVector[_3D]](DiagonalKernel[_3D](Hyperparameters.scalarRGBKernel(), 3))
  }
  def XYZStandardGaussianProcess(): GaussianProcess[_3D, EuclideanVector[_3D]] = {
    GaussianProcess[_3D, EuclideanVector[_3D]](DiagonalKernel[_3D](Hyperparameters.scalarXYZKernel(), 3))
  }
  def RGBCorrelatedGaussianProcess(): GaussianProcess[_3D, EuclideanVector[_3D]] = {
    val matrixValuedKernel = IsotropicDiagonalKernelCoupled[_3D](Hyperparameters.scalarRGBKernel(), Hyperparameters.RGBDenominator, 3)
    GaussianProcess[_3D, EuclideanVector[_3D]](matrixValuedKernel)
  }
  def XYZCorrelatedGaussianProcess(): GaussianProcess[_3D, EuclideanVector[_3D]] = {
    val matrixValuedKernel = IsotropicDiagonalKernelCoupled[_3D](Hyperparameters.scalarXYZKernel(), Hyperparameters.XYZDenominator, 3)
    GaussianProcess[_3D, EuclideanVector[_3D]](matrixValuedKernel)
  }

  /* The positions samplers for RGB and XYZ space, respectively---returning points' locations in color and physical
  space.  The XYZ sampler is also used for shape deformation. */
  def generateRGBSampler(VCM: VertexColorMesh3D)(implicit rand: Random): (MeshColorSampler3D, UPD3D) = {
    val colorIndexedSeq = for (c <- VCM.color.pointData) yield Point3D(c.r, c.g, c.b)
    val sampler = MeshColorSampler3D(colorIndexedSeq, Hyperparameters.samplerNumberOfPoints)
    val colorSet = CreateUnstructuredPointsDomain3D.create(colorIndexedSeq)
    (sampler, colorSet)
  }
  def generateXYZSampler(VCM: VertexColorMesh3D)(implicit rand: Random): (RandomMeshSampler3D, UPD3D) = {
    val shape = VCM.shape
    (RandomMeshSampler3D(shape, Hyperparameters.samplerNumberOfPoints, seed = 42), shape.pointSet)
  }

  // The standard (non-symmetric) shape Gaussian process.
  def shapeGaussianProcess(): GaussianProcess[_3D, EuclideanVector[_3D]] = {
    val shapeMatrixValuedKernel = DiagonalKernel[_3D](Hyperparameters.shapeScalarValuedKernel(), 3)
    GaussianProcess[_3D, EuclideanVector[_3D]](shapeMatrixValuedKernel)
  }

  /* Versions of the Gaussian process models with partial bilateral symmetry. */
  case class ColorSymmetricMatrixValuedKernel(baseKernel: MatrixValuedPDKernel[_3D]) extends MatrixValuedPDKernel[_3D] {
    def outputDim: Int = baseKernel.outputDim
    def domain: Domain[_3D] = baseKernel.domain
    def k(x: Point[_3D], y: Point[_3D]): DenseMatrix[Double] = {
      val xp = Point(-x(0), x(1), x(2))
      val yp = Point(-y(0), y(1), y(2))
      ((baseKernel(x, y) + baseKernel(xp, yp)) * 0.5) + ((baseKernel(xp, y) + baseKernel(x, yp)) * Hyperparameters.symmetryStrength)
    }
  }
  case class ShapeSymmetricMatrixValuedKernel(baseKernel: MatrixValuedPDKernel[_3D]) extends MatrixValuedPDKernel[_3D] {
    def outputDim: Int = baseKernel.outputDim
    def domain: Domain[_3D] = baseKernel.domain
    def k(x: Point[_3D], y: Point[_3D]): DenseMatrix[Double] = {
      val xp = Point(-x(0), x(1), x(2))
      val yp = Point(-y(0), y(1), y(2))
      val Ip = DenseMatrix.eye[Double](outputDim); Ip(0, 0) = -1
      ((baseKernel(x, y) + baseKernel(xp, yp)) * 0.5) + (Ip * (baseKernel(x, yp) + baseKernel(xp, y)) * Hyperparameters.symmetryStrength)
    }
  }
  def XYZSymmetricGaussianProcess(): GaussianProcess[_3D, EuclideanVector[_3D]] = {
    GaussianProcess[_3D, EuclideanVector[_3D]](ColorSymmetricMatrixValuedKernel(XYZCorrelatedGaussianProcess().cov))
  }
  def symmetricShapeGaussianProcess(): GaussianProcess[_3D, EuclideanVector[_3D]] = {
    GaussianProcess[_3D, EuclideanVector[_3D]](ShapeSymmetricMatrixValuedKernel(shapeGaussianProcess().cov))
  }

  /* Build color and shape pancake DLRGPs, from which we can build a 3DMM. */
  def generateColorPancake(VCM: VertexColorMesh3D, gpColor: GaussianProcess[_3D, EuclideanVector[_3D]],
                           sampler: Sampler[_3D], locationSet: UPD3D, numComponents: Int)
                          (implicit rand: Random): PancakeDLRGP[_3D, UPD3D, RGB] = {
    val lowRankGPColor = LowRankGaussianProcess.approximateGPNystrom(gpColor, sampler, numBasisFunctions = numComponents)
    println("lowRankGPColor done.")
    val dlgpColor = DiscreteLowRankGaussianProcess(locationSet, lowRankGPColor)
    println("dlgpColor done.")
    val colorNoiseVariance = 0.0
    val colorPancake = PancakeDLRGP(buildColorDLRGP(dlgpColor, VCM), colorNoiseVariance)
    println("colorPancake done.")
    colorPancake
  }
  def generateShapePancake(VCM: VertexColorMesh3D, gpShape: GaussianProcess[_3D, EuclideanVector[_3D]],
                           sampler: Sampler[_3D], positionSet: UPD3D, numComponents: Int)
                          (implicit rand: Random): PancakeDLRGP[_3D, UPD3D, Point[_3D]] = {
    val lowRankGPShape = LowRankGaussianProcess.approximateGPNystrom(gpShape, sampler, numBasisFunctions = numComponents)
    println("lowRankGPShape done.")
    val dlgpShape = DiscreteLowRankGaussianProcess(positionSet, lowRankGPShape)
    println("dlgpShape done.")
    val shapeNoiseVariance = 0.0
    val shapePancake = PancakeDLRGP(ModelHelpers.vectorToPointDLRGP(dlgpShape, VCM.shape), shapeNoiseVariance)
    println("shapePancake done.")
    shapePancake
  }

  /* Build a random sample from a color pancake and reference mesh, represented as a VCM. */
  def pancakeToVCM(colorPancake: PancakeDLRGP[_3D, UPD3D, RGB],
                   shapeStatic: TriangleMesh3D)(implicit rand: Random): VertexColorMesh3D = {
    val colorComponent = for (x <- colorPancake.gpModel.sample().data.toIndexedSeq) yield x.toRGBA
    VertexColorMesh3D(shapeStatic, SurfacePointProperty[RGBA](shapeStatic.triangulation, colorComponent))
  }
}