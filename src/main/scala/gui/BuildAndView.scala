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

package gui

import java.awt.Dimension
import java.io.{File, IOException}

import javax.swing._
import javax.swing.event.{ChangeEvent, ChangeListener}
import breeze.linalg.min
import onemesh.CombinedModelHelpers
import scalismo.color.{RGB, RGBA}
import scalismo.faces.gui.{GUIBlock, GUIFrame, ImagePanel}
import scalismo.faces.gui.GUIBlock._
import scalismo.faces.parameters.RenderParameter
import scalismo.faces.io.{MeshIO, MoMoIO, PixelImageIO, RenderParameterIO}
import scalismo.faces.sampling.face.MoMoRenderer
import scalismo.faces.image.PixelImage
import scalismo.faces.mesh.OptionalColorNormalMesh3D
import scalismo.utils.Random
import scalismo.faces.momo.MoMo
import scalismo.mesh.VertexColorMesh3D

import scala.reflect.io.Path
import scala.util.{Failure, Try}

/* Adapted from the Simple Morphable Model Viewer
https://github.com/unibas-gravis/basel-face-model-viewer/blob/master/src/main/scala/faces/apps/ModelViewer.scala
That project was released via the Apache License 2.0 by the University of Basel, and uses the same license file as was
provided with this project. */

object BuildAndView extends App {
  final val DEFAULT_DIR = new File(".")

  val modelFile: Option[File] = getModelFile(args)
  modelFile.map(SimpleModelBuilderAndViewer(_))

  private def getModelFile(args: Seq[String]): Option[File] = {
    if (args.nonEmpty) {
      val path = Path(args.head)
      if (path.isFile) return Some(path.jfile)
      if (path.isDirectory) return askUserForModelFile(path.jfile)
    }
    askUserForModelFile(DEFAULT_DIR)
  }

  private def askUserForModelFile(dir: File): Option[File] = {
    val jFileChooser = new JFileChooser(dir)
    if (jFileChooser.showOpenDialog(null) == JFileChooser.APPROVE_OPTION) {
      Some(jFileChooser.getSelectedFile)
    } else {
      println("No ply model select...")
      None
    }
  }
}

case class SimpleModelBuilderAndViewer (modelFile: File, imageWidth: Int = 512, imageHeight: Int = 512,
                                        maximalSliderValue: Int = 2, maximalShapeRank: Option[Int] = None,
                                        maximalColorRank: Option[Int] = None) {
  scalismo.initialize()
  val seed = 1024L
  implicit val rnd: Random = Random(seed)

  val template: OptionalColorNormalMesh3D = MeshIO.read(modelFile).get
  val templateVCM: VertexColorMesh3D = template.vertexColorMesh3D.get
  println("building model (this may take a while)")
  val model: MoMo = CombinedModelHelpers.createStandardModel(templateVCM)
  println("model succesfully built")
  val shapeRank: Int = maximalShapeRank match {
    case Some(rank) => min(model.neutralModel.shape.rank, rank)
    case _ => model.neutralModel.shape.rank
  }

  val colorRank: Int = maximalColorRank match {
    case Some(rank) => min(model.neutralModel.color.rank, rank)
    case _ => model.neutralModel.color.rank
  }

  var renderer: MoMoRenderer = MoMoRenderer(model, RGBA.BlackTransparent).cached(5)

  val initDefault: RenderParameter = RenderParameter.defaultSquare.fitToImageSize(imageWidth, imageHeight)
  val init10: RenderParameter = initDefault.copy(
    momo = initDefault.momo.withNumberOfCoefficients(shapeRank, colorRank, 0)
  )
  var init: RenderParameter = init10

  var changingSliders = false

  val sliderSteps = 1000
  var maximalSigma: Int = maximalSliderValue
  var maximalSigmaSpinner: JSpinner = {
    val spinner = new JSpinner(new SpinnerNumberModel(maximalSigma,0,999,1))
    spinner.addChangeListener( new ChangeListener() {
      override def stateChanged(e: ChangeEvent): Unit = {
        val newMaxSigma = spinner.getModel.asInstanceOf[SpinnerNumberModel].getNumber.intValue()
        maximalSigma = math.abs(newMaxSigma)
        setShapeSliders()
        setColorSliders()
      }
    })
    spinner.setToolTipText("maximal slider value")
    spinner
  }

  def sliderToParam(value: Int): Double = {
    maximalSigma * value.toDouble/sliderSteps
  }

  def paramToSlider(value: Double): Int = {
    (value / maximalSigma * sliderSteps).toInt
  }

  val bg: PixelImage[RGBA] = PixelImage(imageWidth, imageHeight, (_, _) => RGBA.Black)

  val imageWindow: ImagePanel[RGB] = ImagePanel(renderWithBG(init))

  //--- SHAPE -----
  val shapeSlider: IndexedSeq[JSlider] = for (n <- 0 until shapeRank) yield {
    GUIBlock.slider(-sliderSteps, sliderSteps, 0, f => {
      updateShape(n, f)
      updateImage()
    })
  }

  val shapeSliderView: JPanel = GUIBlock.shelf(shapeSlider.zipWithIndex.map(s => GUIBlock.stack(s._1, new JLabel("" + s._2))): _*)
  val shapeScrollPane = new JScrollPane(shapeSliderView)
  val shapeScrollBar: JScrollBar = shapeScrollPane.createVerticalScrollBar()
  shapeScrollPane.setSize(800, 300)
  shapeScrollPane.setPreferredSize(new Dimension(800, 300))

  val rndShapeButton: JButton = GUIBlock.button("random", {
    randomShape(); updateImage()
  })
  val resetShapeButton: JButton = GUIBlock.button("reset", {
    resetShape(); updateImage()
  })
  rndShapeButton.setToolTipText("draw each shape parameter at random from a standard normal distribution")
  resetShapeButton.setToolTipText("set all shape parameters to zero")

  def updateShape(n: Int, value: Int): Unit = {
    init = init.copy(momo = init.momo.copy(shape = {
      for ((v, i) <- init.momo.shape.zipWithIndex) yield if (i == n) sliderToParam(value) else v
    }))
  }

  def randomShape(): Unit = {
    init = init.copy(momo = init.momo.copy(shape = {
      for (_ <- init.momo.shape) yield rnd.scalaRandom.nextGaussian
    }))
    setShapeSliders()
  }

  def resetShape(): Unit = {
    init = init.copy(momo = init.momo.copy(
      shape = IndexedSeq.fill(shapeRank)(0.0)
    ))
    setShapeSliders()
  }

  def setShapeSliders(): Unit = {
    changingSliders = true
    for (i <- 0 until shapeRank) shapeSlider(i).setValue(paramToSlider(init.momo.shape(i)))
    changingSliders = false
  }

  //--- COLOR -----
  val colorSlider: IndexedSeq[JSlider] = for (n <- 0 until colorRank) yield {
    GUIBlock.slider(-sliderSteps, sliderSteps, 0, f => {
      updateColor(n, f)
      updateImage()
    })
  }

  val colorSliderView: JPanel = GUIBlock.shelf(colorSlider.zipWithIndex.map(s => GUIBlock.stack(s._1, new JLabel("" + s._2))): _*)
  val colorScrollPane = new JScrollPane(colorSliderView)
  val colorScrollBar: JScrollBar = colorScrollPane.createHorizontalScrollBar()
  colorScrollPane.setSize(800, 300)
  colorScrollPane.setPreferredSize(new Dimension(800, 300))

  val rndColorButton: JButton = GUIBlock.button("random", {
    randomColor(); updateImage()
  })

  val resetColorButton: JButton = GUIBlock.button("reset", {
    resetColor(); updateImage()
  })
  rndColorButton.setToolTipText("draw each color parameter at random from a standard normal distribution")
  resetColorButton.setToolTipText("set all color parameters to zero")

  def updateColor(n: Int, value: Int): Unit = {
    init = init.copy(momo = init.momo.copy(color = {
      for ((v, i) <- init.momo.color.zipWithIndex) yield if (i == n) sliderToParam(value) else v
    }))
  }

  def randomColor(): Unit = {
    init = init.copy(momo = init.momo.copy(color = {
      for (_ <- init.momo.color) yield rnd.scalaRandom.nextGaussian
    }))
    setColorSliders()
  }

  def resetColor(): Unit = {
    init = init.copy(momo = init.momo.copy(
      color = IndexedSeq.fill(colorRank)(0.0)
    ))
    setColorSliders()
  }

  def setColorSliders(): Unit = {
    changingSliders = true
    for (i <- 0 until colorRank) colorSlider(i).setValue(paramToSlider(init.momo.color(i)))
    changingSliders = false
  }
  //--- ALL TOGETHER -----
  val randomButton: JButton = GUIBlock.button("random", {
    randomShape(); randomColor(); updateImage()
  })
  val resetButton: JButton = GUIBlock.button("reset", {
    resetShape(); resetColor(); updateImage()
  })

  randomButton.setToolTipText("draw each model parameter at random from a standard normal distribution")
  resetButton.setToolTipText("set all model parameters to zero")

  //function to export model as a hdf5 file
  def exportModel (): Try[Unit] ={

    def askToOverwrite(file: File): Boolean = {
      val dialogButton = JOptionPane.YES_NO_OPTION
      JOptionPane.showConfirmDialog(null, s"Would you like to overwrite the existing file: $file?","Warning",dialogButton) == JOptionPane.YES_OPTION
    }

    val fc = new JFileChooser()
    fc.setFileSelectionMode(JFileChooser.FILES_AND_DIRECTORIES)
    fc.setDialogTitle("Select a folder to store the .h5 file and name it")
    if (fc.showSaveDialog(null) == JFileChooser.APPROVE_OPTION) {
      var file = fc.getSelectedFile
      if (file.isDirectory) file = new File(file,"model.h5")
      if ( !file.getName.endsWith(".ply")) file = new File( file+".h5")
      if (!file.exists() || askToOverwrite(file)) {
        MoMoIO.write(model, file)
      } else {
        Failure(new IOException(s"Something went wrong when writing to file the file $file."))
      }
    } else {
      Failure(new Exception("User aborted save dialog."))
    }
  }

  //function to export the current shown face as a .ply file
  def exportShape (): Try[Unit] = {

    def askToOverwrite(file: File): Boolean = {
      val dialogButton = JOptionPane.YES_NO_OPTION
      JOptionPane.showConfirmDialog(null, s"Would you like to overwrite the existing file: $file?","Warning",dialogButton) == JOptionPane.YES_OPTION
    }

    val VCM3D = model.neutralModel.instance(init.momo.coefficients)


    val fc = new JFileChooser()
    fc.setFileSelectionMode(JFileChooser.FILES_AND_DIRECTORIES)
    fc.setDialogTitle("Select a folder to store the .ply file and name it")
    if (fc.showSaveDialog(null) == JFileChooser.APPROVE_OPTION) {
      var file = fc.getSelectedFile
      if (file.isDirectory) file = new File(file,"instance.ply")
      if ( !file.getName.endsWith(".ply")) file = new File( file+".ply")
      if (!file.exists() || askToOverwrite(file)) {
        MeshIO.write(VCM3D, file)
      } else {
        Failure(new IOException(s"Something went wrong when writing to file the file $file."))
      }
    } else {
      Failure(new Exception("User aborted save dialog."))
    }
  }

  //function to export the current shown face as a .ply file
  def exportImage (): Try[Unit] ={

    def askToOverwrite(file: File): Boolean = {
      val dialogButton = JOptionPane.YES_NO_OPTION
      JOptionPane.showConfirmDialog(null, s"Would you like to overwrite the existing file: $file?","Warning",dialogButton) == JOptionPane.YES_OPTION
    }

    val img = renderer.renderImage(init)

    val fc = new JFileChooser()
    fc.setFileSelectionMode(JFileChooser.FILES_AND_DIRECTORIES)
    fc.setDialogTitle("Select a folder to store the .png file and name it")
    if (fc.showSaveDialog(null) == JFileChooser.APPROVE_OPTION) {
      var file = fc.getSelectedFile
      if (file.isDirectory) file = new File(file,"instance.png")
      if ( !file.getName.endsWith(".png")) file = new File( file+".png")
      if (!file.exists() || askToOverwrite(file)) {
        PixelImageIO.write(img, file)
      } else {
        Failure(new IOException(s"Something went wrong when writing to file the file $file."))
      }
    } else {
      Failure(new Exception("User aborted save dialog."))
    }
  }

  //exportModel button and its tooltip
  val exportModelButton: JButton = GUIBlock.button("export hdf5", {exportModel()})
  exportModelButton.setToolTipText("export the current model as .h5")

  //exportShape button and its tooltip
  val exportShapeButton: JButton = GUIBlock.button("export PLY", {exportShape()})
  exportShapeButton.setToolTipText("export the current shape and texture as .ply")

  //exportImage button and its tooltip
  val exportImageButton: JButton = GUIBlock.button("export PNG", {exportImage()})
  exportImageButton.setToolTipText("export the current image as .png")

  //loads parameters from file
  //TODO: load other parameters than the momo shape, expr and color

  def askUserForRPSFile(dir: File): Option[File] = {
    val jFileChooser = new JFileChooser(dir)
    if (jFileChooser.showOpenDialog(null) == JFileChooser.APPROVE_OPTION) {
      Some(jFileChooser.getSelectedFile)
    } else {
      println("No Parameters select...")
      None
    }
  }

  def resizeParameterSequence(params: IndexedSeq[Double], length: Int, fill: Double): IndexedSeq[Double] = {
    val zeros = IndexedSeq.fill[Double](length)(fill)
    (params ++ zeros).slice(0, length) //brute force
  }

  def updateModelParameters(params: RenderParameter): Unit = {
    val newShape = resizeParameterSequence(params.momo.shape, shapeRank, 0)
    val newColor = resizeParameterSequence(params.momo.color, colorRank, 0)
    println("Loaded Parameters")

    init = init.copy(momo = init.momo.copy(shape = newShape, color = newColor))
    setShapeSliders()
    setColorSliders()
    updateImage()
  }

  val loadButton: JButton = GUIBlock.button(
    "load RPS",
    {
      for {rpsFile <- askUserForRPSFile(new File("."))
           rpsParams <- RenderParameterIO.read(rpsFile)} {
        val maxSigma = (rpsParams.momo.shape ++ rpsParams.momo.color).map(math.abs).max
        if ( maxSigma > maximalSigma ) {
          maximalSigma = math.ceil(maxSigma).toInt
          maximalSigmaSpinner.setValue(maximalSigma)
          setShapeSliders()
          setColorSliders()
        }
        updateModelParameters(rpsParams)
      }
    }
  )

  //---- update the image
  def updateImage(): Unit = {
    if (!changingSliders) imageWindow.updateImage(renderWithBG(init))
  }

  def renderWithBG(init: RenderParameter): PixelImage[RGB] = {
    val fg = renderer.renderImage(init)
    fg.zip(bg).map { case (f, b) => b.toRGB.blend(f) }
    //    fg.map(_.toRGB)
  }

  //--- COMPOSE FRAME ------
  val controls = new JTabbedPane()
  controls.addTab("color", GUIBlock.stack(colorScrollPane, GUIBlock.shelf(rndColorButton, resetColorButton)))
  controls.addTab("shape", GUIBlock.stack(shapeScrollPane, GUIBlock.shelf(rndShapeButton, resetShapeButton)))


  val guiFrame: GUIFrame = GUIBlock.stack(
    GUIBlock.shelf(imageWindow,
      GUIBlock.stack(controls,
        GUIBlock.shelf(maximalSigmaSpinner, randomButton, resetButton, loadButton, exportModelButton, exportShapeButton, exportImageButton)
      )
    )
  ).displayIn("MoMo-Viewer")

  //--- ROTATION CONTROLS ------

  import java.awt.event._

  var lookAt = false
  imageWindow.requestFocusInWindow()

  imageWindow.addKeyListener(new KeyListener {
    override def keyTyped(e: KeyEvent): Unit = {}

    override def keyPressed(e: KeyEvent): Unit = {
      if (e.getKeyCode == KeyEvent.VK_CONTROL) lookAt = true
    }

    override def keyReleased(e: KeyEvent): Unit = {
      if (e.getKeyCode == KeyEvent.VK_CONTROL) lookAt = false
    }
  })

  imageWindow.addMouseListener(new MouseListener {
    override def mouseExited(e: MouseEvent): Unit = {}

    override def mouseClicked(e: MouseEvent): Unit = {
      imageWindow.requestFocusInWindow()
    }

    override def mouseEntered(e: MouseEvent): Unit = {}

    override def mousePressed(e: MouseEvent): Unit = {}

    override def mouseReleased(e: MouseEvent): Unit = {}
  })

  imageWindow.addMouseMotionListener(new MouseMotionListener {
    override def mouseMoved(e: MouseEvent): Unit = {
      if (lookAt) {
        val x = e.getX
        val y = e.getY
        val yawPose = math.Pi / 2 * (x - imageWidth * 0.5) / (imageWidth / 2)
        val pitchPose = math.Pi / 2 * (y - imageHeight * 0.5) / (imageHeight / 2)

        init = init.copy(pose = init.pose.copy(yaw = yawPose, pitch = pitchPose))
        updateImage()
      }
    }

    override def mouseDragged(e: MouseEvent): Unit = {}
  })
}
