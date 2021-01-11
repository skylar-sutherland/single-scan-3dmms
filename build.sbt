name := "single-scan-3dmms"

version       := "0.5"

scalaVersion  := "2.11.7"

scalacOptions := Seq("-unchecked", "-deprecation", "-encoding", "utf8")

resolvers += Resolver.bintrayRepo("unibas-gravis", "maven")

resolvers += Opts.resolver.sonatypeSnapshots

libraryDependencies  ++= Seq(
  "ch.unibas.cs.gravis" %% "scalismo-faces" % "0.10.1",
  "ch.unibas.cs.gravis" %% "scalismo" % "0.17.+",
  "ch.unibas.cs.gravis" % "scalismo-native-all" % "4.0.+",
  "ch.unibas.cs.gravis" %% "scalismo-ui" % "0.13.1"
)

mainClass in assembly := Some("gui.BuildAndView")

test in assembly := {}

assemblyJarName in assembly := "release.jar"

