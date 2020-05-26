// swift-tools-version:5.2

import PackageDescription

let package = Package(
  name: "SwiftTraining",
  platforms: [ .macOS(.v10_13) ],
  products: [
    .library(name: "Utils", targets: ["Utils"]),
    .library(name: "TrainingStep", targets: ["TrainingStep"]),
    .library(name: "DifferentiableStep", targets: ["DifferentiableStep"])
  ],
  dependencies: [],
  targets: [
    .target(name: "Utils", path: "Utils"),
    .testTarget(name: "UtilsTests", dependencies: ["Utils"]),
    .target(name: "TrainingStep", dependencies: ["Utils"], path: "TrainingStep"),
    .testTarget(name: "TrainingStepTests", dependencies: ["Utils", "TrainingStep"]),
    .target(name: "DifferentiableStep", dependencies: ["Utils"], path: "DifferentiableStep"),
    //.testTarget(name: "DifferentiableTests", dependencies: ["Utils", "DifferentiableStep"]),
  ])
