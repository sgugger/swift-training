// swift-tools-version:5.2

import PackageDescription

let package = Package(
  name: "SwiftTraining",
  platforms: [ .macOS(.v10_13) ],
  products: [
    .library(name: "Utils", targets: ["Utils"]),
    .library(name: "TrainingStep", targets: ["TrainingStep"])
  ],
  dependencies: [],
  targets: [
    .target(name: "Utils", path: "Utils"),
    .testTarget(name: "UtilsTests", dependencies: ["Utils"]),
    .target(name: "TrainingStep", dependencies: ["Utils"], path: "TrainingStep"),
  ])
