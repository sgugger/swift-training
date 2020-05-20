// swift-tools-version:5.2

import PackageDescription

let package = Package(
  name: "SwiftTraining",
  platforms: [ .macOS(.v10_13) ],
  products: [
    .library(name: "TestUtils", targets: ["TestUtils"])
  ],
  dependencies: [],
  targets: [
    .target(name: "TestUtils", path: "TestUtils"),
    .testTarget(name: "TestUtilsTests", dependencies: ["TestUtils"]),
  ])
