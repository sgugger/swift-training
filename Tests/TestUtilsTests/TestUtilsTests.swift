import XCTest
import TensorFlow

@testable import TestUtils

var entropy = ARC4RandomNumberGenerator(seed: 42)

final class TestUtilsTests: XCTestCase {

  let dataset = RegressionData(entropy: ARC4RandomNumberGenerator(seed: 42))
  let model = RegressionModel(a: 2.0, b: 3.0)
  
  let batchSize: Int = 64
  let trainingCount: Int = 10
  let validationCount: Int = 5

  func testTrainingShapes() {
    for batches in dataset.trainingEpochs.prefix(1) {
      XCTAssertEqual(batches.count, trainingCount, "Incorrect number of batches.")
      for batch in batches {
        let (data, label) = (batch.data, batch.label)
        XCTAssertEqual(data.shape, TensorShape([64]),
          "Wrong shape for batch data: \(data.shape), should be [64]")
        XCTAssertEqual(label.shape, TensorShape([64]),
          "Wrong shape for batch data: \(label.shape), should be [64]")
      }
    }
  }
    
  func testValidationShapes() {
    let batches = dataset.validationBatches
    XCTAssertEqual(batches.count, validationCount, "Incorrect number of batches.")
    for batch in batches {
      let (data, label) = (batch.data, batch.label)
      XCTAssertEqual(data.shape, TensorShape([64]),
        "Wrong shape for batch data: \(data.shape), should be [64]")
      XCTAssertEqual(label.shape, TensorShape([64]),
        "Wrong shape for batch data: \(label.shape), should be [64]")
    }
  }

  func testModelInTraining() {
    for batches in dataset.trainingEpochs.prefix(1) {
      for batch in batches {
        let (data, label) = (batch.data, batch.label)
        let output = model(data)
        XCTAssertEqual(output.shape, TensorShape([64]),
          "Wrong shape for the output: \(output.shape), should be [64]")
        let loss = meanAbsoluteError(predicted: output, expected: label)
        XCTAssert(loss.scalarized() <= 0.1,
          "The loss is too big for the right model!")
      }
    }
  }
  
  func testModelInValidation() {
    let batches = dataset.validationBatches
    for batch in batches {
      let (data, label) = (batch.data, batch.label)
      let output = model(data)
      XCTAssertEqual(output.shape, TensorShape([64]),
        "Wrong shape for the output: \(output.shape), should be [64]")
      let loss = meanAbsoluteError(predicted: output, expected: label)
      XCTAssert(loss.scalarized() <= 0.1,
        "The loss is too big for the right model!")
    }
  }
}

extension TestUtilsTests {
  static var allTests = [
    ("testTrainingShapes", testTrainingShapes),
    ("testValidationShapes", testValidationShapes),
    ("testModelInTraining", testModelInTraining),
    ("testModelInValidation", testModelInValidation)
  ]
}
