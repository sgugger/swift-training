import XCTest
import TensorFlow
import Utils

@testable import DifferentiableStep

fileprivate let entropy = ARC4RandomNumberGenerator(seed: 42)
fileprivate let data = RegressionData(entropy: entropy)
fileprivate let model = RegressionModel(a: 0.0, b: 0.0)
fileprivate let optimizer = SGD(for: model, learningRate: 0.1)

final class RecorderTests: XCTestCase {
  
  /// A callback that stores the losses  
  class LossStorer {
    /// The training losses of each batch.
    var trainingLosses: [Tensor<Float>] = []
    /// The validation losses of each batch.
    var validationLosses: [Tensor<Float>] = []
   
    /// Inspect `trainingLoop` at `event` and can change its state accordingly.
    public func handler<L: TrainingLoopProtocol>(_ loop: inout L, event: TrainingLoopEvent) throws {
      if event == .batchEnd {
        guard let loss = loop.lastLoss else { return }
        if Context.local.learningPhase == .training {
          trainingLosses.append(loss)
        } else {
          validationLosses.append(loss)
        } 
      }
    }
  }

  func testLossValues() {
    let recorder = Recorder()
    let lossStorer = LossStorer()
      
    var trainingLoop = TrainingLoop(
      training: data.trainingEpochs, 
      validation: data.validationBatches, 
      model: model, 
      optimizer: optimizer,
      lossFunction: meanSquaredError,
      callbacks: [recorder.handler, lossStorer.handler])
    try! trainingLoop.fit(for: 1)

    XCTAssertEqual(recorder.trainingLosses.count, 1, 
      "There should be exactly one training loss after fitting for one epoch.")
    let trainingLoss = recorder.trainingLosses.last!
  
    XCTAssertEqual(recorder.validationLosses.count, 1, 
      "There should be exactly one validation loss after fitting for one epoch.")
    let validationLoss = recorder.validationLosses.last!
      
    let observedTrainingLoss = Tensor(concatenating: lossStorer.trainingLosses).mean()
    XCTAssert(abs(observedTrainingLoss.scalarized() - trainingLoss) <= 1e-7, 
      "The training loss computed by the Recorder is wrong.")
      
    let observedValidationLoss = Tensor(concatenating: lossStorer.validationLosses).mean()
    XCTAssert(abs(observedValidationLoss.scalarized() - validationLoss) <= 1e-7, 
      "The validation loss computed by the Recorder is wrong.")
  }
  
  struct AlmostAccuracy: Metric {
    public var name: String { "AlmostAccuracy(\(threshold))" }
    let threshold: Float
    var sampleCount: Int = 0
    var corrects: Int = 0

    public var value: Float? {
      sampleCount == 0 ? nil : Float(corrects) / Float(sampleCount)
    }

    public init(threshold: Float) {
      self.threshold = threshold
    }

    public mutating func reset() {
      sampleCount = 0
      corrects = 0
    }

    public mutating func accumulate<Output, Target>(output: Output, target: Target) {
      guard let tensorOutput = output as? Tensor<Float> else { return }
      guard let tensorTarget = target as? Tensor<Float> else { return }
      sampleCount += tensorOutput.shape[0]
      let diff = abs(tensorOutput - tensorTarget)
      corrects += Int(Tensor<Int32>(diff .<= threshold).sum().scalarized())
    }
  }
    
  func testMetricValues() {
    let thresholds: [Float] = [0.1, 0.01, 0.001]
    let metrics = thresholds.map { AlmostAccuracy(threshold: $0) }
    let recorder = Recorder(metrics: metrics)
    var trainingLoop = TrainingLoop(
      training: data.trainingEpochs, 
      validation: data.validationBatches, 
      model: model, 
      optimizer: optimizer,
      lossFunction: meanSquaredError,
      callbacks: [recorder.handler])
    try! trainingLoop.fit(for: 2)

    XCTAssertEqual(recorder.metricResults.count, 2, 
      "There should be exactly two arrays of metrics after fitting for two epochs.")
    
    for i in (0..<2) {
      XCTAssertEqual(recorder.metricResults[i].count, thresholds.count, 
        "There should be exactly \(thresholds.count) values in each array of metrics.")
    }
    let observedMetrics = recorder.metricResults.last!
    
    let predictions = Tensor<Float>(concatenating: data.validationBatches.map { 
      trainingLoop.model($0.data)
    })
    let targets = Tensor<Float>(concatenating: data.validationBatches.map(\.label))
    let diff = abs(predictions - targets)
    for (observedMetric, threshold) in zip(observedMetrics, thresholds) {
      let corrects = Int(Tensor<Int32>(diff .<= threshold).sum().scalarized())
      let theoreticalMetric = Float(corrects) / Float(targets.shape[0])
      XCTAssertEqual(observedMetric, theoreticalMetric) 
    }
  }
}

extension RecorderTests {
  static var allTests = [
    ("testLossValues", testLossValues),
    ("testMetricValues", testMetricValues),
  ]
}
