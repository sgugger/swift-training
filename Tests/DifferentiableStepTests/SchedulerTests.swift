import XCTest
import TensorFlow
import Utils

@testable import DifferentiableStep

fileprivate let entropy = ARC4RandomNumberGenerator(seed: 42)
fileprivate let data = RegressionData(entropy: entropy)
fileprivate let model = RegressionModel(a: 0.0, b: 0.0)
fileprivate let optimizer = SGD(for: model, learningRate: 0.1)

final class SchedulerTests: XCTestCase {

  /// A callback that checks the optimizer learning rates. 
  class LearningRateObserver {
    /// The learning rates at each training batch.
    var learningRates: [Float] = []

    /// Inspect `trainingLoop` at `event` and can change its state accordingly.
    public func handler<L: TrainingLoopProtocol>(_ loop: inout L, event: TrainingLoopEvent) throws {
      if event == .batchEnd && Context.local.learningPhase == .training {
        learningRates.append(loop.optimizer.learningRate as! Float)
      }
    }
  }

  func testLinearSchedule() {
    let observer = LearningRateObserver()
    var trainingLoop = TrainingLoop(
      training: data.trainingEpochs, 
      validation: data.validationBatches, 
      model: model, 
      optimizer: optimizer,
      lossFunction: meanSquaredError,
      callbacks: [learningRateScheduler(makeSchedule(.linear, from: 0.1, to: 0.0)),
                  observer.handler])
    try! trainingLoop.fit(for: 3)

    let totalBatches = 30
    XCTAssertEqual(observer.learningRates.count, totalBatches, 
      "There were less learning rates observed than the total number of training batches.")

    let expected: [Float] = (0..<30).map { 0.1 - 0.1 * Float($0) / Float(totalBatches-1) } 
    let closeEnough: [Bool] = zip(observer.learningRates, expected).map { abs($0 - $1) < 1e-7 }
    XCTAssert(closeEnough.allSatisfy { $0 },
      "The learning rates did not follow the proper linear schedule.")
  }
}

extension SchedulerTests {
  static var allTests = [
    ("testLinearSchedule", testLinearSchedule),
  ]
}