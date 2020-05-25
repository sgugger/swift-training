import XCTest
import TensorFlow
import Utils

@testable import TrainingStep

fileprivate let entropy = ARC4RandomNumberGenerator(seed: 42)
fileprivate let data = RegressionData(entropy: entropy)
fileprivate let model = RegressionModel(a: 0.0, b: 0.0)
fileprivate let optimizer = SGD(for: model, learningRate: 0.1)

final class SchedulerTests: XCTestCase {

  /// A callback that checks the optimizer learning rates. 
  struct LearningRateObserver: TrainingLoopCallback {
    /// The learning rates at each training batch.
    var learningRates: [Float] = []

    /// Inspect `trainingLoop` at `event` and can change its state accordingly.
    public mutating func call<T: TrainingLoopProtocol>(
      on trainingLoop: T, event: TrainingLoopEvent
    ) throws {
      if event == .batchEnd && Context.local.learningPhase == .training {
        learningRates.append(trainingLoop.optimizer.learningRate as! Float)
      }
    }
  }

  func testLinearSchedule() {
    var trainingLoop = TrainingLoop(
      training: data.trainingEpochs, 
      validation: data.validationBatches, 
      model: model, 
      optimizer: optimizer,
      lossFunction: meanSquaredError,
      callbacks: [Scheduler(schedule: makeSchedule(.linear, from: 0.1, to: 0.0)),
                  LearningRateObserver()])
    try! trainingLoop.fit(for: 3)

    let totalBatches = 30
    let scheduler = trainingLoop.callbacks[0] as! Scheduler
    XCTAssertEqual(scheduler.learningRates.count, totalBatches, 
      "There were less learning rates scheduled than the total number of training batches.")

    let observer = trainingLoop.callbacks[1] as! LearningRateObserver
    XCTAssertEqual(scheduler.learningRates, observer.learningRates,
      "The learning rates were not properly set in the optimizer.")

    let expected: [Float] = (0..<30).map { 0.1 - 0.1 * Float($0) / Float(totalBatches-1) } 
    let closeEnough: [Bool] = zip(scheduler.learningRates, expected).map { abs($0 - $1) < 1e-7 }
    XCTAssert(closeEnough.allSatisfy { $0 },
      "The learning rates did not follow the proper linear schedule.")
  }
}

extension SchedulerTests {
  static var allTests = [
    ("testLinearSchedule", testLinearSchedule),
  ]
}