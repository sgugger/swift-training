import XCTest
import TensorFlow
import Utils

@testable import DifferentiableStep

fileprivate let entropy = ARC4RandomNumberGenerator(seed: 42)
fileprivate let data = RegressionData(entropy: entropy)
fileprivate let model = RegressionModel(a: 0.0, b: 0.0)
fileprivate let optimizer = SGD(for: model, learningRate: 0.1)

// For control flow tests
let trainingBatchEvents: [TrainingLoopEvent] = [.batchStart, .updateStart, .batchEnd]
let validationBatchEvents: [TrainingLoopEvent] = [.batchStart, .inferencePredictionEnd, .batchEnd]
let trainingStepsEvents = [[TrainingLoopEvent]](repeating: trainingBatchEvents, count: 10)
let trainingPhaseEvents: [TrainingLoopEvent] = (
  [.trainingStart] + trainingStepsEvents.reduce(into: [], +=) + [.trainingEnd])
let validationStepsEvents = [[TrainingLoopEvent]](repeating: validationBatchEvents, count: 5)
let validationPhaseEvents: [TrainingLoopEvent] = (
  [.validationStart] + validationStepsEvents.reduce(into: [], +=) + [.validationEnd])
let epochEvents: [TrainingLoopEvent] = (
  [.epochStart] + trainingPhaseEvents + validationPhaseEvents + [.epochEnd])
let fitEvents: [TrainingLoopEvent] = (
  [.fitStart] + epochEvents + epochEvents + [.fitEnd])  

final class TrainingLoopTests: XCTestCase {
    
  func testModelLearnsSomething() {
    var trainingLoop = TrainingLoop(
      training: data.trainingEpochs, 
      validation: data.validationBatches, 
      model: model, 
      optimizer: optimizer,
      lossFunction: meanSquaredError)
    try! trainingLoop.fit(for: 3)
    // Different seed might need some adjustment to the tolerance
    XCTAssert(abs(trainingLoop.model.a.scalarized() - 2.0) <= 0.01, 
      "Model should have learned a better value for b.")
    XCTAssert(abs(trainingLoop.model.b.scalarized() - 3.0) <= 0.01, 
      "Model should have learned a better value for b.")
  }
  
  class TestAllEvents {
    var events: [TrainingLoopEvent] = []
  
    public func handler<L: TrainingLoopProtocol>(_ loop: inout L, event: TrainingLoopEvent) throws {
      events.append(event)
    }
  }
  
  func testAllEvents() {
    let testEvents = TestAllEvents()
    var trainingLoop = TrainingLoop(
      training: data.trainingEpochs, 
      validation: data.validationBatches, 
      model: model, 
      optimizer: optimizer,
      lossFunction: meanSquaredError,
      callbacks: [testEvents.handler])
    
    try! trainingLoop.fit(for: 2)
    XCTAssertEqual(testEvents.events, fitEvents, 
      "The training loop didn't pass through all expected events.")
  }
  
  class TestCancelBatch {
    var events: [TrainingLoopEvent] = []
    
    public func handler<L: TrainingLoopProtocol>(_ loop: inout L, event: TrainingLoopEvent) throws {
      events.append(event)
      if event == .batchStart {
        throw TrainingLoopAction.cancelBatch
      }
    }
  }
    
  func testCancelBatch() {
    let testEvents = TestCancelBatch()
    var trainingLoop = TrainingLoop(
      training: data.trainingEpochs, 
      validation: data.validationBatches, 
      model: model, 
      optimizer: optimizer,
      lossFunction: meanSquaredError,
      callbacks: [testEvents.handler])
    
    try! trainingLoop.fit(for: 2)
      
    let batchEvents: [TrainingLoopEvent] = [.batchStart, .batchEnd]
    let thisTrainingStepsEvents = [[TrainingLoopEvent]](repeating: batchEvents, count: 10)
    let thisTrainingPhaseEvents: [TrainingLoopEvent] = (
      [.trainingStart] + thisTrainingStepsEvents.reduce(into: [], +=) + [.trainingEnd])
    let thisValidationStepsEvents = [[TrainingLoopEvent]](repeating: batchEvents, count: 5)
    let thisValidationPhaseEvents: [TrainingLoopEvent] = (
      [.validationStart] + thisValidationStepsEvents.reduce(into: [], +=) + [.validationEnd])
    let thisEpochEvents: [TrainingLoopEvent] = (
      [.epochStart] + thisTrainingPhaseEvents + thisValidationPhaseEvents + [.epochEnd])
    let expectedEvents: [TrainingLoopEvent] = (
      [.fitStart] + thisEpochEvents + thisEpochEvents + [.fitEnd]) 
    XCTAssertEqual(testEvents.events, expectedEvents, 
      "The training loop didn't pass through all expected events.")
  }  
    
  class TestCancelTraining {
    var events: [TrainingLoopEvent] = []
    
    public func handler<L: TrainingLoopProtocol>(_ loop: inout L, event: TrainingLoopEvent) throws {
      events.append(event)
      if event == .batchStart && Context.local.learningPhase == .training {
        throw TrainingLoopAction.cancelTraining
      }
    }
  }
    
  func testCancelTraining() {
    let testEvents = TestCancelTraining()
    var trainingLoop = TrainingLoop(
      training: data.trainingEpochs, 
      validation: data.validationBatches, 
      model: model, 
      optimizer: optimizer,
      lossFunction: meanSquaredError,
      callbacks: [testEvents.handler])
    
    try! trainingLoop.fit(for: 2)
    let thisEpochEvents: [TrainingLoopEvent] = (
      [.epochStart, .trainingStart, .batchStart, .trainingEnd] + validationPhaseEvents + [.epochEnd])
    let expectedEvents: [TrainingLoopEvent] = (
      [.fitStart] + thisEpochEvents + thisEpochEvents + [.fitEnd])
    XCTAssertEqual(testEvents.events, expectedEvents, 
      "The training loop didn't pass through all expected events.")
  }
    
  class TestCancelValidation {
    var events: [TrainingLoopEvent] = []
    
    public func handler<L: TrainingLoopProtocol>(_ loop: inout L, event: TrainingLoopEvent) throws {
      events.append(event)
      if event == .batchStart && Context.local.learningPhase == .inference {
        throw TrainingLoopAction.cancelValidation
      }
    }
  }
    
  func testCancelValidation() {
    let testEvents = TestCancelValidation()
    var trainingLoop = TrainingLoop(
      training: data.trainingEpochs, 
      validation: data.validationBatches, 
      model: model, 
      optimizer: optimizer,
      lossFunction: meanSquaredError,
      callbacks: [testEvents.handler])
    
    try! trainingLoop.fit(for: 2)
    let thisEpochEvents: [TrainingLoopEvent] = ([.epochStart] + trainingPhaseEvents + 
      [.validationStart, .batchStart, .validationEnd, .epochEnd])
    let expectedEvents: [TrainingLoopEvent] = (
      [.fitStart] + thisEpochEvents + thisEpochEvents + [.fitEnd])
    XCTAssertEqual(testEvents.events, expectedEvents, 
      "The training loop didn't pass through all expected events.")
  }
    
  class TestCancelEpoch {
    var events: [TrainingLoopEvent] = []
    
    public func handler<L: TrainingLoopProtocol>(_ loop: inout L, event: TrainingLoopEvent) throws {
      events.append(event)
      if event == .batchStart {
        throw TrainingLoopAction.cancelEpoch
      }
    }
  }
    
  func testCancelEpoch() {
    let testEvents = TestCancelEpoch()
    var trainingLoop = TrainingLoop(
      training: data.trainingEpochs, 
      validation: data.validationBatches, 
      model: model, 
      optimizer: optimizer,
      lossFunction: meanSquaredError,
      callbacks: [testEvents.handler])
    
    try! trainingLoop.fit(for: 2)
    let thisEpochEvents: [TrainingLoopEvent] = (
      [.epochStart, .trainingStart, .batchStart, .epochEnd])
    let expectedEvents: [TrainingLoopEvent] = (
      [.fitStart] + thisEpochEvents + thisEpochEvents + [.fitEnd])
    XCTAssertEqual(testEvents.events, expectedEvents, 
      "The training loop didn't pass through all expected events.")
  }
  
  class TestCancelFit {
    var events: [TrainingLoopEvent] = []
    
    public func handler<L: TrainingLoopProtocol>(_ loop: inout L, event: TrainingLoopEvent) throws {
      events.append(event)
      if event == .batchStart {
        throw TrainingLoopAction.cancelFit
      }
    }
  }
    
  func testCancelFit() {
    let testEvents = TestCancelFit()
    var trainingLoop = TrainingLoop(
      training: data.trainingEpochs, 
      validation: data.validationBatches, 
      model: model, 
      optimizer: optimizer,
      lossFunction: meanSquaredError,
      callbacks: [testEvents.handler])
    
    try! trainingLoop.fit(for: 2)
    let expectedEvents: [TrainingLoopEvent] = (
      [.fitStart, .epochStart, .trainingStart, .batchStart, .fitEnd])
    XCTAssertEqual(testEvents.events, expectedEvents, 
      "The training loop didn't pass through all expected events.")
  }
}

extension TrainingLoopTests {
  static var allTests = [
    ("testModelLearnsSomething", testModelLearnsSomething),
    ("testAllEvents", testAllEvents),
    ("testCancelBatch", testCancelBatch),
    ("testCancelTraining", testCancelTraining),
    ("testCancelValidation", testCancelValidation),
    ("testCancelEpoch", testCancelEpoch),
    ("testCancelFit", testCancelFit),
  ]
}
