import XCTest
import TensorFlow
import Utils

@testable import TrainingStep

fileprivate let entropy = ARC4RandomNumberGenerator(seed: 42)
fileprivate let data = RegressionData(entropy: entropy)
fileprivate let model = RegressionModel(a: 0.0, b: 0.0)
fileprivate let optimizer = SGD(for: model, learningRate: 0.1)

// For control flow tests
let batchEvents: [TrainingLoopEvent] = [.batchStart, .batchEnd]
let trainingStepsEvents = [[TrainingLoopEvent]](repeating: batchEvents, count: 10)
let trainingPhaseEvents: [TrainingLoopEvent] = (
  [.trainingStart] + trainingStepsEvents.reduce(into: [], +=) + [.trainingEnd])
let validationStepsEvents = [[TrainingLoopEvent]](repeating: batchEvents, count: 5)
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
  
  struct TestAllEvents: TrainingLoopCallback {
    var events: [TrainingLoopEvent] = []
  
    public mutating func call<T: TrainingLoopProtocol>(
      on trainingLoop: T, event: TrainingLoopEvent
    ) throws {
      events.append(event)
    }
  }
  
  func testAllEvents() {
    var trainingLoop = TrainingLoop(
      training: data.trainingEpochs, 
      validation: data.validationBatches, 
      model: model, 
      optimizer: optimizer,
      lossFunction: meanSquaredError,
      callbacks: [TestAllEvents()])
    
    try! trainingLoop.fit(for: 2)
    let events = (trainingLoop.callbacks[0] as! TestAllEvents).events
    XCTAssertEqual(events, fitEvents, "The training loop didn't pass through all expected events.")
  }
  
  struct TestCancelBatch: TrainingLoopCallback {
    var events: [TrainingLoopEvent] = []
    
    public mutating func call<T: TrainingLoopProtocol>(
      on trainingLoop: T, event: TrainingLoopEvent
    ) throws {
      events.append(event)
      if event == .batchStart {
        throw TrainingLoopAction.cancelBatch
      }
    }
  }
    
  func testCancelBatch() {
    var trainingLoop = TrainingLoop(
      training: data.trainingEpochs, 
      validation: data.validationBatches, 
      model: model, 
      optimizer: optimizer,
      lossFunction: meanSquaredError,
      callbacks: [TestCancelBatch()])
    
    try! trainingLoop.fit(for: 2)
    let events = (trainingLoop.callbacks[0] as! TestCancelBatch).events
    XCTAssertEqual(events, fitEvents, 
      "The training loop didn't pass through all expected events.")
  }  
    
  struct TestCancelTraining: TrainingLoopCallback {
    var events: [TrainingLoopEvent] = []
    
    public mutating func call<T: TrainingLoopProtocol>(
      on trainingLoop: T, event: TrainingLoopEvent
    ) throws {
      events.append(event)
      if event == .batchStart && Context.local.learningPhase == .training {
        throw TrainingLoopAction.cancelTraining
      }
    }
  }
    
  func testCancelTraining() {
    var trainingLoop = TrainingLoop(
      training: data.trainingEpochs, 
      validation: data.validationBatches, 
      model: model, 
      optimizer: optimizer,
      lossFunction: meanSquaredError,
      callbacks: [TestCancelTraining()])
    
    try! trainingLoop.fit(for: 2)
    let events = (trainingLoop.callbacks[0] as! TestCancelTraining).events
    let thisEpochEvents: [TrainingLoopEvent] = (
      [.epochStart, .trainingStart, .batchStart, .trainingEnd] + validationPhaseEvents + [.epochEnd])
    let expectedEvents: [TrainingLoopEvent] = (
      [.fitStart] + thisEpochEvents + thisEpochEvents + [.fitEnd])
    XCTAssertEqual(events, expectedEvents, 
      "The training loop didn't pass through all expected events.")
  }
    
  struct TestCancelValidation: TrainingLoopCallback {
    var events: [TrainingLoopEvent] = []
    
    public mutating func call<T: TrainingLoopProtocol>(
      on trainingLoop: T, event: TrainingLoopEvent
    ) throws {
      events.append(event)
      if event == .batchStart && Context.local.learningPhase == .inference {
        throw TrainingLoopAction.cancelValidation
      }
    }
  }
    
  func testCancelValidation() {
    var trainingLoop = TrainingLoop(
      training: data.trainingEpochs, 
      validation: data.validationBatches, 
      model: model, 
      optimizer: optimizer,
      lossFunction: meanSquaredError,
      callbacks: [TestCancelValidation()])
    
    try! trainingLoop.fit(for: 2)
    let events = (trainingLoop.callbacks[0] as! TestCancelValidation).events
    let thisEpochEvents: [TrainingLoopEvent] = ([.epochStart] + trainingPhaseEvents + 
      [.validationStart, .batchStart, .validationEnd, .epochEnd])
    let expectedEvents: [TrainingLoopEvent] = (
      [.fitStart] + thisEpochEvents + thisEpochEvents + [.fitEnd])
    XCTAssertEqual(events, expectedEvents, 
      "The training loop didn't pass through all expected events.")
  }
    
  struct TestCancelEpoch: TrainingLoopCallback {
    var events: [TrainingLoopEvent] = []
    
    public mutating func call<T: TrainingLoopProtocol>(
      on trainingLoop: T, event: TrainingLoopEvent
    ) throws {
      events.append(event)
      if event == .batchStart {
        throw TrainingLoopAction.cancelEpoch
      }
    }
  }
    
  func testCancelEpoch() {
    var trainingLoop = TrainingLoop(
      training: data.trainingEpochs, 
      validation: data.validationBatches, 
      model: model, 
      optimizer: optimizer,
      lossFunction: meanSquaredError,
      callbacks: [TestCancelEpoch()])
    
    try! trainingLoop.fit(for: 2)
    let events = (trainingLoop.callbacks[0] as! TestCancelEpoch).events
    let thisEpochEvents: [TrainingLoopEvent] = (
      [.epochStart, .trainingStart, .batchStart, .epochEnd])
    let expectedEvents: [TrainingLoopEvent] = (
      [.fitStart] + thisEpochEvents + thisEpochEvents + [.fitEnd])
    XCTAssertEqual(events, expectedEvents, 
      "The training loop didn't pass through all expected events.")
  }
  
  struct TestCancelFit: TrainingLoopCallback {
    var events: [TrainingLoopEvent] = []
    
    public mutating func call<T: TrainingLoopProtocol>(
      on trainingLoop: T, event: TrainingLoopEvent
    ) throws {
      events.append(event)
      if event == .batchStart {
        throw TrainingLoopAction.cancelFit
      }
    }
  }
    
  func testCancelFit() {
    var trainingLoop = TrainingLoop(
      training: data.trainingEpochs, 
      validation: data.validationBatches, 
      model: model, 
      optimizer: optimizer,
      lossFunction: meanSquaredError,
      callbacks: [TestCancelFit()])
    
    try! trainingLoop.fit(for: 2)
    let events = (trainingLoop.callbacks[0] as! TestCancelFit).events
    let expectedEvents: [TrainingLoopEvent] = (
      [.fitStart, .epochStart, .trainingStart, .batchStart, .fitEnd])
    XCTAssertEqual(events, expectedEvents, 
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
