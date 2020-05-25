import TensorFlow

/// The shapes of usual hyperparameters schedule.
public enum SchedulerShape {
  case constant
  case cosine
  case linear
  case exponential
  case polynomial(Float)
}

/// Returns a schedule using `shape` from `start` to `end`.
public func makeSchedule(_ shape: SchedulerShape, from start: Float, to end: Float? = nil
) -> (Float) -> Float{
  switch shape {
  case .constant:
    return { _ in start }
  case .cosine:
    return { (x: Float) -> Float in
      // Cause the compiler doesn't like all in the same line.
      let slope: Float = (1.0 + cos(Float.pi * (1.0 - x))) / 2.0
      return start + (end! - start) * slope 
    }
  case .linear:
    return { start + (end! - start) * $0 }
  case .exponential:
    return { start * pow((end! / start), $0) }
  case .polynomial(let beta):
    return { start + (end! - start) * pow($0, beta) }
  }
}

/// A callback that will schedule the learning rate.
public struct Scheduler: TrainingLoopCallback {
  /// The list of all metrics to compute.
  public let schedule: (Float) -> Float
  /// List of all learning rates.
  public var learningRates: [Float] = []

  /// The total number of batches in the current call to fit.
  private var batchesCount: Int = 0
  /// The number of batches in one epoch.
  private var batchesPerEpoch: Int = 0
    
  /// Creates an instance from `schedule`.
  public init(schedule: @escaping (Float) -> Float) {
    self.schedule = schedule
  }
   
  /// Inspect `trainingLoop` at `event` and can change its state accordingly.
  public mutating func call<T: TrainingLoopProtocol>(
    on trainingLoop: T, event: TrainingLoopEvent
  ) throws {
    switch event {
    // Set total number of batches at the start of fit.
    case .fitStart:
      for mockBatches in trainingLoop.training.prefix(1) {
        batchesPerEpoch = mockBatches.count
        batchesCount = trainingLoop.epochCount! * batchesPerEpoch
      }
      learningRates = []

    // Set the proper learning rate before the training step.
    case .batchStart:
      if Context.local.learningPhase == .inference { return }
      let iterIndex = trainingLoop.batchIndex! + trainingLoop.epochIndex! * batchesPerEpoch
      let percent = Float(iterIndex) / Float(batchesCount-1)
      var opt = trainingLoop.optimizer
      opt.learningRate = schedule(percent) as! T.Opt.Scalar
      learningRates.append(schedule(percent))

    default: return
    }
  }
}