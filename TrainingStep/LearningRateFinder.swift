import TensorFlow

/// A callback that will launch a mock training with exponentially growing learning rates.
public struct LearningRateFinder: TrainingLoopCallback {
  /// The minimum learning rate.
  private let schedule: (Float) -> Float
  /// The number of iterations in the mock training.
  private let iterationCount: Int
  /// The smoothing factor for the moving average of the losses.
  private let smoothingFactor: Float
  /// The current iteration index.
  private var iterationIndex: Int = 0
  /// The smoothed losses at each iteration.
  public var smoothedLosses: [Float] = []
  /// The learning rates at each iteration.
  public var learningRates: [Float] = []
    
  /// Creates an instance from `schedule`.
  public init(minLearningRate: Float = 1e-7, maxLearningRate: Float = 10, 
              iterationCount: Int = 100, smoothingFactor: Float = 0.98) {
    self.schedule = makeSchedule(.exponential, from: minLearningRate, to: maxLearningRate)
    self.iterationCount = iterationCount
    self.smoothingFactor = smoothingFactor 
  }
   
  /// Inspect `trainingLoop` at `event` and can change its state accordingly.
  public mutating func call<T: TrainingLoopProtocol>(
    on trainingLoop: T, event: TrainingLoopEvent
  ) throws {
    switch event {
    // Sets total number of batches at the start of fit.
    case .fitStart:
      iterationIndex = 0
      smoothedLosses = []
      learningRates = []

    // Sets the proper learning rate before the training step.
    case .batchStart:
      let percent = Float(iterationIndex) / Float(iterationCount-1)
      var opt = trainingLoop.optimizer
      opt.learningRate = schedule(percent) as! T.Opt.Scalar
      learningRates.append(schedule(percent))
        
    // Record the smoothed loss and ends training if necessary.
    case .batchEnd:
      iterationIndex += 1
      let smoothedLoss: Float
      if let lastLoss = smoothedLosses.last {
        smoothedLoss = (lastLoss * smoothingFactor 
          + trainingLoop.lastLoss!.scalarized() * (1 - smoothingFactor))
      } else {
        smoothedLoss = trainingLoop.lastLoss!.scalarized()
      }
      smoothedLosses.append(smoothedLoss)
      if let minLoss = smoothedLosses.min() {
        if (smoothedLoss > 4.0 * minLoss || iterationIndex >= iterationCount) {
          throw TrainingLoopAction.cancelFit
        }
      }
        
    // Skip all validation phases.
    case .validationStart:
      throw TrainingLoopAction.cancelValidation

    default: return
    }
  }
}

// TODO: save and restore the model at the end.
extension TrainingLoop {
  /// Launches a mock training from `minLearningRate` to `maxLearningRate` and returns the
  /// `LearningRateFinder`.
  public mutating func learningRateFinder(
    minLearningRate: Float = 1e-7, 
    maxLearningRate: Float = 10, 
    iterationCount: Int = 100
  ) -> LearningRateFinder {
    callbacks.append(LearningRateFinder())
    try! fit(for: iterationCount)
    let lrFinder = callbacks.popLast() as! LearningRateFinder
    return lrFinder
  }
}