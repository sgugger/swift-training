import TensorFlow

public protocol Metric {
  var value: Float? { get };
  mutating func reset()
  mutating func accumulate<Output, Target>(output: Output, target: Target)
}

public struct Recorder: TrainingLoopCallback {
  public var metrics: [Metric]
  public var trainingLosses: [Float] = []
  public var validationLosses: [Float] = []
  public var metricResults: [[Float?]] = [[]]

  private var summedLosses: Tensor<Float> = .zero
  private var samplesCount: Int = 0
    
  public init(metrics: [Metric] = []) {
    self.metrics = metrics
  }
    
  public mutating func call<T: TrainingLoopProtocol>(
    on trainingLoop: T, event: TrainingLoopEvent
  ) throws {
    switch event {
    case .trainingStart, .validationStart:
      summedLosses = .zero
      samplesCount = .zero
      for i in metrics.indices {
        metrics[i].reset()
      }
    case .batchEnd:
      // TODO: deal with the batch size in a more general way.
      let batchSize = (trainingLoop.lastTarget as! Tensor<Float>).shape[0]
      summedLosses += trainingLoop.lastLoss! * Float(batchSize)
      samplesCount += batchSize
      for i in metrics.indices {
        metrics[i].accumulate(output: trainingLoop.lastOutput!, target: trainingLoop.lastTarget!)
      }
    case .trainingEnd:
      trainingLosses.append(summedLosses.scalarized() / Float(samplesCount))
    case .validationEnd:
      validationLosses.append(summedLosses.scalarized() / Float(samplesCount))
      metricResults.append(metrics.map(\.value))
    default: return
    }
  }
}