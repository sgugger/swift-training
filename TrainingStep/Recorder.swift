import TensorFlow

/// Types whose elements represent a metric function.
public protocol Metric {
  /// The name of the metric (to be displayed with the value at the end of each epoch).
  var name: String { get }
  /// The value of the metric.
  var value: Float? { get }
  /// Resets the state before a new computation starts.
  mutating func reset()
  /// Accumulates model `output` and `target` in the state.
  mutating func accumulate<Output, Target>(output: Output, target: Target)
}

/// A callback that computes and stores the training and validation losses, as well as all the 
/// metrics.
public class Recorder {
  /// The list of all metrics to compute.
  public var metrics: [Metric]
  /// The training losses of each epoch.
  public var trainingLosses: [Float] = []
  /// The validation losses of each epoch.
  public var validationLosses: [Float] = []
  /// The computed metrics for each epochs.
  public var metricResults: [[Float?]] = []

  private var summedLosses: Tensor<Float> = .zero
  private var samplesCount: Int = 0
    
  /// Creates an instance from `metrics`.
  public init(metrics: [Metric] = []) {
    self.metrics = metrics
  }
   
  /// Inspect `trainingLoop` at `event` and can change its state accordingly.
  public func handler<L: TrainingLoopProtocol>(_ loop: L, event: TrainingLoopEvent) throws {
    switch event {

    // Resets the state for the computation of new loss and metrics.
    case .trainingStart, .validationStart:
      summedLosses = .zero
      samplesCount = .zero
      for i in metrics.indices {
        metrics[i].reset()
      }

    // Accumulates the latest results for the computation of the loss and metrics.
    case .batchEnd:
      // TODO: deal with the batch size in a more general way.
      let batchSize = (loop.lastTarget as! Tensor<Float>).shape[0]
      summedLosses += loop.lastLoss! * Float(batchSize)
      samplesCount += batchSize
      for i in metrics.indices {
        metrics[i].accumulate(output: loop.lastOutput!, target: loop.lastTarget!)
      }

    // Stores the training loss.
    case .trainingEnd:
      trainingLosses.append(summedLosses.scalarized() / Float(samplesCount))

    // Stores the validation loss and metrics
    case .validationEnd:
      validationLosses.append(summedLosses.scalarized() / Float(samplesCount))
      metricResults.append(metrics.map(\.value))

    // Prints the training and validation losses, as well as the metric values for this epoch.
    case .epochEnd:
      let names = ["Training loss", "Validation loss"] + metrics.map(\.name)
      let values = [trainingLosses.last, validationLosses.last] + metrics.map(\.value)
      let stringValues = values.map{ (v: Float?) -> String in
        if let value = v { 
          return String(format: "%.02f", value) 
        } else { 
          return "nil" 
        }
      }
      let namedValues = zip(names, stringValues).map{ "\($0): \($1)" }
      print("Epoch \(loop.epochIndex! + 1) -- " + namedValues.joined(separator: ", "))

    default: return
    }
  }
}