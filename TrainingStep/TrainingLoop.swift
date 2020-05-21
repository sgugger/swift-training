import TensorFlow

/// A generic training loop.
///
/// - Parameter `Training`: the type of the sequence of epochs for training data.
/// - Parameter `Validation`: the type of the collection of batches for validation.
/// - Parameter `Target`: the type of the target.
/// - Parameter `Opt`: the type of the optimizer used.
public struct TrainingLoop<Training: Sequence, Validation: Collection, Target, Opt: Optimizer> 
where Training.Element: Collection, Training.Element.Element == LabeledData<Opt.Model.Input, Target>,
  Validation.Element == LabeledData<Opt.Model.Input, Target>, Opt.Model: Module {
  // Typealiases
  /// The type of the model.
  public typealias Model = Opt.Model
  /// The type of the input of the model.
  public typealias Input = Opt.Model.Input
  /// The type of the output of the model.
  public typealias Output = Opt.Model.Output
  /// The type of a batch.
  public typealias Batch = LabeledData<Input, Target>
  /// The type of the loss function.
  public typealias LossFunction = @differentiable (Output, @noDerivative Target) -> Tensor<Float>
      
  // Data
  /// The training epochs.
  public let training: Training
  /// The validation batches.
  public let validation: Validation
  
  // Model, optimizer and loss function
  /// The model.
  public var model: Model
  /// The optimizer.
  public var optimizer: Opt
  /// The loss function
  public let lossFunction: LossFunction
  
  // Temporary data
  /// The last input fed to the model.
  public var lastInput: Input? = nil
  /// The last target.
  public var lastTarget: Target? = nil
  /// The last predictions of the model
  public var lastOutput: Output? = nil
  /// The last loss.
  public var lastLoss: Tensor<Float>? = nil
      
  /// Creates an instance from `training` and `validation` data, a `model`, an `optimizer` and a
  /// `lossFunction`.
  public init(training: Training, validation: Validation, model: Model, optimizer: Opt, 
              lossFunction: @escaping LossFunction) {
    self.training = training
    self.validation = validation
    self.model = model
    self.optimizer = optimizer
    self.lossFunction = lossFunction
  }
}

public extension TrainingLoop {
  mutating func trainingStep() {
    guard let data = lastInput else { return }
    guard let target = lastTarget else { return }
    let (loss, gradient) = valueWithGradient(at: model) { (model: Model) -> Tensor<Float> in
      let predictions = model(data)
      lastOutput = predictions
      return lossFunction(predictions, target)
    }
    lastLoss = loss
    optimizer.update(&model, along: gradient)
  }
}

public extension TrainingLoop {
  mutating func fit(for epochs: Int) {
    for batches in training.prefix(epochs) {
      for batch in batches {
        (lastInput, lastTarget) = (batch.data, batch.label)
        trainingStep()
      }
    }
  }
}