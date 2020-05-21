import TensorFlow

// Workaround https://bugs.swift.org/browse/TF-1122 that prevents us from registering a
// loss function inside our TrainingLoop struct
public final class LossFunctionWrapper<Output: Differentiable, Target> {
  public typealias F = @differentiable (Output, @noDerivative Target) -> Tensor<Float> 
  public var f: F
  init(_ f: @escaping F) { self.f = f }
} 

public protocol TrainingLoopProtocol {
  // Associatedtypes
  /// The type of the sequence of epochs for the training data.
  associatedtype Training where Training: Sequence, Training.Element: Collection,
    Training.Element.Element == LabeledData<Opt.Model.Input, Target>
  /// The type of the collection of batches for the validation data.
  associatedtype Validation where Validation: Collection, 
    Validation.Element == LabeledData<Opt.Model.Input, Target> 
  /// The type of the target of our model.
  associatedtype Target
  /// The type of the optimizer used.
  associatedtype Opt: Optimizer where Opt.Model: Module

  // Typealiases
  /// The type of the model.
  typealias Model = Opt.Model
  /// The type of the input of the model.
  typealias Input = Opt.Model.Input
  /// The type of the output of the model.
  typealias Output = Opt.Model.Output
  /// The type of a batch.
  typealias Batch = LabeledData<Input, Target>
  // In a wrapper for now because of TF-1122.
  /// The type of the loss function.
  typealias LossFunction = LossFunctionWrapper<Output, Target>

  // Data
  /// The training epochs.
  var training: Training { get }
  /// The validation batches.
  var validation: Validation { get }

  // Model, optimizer and loss function
  /// The model.
  var model: Model { get set }
  /// The optimizer.
  var optimizer: Opt { get set }
  /// The loss function.
  var lossFunction: LossFunction { get set }

  // Temporary data
  /// The last input fed to the model.
  var lastInput: Input? { get set }
  /// The last target.
  var lastTarget: Target? { get set }
  /// The last predictions of the model
  var lastOutput: Output? { get set }
  /// The last loss.
  var lastLoss: Tensor<Float>? { get set }
}

public protocol TrainingLoopCallback {
  mutating func call<T: TrainingLoopProtocol>(on trainingLoop: T, event: TrainingLoopEvent) throws
}

/// The events that occur during a call to `fit` in the `TrainingLoop`
///
/// - Note: The method is called `fit` and not `train` because it trains the model and validates it.
///   Each epoch is composed of a *training* phase and a *validation* phase.
public enum TrainingLoopEvent {
  /// The start of a fit.
  case fitStart
  /// The end of a fit.
  case fitEnd
  /// The start of one epoch (training + validation).
  case epochStart
  /// The start of one epoch (training + validation).
  case epochEnd
  /// The start of a training phase.
  case trainingStart
  /// The end of a training phase.
  case trainingEnd
  /// The start of a validation phase.
  case validationStart
  /// The end of a validation phase.
  case validationEnd
  /// The start of a training or inference step on a batch.
  case batchStart
  /// The end of a training or inference step on a batch.
  case batchEnd
}

/// A generic training loop.
///
/// - Parameter `Training`: the type of the sequence of epochs for training data.
/// - Parameter `Validation`: the type of the collection of batches for validation.
/// - Parameter `Target`: the type of the target.
/// - Parameter `Opt`: the type of the optimizer used.
public struct TrainingLoop<Training: Sequence, Validation: Collection, Target, Opt: Optimizer>: TrainingLoopProtocol
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
  // In a wrapper for now because of TF-1122.
  /// The type of the loss function.
  public typealias LossFunction = LossFunctionWrapper<Output, Target>
      
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
  public var lossFunction: LossFunction
      
  // Callbacks
  /// The callbacks used to customize the training loop
  public var callbacks: [TrainingLoopCallback] = []
  
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
              lossFunction: @escaping LossFunction.F) {
    self.training = training
    self.validation = validation
    self.model = model
    self.optimizer = optimizer
    self.lossFunction = LossFunction(lossFunction)
  }
}

public extension TrainingLoop {
  mutating func inferenceStep() throws {
    guard let data = lastInput else { return }
    lastOutput = model(data)
    guard let target = lastTarget else { return }
    lastLoss = lossFunction.f(lastOutput!, target)
  }

  mutating func trainingStep() throws {
    guard let data = lastInput else { return }
    guard let target = lastTarget else { return }
    let (loss, gradient) = valueWithGradient(at: model) { (model: Model) -> Tensor<Float> in
      let predictions = model(data)
      lastOutput = predictions
      return lossFunction.f(predictions, target)
    }
    lastLoss = loss
    optimizer.update(&model, along: gradient)
  }
}

enum TrainingLoopAction: Error {
    case cancelBatch
    case cancelTraining
    case cancelValidation
    case cancelEpoch
    case cancelFit
}

public extension TrainingLoop {
  mutating func callEvent(_ event: TrainingLoopEvent) throws {
    for i in callbacks.indices {
      try callbacks[i].call(on: self, event: event)
    }
  }
}

public extension TrainingLoop {
  mutating func fit(for epochs: Int, callbacks: [TrainingLoopCallback],
                    step: (inout Self) throws -> Void = { try $0.inferenceStep() },
                    trainingStep: (inout Self) throws -> Void = { try $0.trainingStep() }) throws {
    self.callbacks += callbacks
    do{
      try callEvent(.fitStart)
      for batches in training.prefix(epochs) {
        do { 
          try callEvent(.epochStart)
          do {      
            try callEvent(.trainingStart)
            for batch in batches {
              (lastInput, lastTarget) = (batch.data, batch.label)
              do {
                try callEvent(.batchStart)
                try trainingStep(&self)
              } catch TrainingLoopAction.cancelBatch {}
              try callEvent(.batchEnd)
            }
          } catch TrainingLoopAction.cancelTraining {}
          try callEvent(.trainingEnd)
          do {   
            try callEvent(.validationStart)
            for batch in validation {
              (lastInput, lastTarget) = (batch.data, batch.label)
              do {
                try callEvent(.batchStart)
                try inferenceStep(&self)
              } catch TrainingLoopAction.cancelBatch {}
              try callEvent(.batchEnd)
            }
          } catch TrainingLoopAction.cancelValidation {}
          try callEvent(.validationEnd)
        } catch TrainingLoopAction.cancelEpoch {}
        try callEvent(.epochEnd)
      }
    } catch TrainingLoopAction.cancelFit {}
    try callEvent(.fitEnd)
  }
}