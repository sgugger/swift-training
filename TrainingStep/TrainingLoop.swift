import TensorFlow
import Utils

// Workaround https://bugs.swift.org/browse/TF-1122 that prevents us from registering a
// loss function inside our TrainingLoop struct
public final class LossFunctionWrapper<Output: Differentiable, Target> {
  public typealias F = @differentiable (Output, @noDerivative Target) -> Tensor<Float> 
  public var f: F
  init(_ f: @escaping F) { self.f = f }
}

/// Types whose elements represent a training loop.
/// 
/// - Note: This protocol is mainly there to give us an easy type for a generic `TrainingLoop`
///   and unless you need to rewrite your own training loop entirely, you should use `TrainingLoop`.
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

/// Types whose elements are callbacks that can inject custom behavior in a training loop.
public protocol TrainingLoopCallback {
  /// Inspect `trainingLoop` at `event` and can change its state accordingly.
  mutating func call<T: TrainingLoopProtocol>(on trainingLoop: T, event: TrainingLoopEvent) throws
}

/// A generic training loop.
///
/// - Parameter `Training`: the type of the sequence of epochs for training data.
/// - Parameter `Validation`: the type of the collection of batches for validation.
/// - Parameter `Target`: the type of the target.
/// - Parameter `Opt`: the type of the optimizer used.
public struct TrainingLoop<
  Training: Sequence, Validation: Collection, Target, Opt: Optimizer
>: TrainingLoopProtocol where 
  Training.Element: Collection, Training.Element.Element == LabeledData<Opt.Model.Input, Target>,
  Validation.Element == LabeledData<Opt.Model.Input, Target>, Opt.Model: Module 
{
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
  /// The default step used for inference.
  mutating func inferenceStep() throws {
    guard let data = lastInput else { return }
    lastOutput = model(data)
    guard let target = lastTarget else { return }
    lastLoss = lossFunction.f(lastOutput!, target)
  }

  /// The default step used for training.
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

/// Control flow of the training loop.
///
/// - Note: Each of the "end" event is called after its corresponding "cancel" action for cleanup.
enum TrainingLoopAction: Error {
  /// Abort actions in the current training/inference step and goes to the next batch.  
  case cancelBatch
  /// Abort actions in the current training phase and goes to the validation phase.
  case cancelTraining
  /// Abort actions in the current validation phase and goes to the next epoch.
  case cancelValidation
  /// Abort actions in the current epoch and goes to the next epoch.
  case cancelEpoch
  /// Abort actions in the current fit and ends fitting.
  case cancelFit
}

extension TrainingLoop {
  /// Call `event` on all callbacks.
  mutating private func callEvent(_ event: TrainingLoopEvent) throws {
    for i in callbacks.indices {
      try callbacks[i].call(on: self, event: event)
    }
  }
}

extension TrainingLoop {
  /// Performs `step` on each of `batches`.
  mutating private func multipleSteps<Batches: Collection>(
    on batches: Batches, step: (inout Self) throws -> Void
  ) throws where Batches.Element == Batch {
    for batch in batches {
      (lastInput, lastTarget) = (batch.data, batch.label)
      do {
        try callEvent(.batchStart)
        try step(&self)
      } catch TrainingLoopAction.cancelBatch {}
      try callEvent(.batchEnd)
    }
  }
}

public extension TrainingLoop {
  /// Fit the model for `epochs` using `callbacks` to customize the default training loop.
  ///
  /// - Parameters:
  ///   - inferenceStep: The step used during the validation phase of each epoch. The default value
  ///     uses the `inferenceStep` method of `TrainingLoop`.
  ///   - trainingStep: The step used during the training phase of each epoch. The default value
  ///     uses the `trainingStep` method of `TrainingLoop`. 
  mutating func fit(
    for epochs: Int, callbacks: [TrainingLoopCallback] = [],
    inferenceStep: (inout Self) throws -> Void = { try $0.inferenceStep() },
    trainingStep: (inout Self) throws -> Void = { try $0.trainingStep() }
  ) throws {
    self.callbacks += callbacks
    do{
      try callEvent(.fitStart)
      for batches in training.prefix(epochs) {
        do { 
          try callEvent(.epochStart)

          // Training phase
          do {      
            try callEvent(.trainingStart)
            try multipleSteps(on: batches, step: trainingStep)
          } catch TrainingLoopAction.cancelTraining {}
          try callEvent(.trainingEnd)

          // Validation phase
          do {   
            try callEvent(.validationStart)
            try multipleSteps(on: validation, step: inferenceStep)
          } catch TrainingLoopAction.cancelValidation {}
          try callEvent(.validationEnd)
        } catch TrainingLoopAction.cancelEpoch {}

        try callEvent(.epochEnd)
      }
    } catch TrainingLoopAction.cancelFit {}
    try callEvent(.fitEnd)
  }
}