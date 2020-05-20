import TensorFlow

// Note: This is a struct and not a tuple because we need the `Collatable`
// conformance below.
/// A tuple (data, label) that can be used to train a deep learning model.
///
/// - Parameter `Data`: the type of the input.
/// - Parameter `Label`: the type of the target.
public struct LabeledData<Data, Label> {
  /// The `data` of our sample (usually used as input for a model).
  public let data: Data
  /// The `label` of our sample (usually used as target for a model).
  public let label: Label

  /// Creates an instance from `data` and `label`.
  public init(data: Data, label: Label) {
    self.data = data
    self.label = label
  }
}

extension LabeledData: Collatable where Data: Collatable, Label: Collatable {
  /// Creates an instance from collating `samples`.
  public init<BatchSamples: Collection>(collating samples: BatchSamples)
  where BatchSamples.Element == Self {
    self.init(data: .init(collating: samples.map(\.data)),
              label: .init(collating: samples.map(\.label)))
  }
}

func generateDataset<Entropy: RandomNumberGenerator>(
  _ a: Float, _ b: Float, count: Int, entropy: inout Entropy
) -> [(Float, Float)] {
  return (0..<count).map { (_) -> (Float, Float) in
    let x = Float.random(in: -4...4, using: &entropy)
    let noise = Float.random(in: -0.1...0.1, using: &entropy)
    return (x, a * x + b + noise)
  }
}

func makeBatch<Samples: Collection>(samples: Samples) -> LabeledData<Tensor<Float>, Tensor<Float>> 
where Samples.Element == (Float, Float) {
  return LabeledData(data: Tensor(samples.map(\.0)), label: Tensor(samples.map(\.1)))
}

/// Synthetic data to test training on a simple linear regression `y = a * x  + b`.
public struct RegressionData<Entropy: RandomNumberGenerator> {
  /// The theoretical value for `a`.
  private let a: Float
  /// The theoretical value for `b`.
  private let b: Float
  /// The number of training batches.
  private let trainingCount: Int
  /// The number of validation batches.
  private let validationCount: Int
  /// The size of each batch.
  private let batchSize: Int
  /// The source of entropy used for deterministic randomness.
  private var entropy: Entropy
  /// The training samples.
  public let trainingSamples: [(Float, Float)]
  /// The validation samples.
  public let validationSamples: [(Float, Float)]

  /// The type of a concatenated batch.
  public typealias Batch = LabeledData<Tensor<Float>, Tensor<Float>>
  /// The type of the training batches (before concatenation).
  public typealias TrainingBatches = Slices<Sampling<[(Float, Float)], ArraySlice<Int>>>
  /// The type of the Epochs used for training.
  public typealias Training = LazyMapSequence<
    TrainingEpochs<[(Float, Float)], Entropy>,
    LazyMapSequence<TrainingBatches, Batch>
  >

  public var trainingEpochs: Training {
    TrainingEpochs(samples: trainingSamples, batchSize: batchSize, entropy: entropy)
      .lazy.map { (batches: TrainingBatches) ->  LazyMapSequence<TrainingBatches, Batch> in
        batches.lazy.map(makeBatch)          
      }
  }
    
  public var validationBatches: LazyMapSequence<Slices<[(Float, Float)]>, Batch> {
    validationSamples.inBatches(of: batchSize).lazy.map(makeBatch)
  }

  /// Creates an instance of data `(x, y)` almost satisfying `y = a * x + b` (with a bit of random
  /// noise) using `batchSize`.
  ///
  /// Parameters:
  ///  - trainingCount: The number of training batches.
  ///  - validationCount: The number of validation batches.
  ///  - entropy: The source of randonmness used.
  public init(a: Float = 2.0, b: Float = 3.0, batchSize: Int = 64, 
              trainingCount: Int = 10, validationCount: Int = 5, entropy: Entropy) {
    self.a = a
    self.b = b
    self.trainingCount = trainingCount
    self.validationCount = validationCount
    self.batchSize = batchSize
    self.entropy = entropy
    trainingSamples = generateDataset(a, b, count: trainingCount * batchSize, 
                                      entropy: &self.entropy)
    validationSamples = generateDataset(a, b, count: validationCount * batchSize, 
                                        entropy: &self.entropy)
  }
}