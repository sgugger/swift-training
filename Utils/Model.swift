import TensorFlow

/// A linear regression model: `y = a * x  + b`.
public struct RegressionModel: Layer {
  /// The learnable parameter `a`.
  public var a: Tensor<Float>
  /// The learnable parameter `b`.
  public var b: Tensor<Float>

  public init(a: Tensor<Float>, b: Tensor<Float>) {
    self.a = a
    self.b = b
  }

  public init(a: Float, b: Float) {
    self.init(a: Tensor(a), b: Tensor(b))
  }

  @differentiable
  public func callAsFunction(_ x: Tensor<Float>) -> Tensor<Float> {
    return a * x + b
  }
}