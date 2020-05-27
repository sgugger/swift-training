import TensorFlow

/// The shapes of usual hyperparameters schedule.
public enum ScheduleShape {
  case constant
  case cosine
  case linear
  case exponential
  case polynomial(Float)
}

/// Returns a schedule using `shape` from `start` to `end`.
public func makeSchedule(_ shape: ScheduleShape, from start: Float, to end: Float? = nil
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

/// Returns a callback that will change the learning rate according to `schedule`.
public func learningRateScheduler<L: TrainingLoopProtocol>(
  _ schedule: @escaping (Float) -> Float) -> TrainingLoopCallback<L> {
  var batchesCount: Int = 0
   
  return {(loop, event) throws -> Void in 
    // Sets the proper learning rate before the training step.
    if event == .batchStart {
      if Context.local.learningPhase == .inference { return }
      if batchesCount == 0 {
        batchesCount = loop.batchCount! * loop.epochCount!
      }
      let iterIndex = loop.batchIndex! + loop.epochIndex! * loop.batchCount!
      let percent = Float(iterIndex) / Float(batchesCount-1)
      var opt = loop.optimizer
      opt.learningRate = schedule(percent) as! L.Opt.Scalar
    }
  }
}