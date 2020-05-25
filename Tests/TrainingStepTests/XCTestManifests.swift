import XCTest

#if !os(macOS)
  public func allTests() -> [XCTestCaseEntry] {
    return [
      testCase(TrainingLoopTests.allTests),
      testCase(RecorderTests.allTests),
    ]
  }
#endif