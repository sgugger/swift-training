import UtilsTests
import TrainingStepTests
import DifferentiableStepTests
import XCTest

var tests = [XCTestCaseEntry]()
tests += UtilsTests.allTests()
tests += TrainingStepTests.allTests()
tests += DifferentiableStepTests.allTests()
XCTMain(tests)