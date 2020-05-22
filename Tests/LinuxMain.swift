import UtilsTests
import TrainingStepTests
import XCTest

var tests = [XCTestCaseEntry]()
tests += UtilsTests.allTests()
tests += TrainingStepTests.allTests()
XCTMain(tests)