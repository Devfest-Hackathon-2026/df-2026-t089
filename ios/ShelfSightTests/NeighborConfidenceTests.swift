import XCTest

/// Android の NeighborConfidenceTest と同じ確かめ方。
final class NeighborConfidenceTests: XCTestCase {
    private func top(_ a: String, _ b: String, _ p: Float) -> [(String, Float)] { [(a, p), (b, 1 - p)] }
    private func sigmoid(_ z: Double) -> Float { Float(1 / (1 + exp(-z))) }

    func testFeaturesMatchThePythonScript() {
        let c = NeighborConfidence(coef: [1, 2, 3, 4], intercept: 0.5, k: 4, radius: 3)
        XCTAssertEqual(c.adjust(Vec3(0, 0, 0), sizeM: 1, top: top("A", "B", 0.5)), sigmoid(0.5), accuracy: 1e-5)
        _ = c.adjust(Vec3(1, 0, 0), sizeM: 1, top: top("B", "A", 0.6))
        let z = log(0.8 / 0.2) + 2 + 3 * 0.5 + 4 * 0.45 + 0.5
        XCTAssertEqual(c.adjust(Vec3(0.5, 0, 0), sizeM: 1, top: top("A", "B", 0.8)), sigmoid(z), accuracy: 1e-4)
    }

    func testFarPointsAndTheSameSpotAreNotNeighbours() {
        let c = NeighborConfidence(coef: [0, 1, 0, 0], intercept: 0, k: 4, radius: 3)
        _ = c.adjust(Vec3(0, 0, 0), sizeM: 0.1, top: top("A", "B", 0.9))
        XCTAssertEqual(c.adjust(Vec3(5, 0, 0), sizeM: 0.1, top: top("A", "B", 0.9)), 0.5, accuracy: 1e-5)
        XCTAssertEqual(c.adjust(Vec3(5.05, 0, 0), sizeM: 0.1, top: top("A", "B", 0.9)), 0.5, accuracy: 1e-5)
        XCTAssertGreaterThan(c.adjust(Vec3(5.2, 0, 0), sizeM: 0.1, top: top("A", "B", 0.9)), 0.7)
    }

    func testWithoutWeightsTheFirstProbabilityIsKept() {
        XCTAssertEqual(NeighborConfidence.parse(nil).adjust(Vec3(0, 0, 0), sizeM: 1, top: top("A", "B", 0.83)), 0.83)
    }
}
