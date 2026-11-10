import XCTest

/// Android の StoreModelTest と同じ観測を流して、同じ記述になることを確かめる（データは python products/store_sim.py --fixture）。
final class StoreModelTests: XCTestCase {
    private func resource(_ name: String, _ ext: String) throws -> Data {
        let url = try XCTUnwrap(Bundle(for: Self.self).url(forResource: name, withExtension: ext))
        return try Data(contentsOf: url)
    }

    func testMeasuredIslandKeepsItsDepthAndFitsMixedModuleWidths() throws {
        let model = StoreModel()
        for i in 0...100 {
            let z = 0.05 + Double(i) * 2.9 / 100
            model.addProduct(0.4, z, 1.56, z, "A", "")
            model.addProduct(3.6, z, 2.44, z, "B", "")
        }
        let island = try XCTUnwrap(model.estimate().fixtures.first { $0.kind == "island" })
        XCTAssertEqual(island.d, 0.88, accuracy: 0.05)
        XCTAssertEqual(island.n, 3)
        XCTAssertEqual(island.moduleWidths.filter { $0 > 1.05 }.count, 1)
        XCTAssertEqual(island.b - island.a, 2.9, accuracy: 0.05)
    }

    func testMatchesThePythonEstimator() throws {
        let all = String(decoding: try resource("store_model_trace", "tsv"), as: UTF8.self).replacingOccurrences(of: "\r\n", with: "\n")
        let parts = all.components(separatedBy: "\n===\n")
        XCTAssertEqual(parts.count, 2)
        let model = StoreModel(themes: StoreModel.parseThemes(try resource("store_themes", "json")))
        for line in parts[0].split(separator: "\n") {
            let p = line.components(separatedBy: "\t")
            switch p[0] {
            case "P": model.addProduct(Double(p[1])!, Double(p[2])!, Double(p[3])!, Double(p[4])!, p[5], p[6])
            case "C": model.addCounter(Double(p[1])!, Double(p[2])!)
            case "T": model.addCamera(Double(p[1])!, Double(p[2])!)
            default: break
            }
        }
        XCTAssertEqual(StoreModel.text(model.estimate()), parts[1].trimmingCharacters(in: .whitespacesAndNewlines))
    }
}
