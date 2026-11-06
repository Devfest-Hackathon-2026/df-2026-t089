import Foundation
#if canImport(OnnxRuntimeBindings)
import OnnxRuntimeBindings
#else
import onnxruntime_objc
#endif

/// ONNX Runtime の薄い包み。商品認識（products/）で使う。端末内だけで動く。
enum Onnx {
    struct Failure: Error, CustomStringConvertible {
        let description: String
    }

    private static let env: ORTEnv? = try? ORTEnv(loggingLevel: .warning)

    final class Model {
        private let session: ORTSession
        private let input: String
        private let output: String

        init(path: String, threads: Int) throws {
            guard let env = Onnx.env else { throw Failure(description: "ONNX Runtime を初期化できません") }
            let options = try ORTSessionOptions()
            try options.setGraphOptimizationLevel(.all)
            try options.setIntraOpNumThreads(Int32(threads))
            session = try ORTSession(env: env, modelPath: path, sessionOptions: options)
            input = try session.inputNames().first ?? "input"
            output = try session.outputNames().first ?? "output"
        }

        /// 入力 1 つ・出力 1 つのモデルを動かし、出力の値（行優先）と形を返す。
        func run(_ data: [Float], shape: [Int]) throws -> (values: [Float], shape: [Int]) {
            let bytes = data.withUnsafeBytes { NSMutableData(bytes: $0.baseAddress!, length: $0.count) }
            let value = try ORTValue(tensorData: bytes, elementType: .float, shape: shape.map { NSNumber(value: $0) })
            let outputs = try session.run(withInputs: [input: value], outputNames: [output], runOptions: nil)
            guard let out = outputs[output] else { throw Failure(description: "出力 \(output) がありません") }
            let outShape = try out.tensorTypeAndShapeInfo().shape.map { $0.intValue }
            let raw = try out.tensorData() as Data
            let values = raw.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            return (values, outShape)
        }
    }
}
