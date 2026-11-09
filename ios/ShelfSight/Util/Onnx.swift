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
        /// 入力のバッファ。毎回作らずに使い回す（解析のたびにメモリが増えていたため）。同時に動かさないよう lock で守る
        private var inputBuffer: NSMutableData?
        private let lock = NSLock()
        /// 1 回動かすたびに、ONNX Runtime が取っておいた作業用のメモリ（CPU のメモリアリーナ）のうち使っていない分を OS に返す。
        /// 返さないと、取っておく分が増える一方で、iOS のアプリごとの上限（約 3GB）を超えて止められる
        private let runOptions: ORTRunOptions?
        /// 計算を Core ML に任せているか（画面の表示用）。
        let usesCoreML: Bool

        init(path: String, threads: Int) throws {
            guard let env = Onnx.env else { throw Failure(description: "ONNX Runtime を初期化できません") }
            let options = try ORTSessionOptions()
            try options.setGraphOptimizationLevel(.all)
            try options.setIntraOpNumThreads(Int32(threads))
            // 入力の形ごとにメモリの割り当てを覚えておく仕組み（memory pattern）を止める。形が変わるたびに取り直していた
            try? options.addConfigEntry(withKey: "session.enable_mem_pattern", value: "0")
            // 計算を Core ML（Neural Engine・GPU）に任せる。CPU で動かすと、途中の計算結果のための作業用メモリを
            // ONNX Runtime が malloc で取り続け（枠の検出 1 回で約 19MB）、3GB ほどで iOS に止められていた。
            // Core ML に任せた部分は ONNX Runtime の作業用メモリを使わない。Core ML が使えない演算だけ CPU に残る
            var coreML = false
            if ORTIsCoreMLExecutionProviderAvailable() {
                let o = ORTCoreMLExecutionProviderOptions()
                o.createMLProgram = true            // 新しい形式（YOLO の演算をより多く Core ML で動かせる）
                o.enableOnSubgraphs = true
                o.onlyAllowStaticInputShapes = true // 入力の形はいつも同じ
                do {
                    try options.appendCoreMLExecutionProvider(with: o)
                    coreML = true
                } catch {
                    logInfo("Onnx", "Core ML を使えないので CPU で動かす: \(error)")
                }
            }
            usesCoreML = coreML
            session = try ORTSession(env: env, modelPath: path, sessionOptions: options)
            logInfo("Onnx", "\((path as NSString).lastPathComponent): \(coreML ? "Core ML" : "CPU")")
            let run = try? ORTRunOptions()
            try? run?.addConfigEntry(withKey: "memory.enable_memory_arena_shrinkage", value: "cpu:0")
            runOptions = run
            input = try session.inputNames().first ?? "input"
            output = try session.outputNames().first ?? "output"
        }

        /// 入力 1 つ・出力 1 つのモデルを動かし、出力の値（行優先）と形を返す。
        func run(_ data: [Float], shape: [Int]) throws -> (values: [Float], shape: [Int]) {
            lock.lock()
            defer { lock.unlock() }
            // 出力（autorelease される）を、値を取り出したらすぐ捨てる
            return try autoreleasepool { try runOnce(data, shape: shape) }
        }

        private func runOnce(_ data: [Float], shape: [Int]) throws -> (values: [Float], shape: [Int]) {
            let length = data.count * MemoryLayout<Float>.size
            let bytes: NSMutableData
            if let b = inputBuffer, b.length == length {
                bytes = b
            } else {
                bytes = NSMutableData(length: length)!
                inputBuffer = bytes
            }
            data.withUnsafeBytes { src in
                bytes.replaceBytes(in: NSRange(location: 0, length: length), withBytes: src.baseAddress!)
            }
            let value = try ORTValue(tensorData: bytes, elementType: .float, shape: shape.map { NSNumber(value: $0) })
            let outputs = try session.run(withInputs: [input: value], outputNames: [output], runOptions: runOptions)
            guard let out = outputs[output] else { throw Failure(description: "出力 \(output) がありません") }
            let outShape = try out.tensorTypeAndShapeInfo().shape.map { $0.intValue }
            let raw = try out.tensorData() as Data
            let values = raw.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            return (values, outShape)
        }
    }
}
