import Foundation
import os

/// 残りのメモリと、CPU 側（malloc）で使っている量。メモリが足りなくなる前に休む・作り直すのに使う。
enum Memory {
    /// malloc で今使っている量（MB）。
    static func mallocMB() -> Double {
        var stats = malloc_statistics_t()
        malloc_zone_statistics(nil, &stats)
        return Double(stats.size_in_use) / 1_048_576
    }

    /// OS に止められるまでに、あとどれだけメモリを使えるか（MB）。
    static func availableMB() -> Double { Double(os_proc_available_memory()) / 1_048_576 }
}
