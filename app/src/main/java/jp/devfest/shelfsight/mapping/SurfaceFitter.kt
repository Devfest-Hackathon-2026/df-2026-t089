package jp.devfest.shelfsight.mapping

import java.util.Random
import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.sqrt

/**
 * 面の材料。ARCore が検出した平面か、点から見つけた平面。
 *
 * @param polygon ARCore が検出した範囲（ワールド座標 x, y, z の凸多角形）。点から見つけた面は null
 */
internal class PlaneSpec(
    val origin: FloatArray,
    val normal: FloatArray,
    val kind: SurfaceKind,
    val polygon: FloatArray?,
    val area: Float,
    /** 遠くの深度から見つけた面。位置の誤差が大きいので、近くで見つかった面にまとめやすくする。 */
    val fromFarPoints: Boolean = false,
)

/** 点で裏付けられた範囲の面。頂点はワールド座標 x, y, z。 */
internal class FittedSurface(
    val reference: PlaneSpec,
    val members: List<PlaneSpec>,
    val kind: SurfaceKind,
    val normal: FloatArray,
    val triangles: FloatArray,
    val outline: FloatArray,
    val area: Float,
    /** 面の座標系。点 = origin + u * axisU + v * axisV。壁なら v は真上。 */
    val basis: SurfaceBasis,
)

/** 面の上の 2 次元座標系と、面が広がっている範囲（メートル）。 */
internal class SurfaceBasis(
    val origin: FloatArray,
    val axisU: FloatArray,
    val axisV: FloatArray,
    val uMin: Float,
    val uMax: Float,
    val vMin: Float,
    val vMax: Float,
)

internal class FitResult(val surfaces: List<FittedSurface>, val discovered: List<PlaneSpec>)

/**
 * 深度の点を手がかりに面を補正する。ARCore に依存しない。
 *
 * - 同じ平面に乗る ARCore の平面を 1 枚にまとめる
 * - 平面に乗っていてつながっている点の範囲まで面を広げる。形は凸多角形に限られない
 * - どの面にも乗らない点から、ARCore が見落とした平面を探す
 * - 遠くの深度は壁にだけ使う。誤差が大きいので、床は近くの点だけで作る
 *
 * 点の範囲は格子で表すので、ギザギザにならないよう格子の向きを店の壁の向きに合わせる。
 */
internal class SurfaceFitter {
    /** 店の壁の向き（x, z の単位ベクトル）。面の向きが揺れないよう、一度決めたら変えない。 */
    private var dominantX = 1f
    private var dominantZ = 0f
    private var dominantFixed = false

    /**
     * @param points ワールド座標 x, y, z の順。先頭 [pointCount] 点を使う
     * @param farPoints 遠くまで届くが誤差の大きい点。形式は [points] と同じ
     */
    fun fit(
        points: FloatArray,
        pointCount: Int,
        specs: List<PlaneSpec>,
        ceilingY: Float,
        farPoints: FloatArray = FloatArray(0),
        farCount: Int = 0,
    ): FitResult {
        val groups = buildGroups(specs)
        fixDominantDirection(groups)
        groups.forEach { it.prepareBasis(dominantX, dominantZ) }

        val residual = FloatList()
        assignPoints(points, pointCount, groups, ceilingY, ON_PLANE_TOLERANCE_M, residual) { group, x, y, z ->
            group.addPoint(x, y, z)
        }
        val farResidual = FloatList()
        val walls = groups.filter { it.reference.kind == SurfaceKind.WALL }
        assignPoints(farPoints, farCount, walls, ceilingY, FAR_ON_PLANE_TOLERANCE_M, farResidual) { group, x, y, z ->
            group.addFarPoint(x, y, z)
        }
        val surfaces = groups.mapNotNull { fitGroup(it) }

        val floorY = specs.filter { it.kind == SurfaceKind.FLOOR }.minOfOrNull { it.origin[1] }
        val discovered = if (floorY == null) emptyList() else discover(residual, floorY) + discoverFar(farResidual, floorY)
        return FitResult(surfaces, discovered)
    }

    private fun buildGroups(specs: List<PlaneSpec>): List<Group> {
        val groups = ArrayList<Group>()
        val ordered = specs.sortedWith(compareBy<PlaneSpec> { it.polygon == null }.thenBy { it.fromFarPoints }.thenByDescending { it.area })
        for (spec in ordered) {
            val group = groups.firstOrNull { it.accepts(spec) }
            if (group != null) group.members += spec else groups += Group(spec)
        }
        return groups
    }

    private fun fixDominantDirection(groups: List<Group>) {
        if (dominantFixed) return
        val wall = groups
            .filter { it.reference.kind == SurfaceKind.WALL && it.reference.area >= MIN_DOMINANT_WALL_AREA_M2 }
            .maxByOrNull { it.reference.area } ?: return
        val n = wall.reference.normal
        val length = sqrt(n[0] * n[0] + n[2] * n[2])
        if (length < 1e-3f) return
        dominantX = n[2] / length
        dominantZ = -n[0] / length
        dominantFixed = true
    }

    /** 点はいちばん近い面に 1 回だけ割り当てる。どの面にも乗らない点は [residual] に回す。 */
    private inline fun assignPoints(
        points: FloatArray,
        count: Int,
        groups: List<Group>,
        ceilingY: Float,
        tolerance: Float,
        residual: FloatList,
        add: (Group, Float, Float, Float) -> Unit,
    ) {
        for (i in 0 until count) {
            val x = points[i * 3]
            val y = points[i * 3 + 1]
            val z = points[i * 3 + 2]
            if (y > ceilingY) continue
            var best: Group? = null
            var bestDistance = tolerance
            for (group in groups) {
                val distance = abs(group.signedDistance(x, y, z))
                if (distance < bestDistance) {
                    bestDistance = distance
                    best = group
                }
            }
            if (best != null) add(best, x, y, z) else residual.add3(x, y, z)
        }
    }

    private fun fitGroup(group: Group): FittedSurface? {
        val seeds = group.seedPolygons()
        val detected = seeds.isNotEmpty()
        val grid = CellGrid.build(group, seeds) ?: return null
        grid.bridgeGaps()
        val keep = if (detected) grid.componentsTouchingSeeds() else grid.componentsPassingGates(NEAR_GATES)
        if (keep.none { it }) return null
        return group.toSurface(grid, keep)
    }

    private fun discover(residual: FloatList, floorY: Float): List<PlaneSpec> {
        val found = ArrayList<PlaneSpec>()
        val remaining = residual.filterTriples { _, y, _ -> y > floorY + MIN_HEIGHT_ABOVE_FLOOR_M }
        var attempts = 0
        var points = remaining
        while (attempts < MAX_DISCOVERY_ATTEMPTS && found.size < MAX_DISCOVERED_PER_FIT) {
            attempts++
            val candidate = bestVerticalPlane(points, DISCOVERY_TOLERANCE_M, MIN_DISCOVERY_POINTS, far = false) ?: break
            if (candidateHasSurface(points, candidate, DISCOVERY_TOLERANCE_M, NEAR_GATES)) found += candidate
            points = points.filterTriples { x, y, z -> abs(distance(candidate, x, y, z)) >= DISCOVERY_TOLERANCE_M }
        }
        attempts = 0
        while (attempts < MAX_DISCOVERY_ATTEMPTS && found.size < MAX_DISCOVERED_PER_FIT) {
            attempts++
            val candidate = bestHorizontalPlane(points, floorY) ?: break
            if (candidateHasSurface(points, candidate, DISCOVERY_TOLERANCE_M, NEAR_GATES)) found += candidate
            points = points.filterTriples { x, y, z -> abs(distance(candidate, x, y, z)) >= DISCOVERY_TOLERANCE_M }
        }
        return found
    }

    /** 遠くの深度から壁だけを探す。床の高さの揺れを壁と間違えないよう、床から十分に離れた点を使う。 */
    private fun discoverFar(residual: FloatList, floorY: Float): List<PlaneSpec> {
        val found = ArrayList<PlaneSpec>()
        var points = residual.filterTriples { _, y, _ -> y > floorY + FAR_MIN_HEIGHT_ABOVE_FLOOR_M }
        var attempts = 0
        while (attempts < MAX_DISCOVERY_ATTEMPTS && found.size < MAX_DISCOVERED_PER_FIT) {
            attempts++
            val candidate = bestVerticalPlane(points, FAR_DISCOVERY_TOLERANCE_M, MIN_FAR_DISCOVERY_POINTS, far = true) ?: break
            if (candidateHasSurface(points, candidate, FAR_DISCOVERY_TOLERANCE_M, FAR_GATES)) found += candidate
            points = points.filterTriples { x, y, z -> abs(distance(candidate, x, y, z)) >= FAR_DISCOVERY_TOLERANCE_M }
        }
        return found
    }

    private fun candidateHasSurface(points: FloatList, candidate: PlaneSpec, tolerance: Float, gates: Gates): Boolean {
        val group = Group(candidate).also { it.prepareBasis(dominantX, dominantZ) }
        var i = 0
        while (i < points.size) {
            val x = points[i]
            val y = points[i + 1]
            val z = points[i + 2]
            if (abs(group.signedDistance(x, y, z)) < tolerance) {
                if (candidate.fromFarPoints) group.addFarPoint(x, y, z) else group.addPoint(x, y, z)
            }
            i += 3
        }
        val grid = CellGrid.build(group, emptyList()) ?: return false
        grid.bridgeGaps()
        return grid.componentsPassingGates(gates).any { it }
    }

    /** 2 点を通る縦の線を試し、点がいちばん多く乗るものを選ぶ。乱数は固定して結果を揺らさない。 */
    private fun bestVerticalPlane(points: FloatList, tolerance: Float, minPoints: Int, far: Boolean): PlaneSpec? {
        val total = points.size / 3
        if (total < minPoints) return null
        val stride = maxOf(1, total / MAX_RANSAC_SAMPLES)
        val sampleCount = total / stride
        val xs = FloatArray(sampleCount) { points[it * stride * 3] }
        val zs = FloatArray(sampleCount) { points[it * stride * 3 + 2] }
        val random = Random(RANSAC_SEED)

        var bestCount = 0
        var bestNx = 0f
        var bestNz = 0f
        var bestC = 0f
        repeat(RANSAC_ITERATIONS) {
            val a = random.nextInt(sampleCount)
            val b = random.nextInt(sampleCount)
            val dx = xs[b] - xs[a]
            val dz = zs[b] - zs[a]
            val length = sqrt(dx * dx + dz * dz)
            if (length < RANSAC_MIN_SPAN_M || length > RANSAC_MAX_SPAN_M) return@repeat
            val nx = -dz / length
            val nz = dx / length
            val c = nx * xs[a] + nz * zs[a]
            var count = 0
            for (i in 0 until sampleCount) {
                if (abs(nx * xs[i] + nz * zs[i] - c) < tolerance) count++
            }
            if (count > bestCount) {
                bestCount = count
                bestNx = nx
                bestNz = nz
                bestC = c
            }
        }
        if (bestCount * stride < minPoints) return null
        return refineVertical(points, bestNx, bestNz, bestC, tolerance, far)
    }

    /** 乗っている点で線を引き直し、店の壁の向きに近ければ揃える。 */
    private fun refineVertical(points: FloatList, nx0: Float, nz0: Float, c: Float, tolerance: Float, far: Boolean): PlaneSpec? {
        var count = 0
        var sumX = 0f
        var sumY = 0f
        var sumZ = 0f
        var i = 0
        while (i < points.size) {
            if (abs(nx0 * points[i] + nz0 * points[i + 2] - c) < tolerance) {
                sumX += points[i]
                sumY += points[i + 1]
                sumZ += points[i + 2]
                count++
            }
            i += 3
        }
        if (count == 0) return null
        val cx = sumX / count
        val cy = sumY / count
        val cz = sumZ / count

        var xx = 0f
        var xz = 0f
        var zz = 0f
        i = 0
        while (i < points.size) {
            if (abs(nx0 * points[i] + nz0 * points[i + 2] - c) < tolerance) {
                val dx = points[i] - cx
                val dz = points[i + 2] - cz
                xx += dx * dx
                xz += dx * dz
                zz += dz * dz
            }
            i += 3
        }
        // 2x2 の共分散行列の、固有値が大きい方の固有ベクトルが線の向き
        val trace = xx + zz
        val root = sqrt(((xx - zz) * (xx - zz) + 4 * xz * xz).coerceAtLeast(0f))
        val lambda = (trace + root) / 2f
        var ax = xz
        var az = lambda - xx
        if (abs(ax) + abs(az) < 1e-6f) {
            ax = -nz0
            az = nx0
        }
        val length = sqrt(ax * ax + az * az)
        ax /= length
        az /= length

        val along = ax * dominantX + az * dominantZ
        val across = ax * -dominantZ + az * dominantX
        if (abs(along) >= SNAP_COSINE) {
            val sign = if (along > 0) 1f else -1f
            ax = dominantX * sign
            az = dominantZ * sign
        } else if (abs(across) >= SNAP_COSINE) {
            val sign = if (across > 0) 1f else -1f
            ax = -dominantZ * sign
            az = dominantX * sign
        }
        return PlaneSpec(floatArrayOf(cx, cy, cz), floatArrayOf(-az, 0f, ax), SurfaceKind.WALL, null, count * VOXEL_AREA_M2, far)
    }

    /** 高さの分布で、点が集まっている水平面を探す。棚やケースの上面を想定する。 */
    private fun bestHorizontalPlane(points: FloatList, floorY: Float): PlaneSpec? {
        val minY = floorY + MIN_RAISED_HEIGHT_M
        val bins = IntArray(HEIGHT_BINS)
        var i = 0
        while (i < points.size) {
            val bin = ((points[i + 1] - minY) / HEIGHT_BIN_M).toInt()
            if (points[i + 1] >= minY && bin < HEIGHT_BINS) bins[bin]++
            i += 3
        }
        var best = -1
        var bestCount = 0
        for (b in 1 until HEIGHT_BINS - 1) {
            val count = bins[b - 1] + bins[b] + bins[b + 1]
            if (count > bestCount) {
                bestCount = count
                best = b
            }
        }
        if (best < 0 || bestCount < MIN_DISCOVERY_POINTS) return null
        val center = minY + (best + 0.5f) * HEIGHT_BIN_M
        var count = 0
        var sumX = 0f
        var sumY = 0f
        var sumZ = 0f
        i = 0
        while (i < points.size) {
            if (abs(points[i + 1] - center) < HEIGHT_BIN_M * 1.5f) {
                sumX += points[i]
                sumY += points[i + 1]
                sumZ += points[i + 2]
                count++
            }
            i += 3
        }
        if (count == 0) return null
        return PlaneSpec(
            floatArrayOf(sumX / count, sumY / count, sumZ / count),
            floatArrayOf(0f, 1f, 0f),
            SurfaceKind.RAISED,
            null,
            count * VOXEL_AREA_M2,
        )
    }

    private fun distance(spec: PlaneSpec, x: Float, y: Float, z: Float): Float =
        (x - spec.origin[0]) * spec.normal[0] + (y - spec.origin[1]) * spec.normal[1] + (z - spec.origin[2]) * spec.normal[2]

    /** 1 枚にまとめる平面の集まりと、そこに乗った点。 */
    private class Group(val reference: PlaneSpec) {
        val members = arrayListOf(reference)
        val us = FloatList()
        val vs = FloatList()
        val farUs = FloatList()
        val farVs = FloatList()

        private val ox = reference.origin[0]
        private val oy = reference.origin[1]
        private val oz = reference.origin[2]
        private var nx = 0f
        private var ny = 1f
        private var nz = 0f
        private var ux = 1f
        private var uy = 0f
        private var uz = 0f
        private var vx = 0f
        private var vy = 0f
        private var vz = 1f

        init {
            if (reference.kind == SurfaceKind.WALL) {
                val n = reference.normal
                val length = sqrt(n[0] * n[0] + n[2] * n[2]).coerceAtLeast(1e-6f)
                nx = n[0] / length
                ny = 0f
                nz = n[2] / length
            }
        }

        fun accepts(spec: PlaneSpec): Boolean {
            if (spec.kind != reference.kind) return false
            val n = reference.normal
            val m = spec.normal
            if (abs(n[0] * m[0] + n[1] * m[1] + n[2] * m[2]) < SAME_PLANE_COSINE) return false
            val limit = if (spec.fromFarPoints) FAR_SAME_PLANE_OFFSET_M else SAME_PLANE_OFFSET_M
            return abs(signedDistance(spec.origin[0], spec.origin[1], spec.origin[2])) < limit
        }

        fun prepareBasis(dominantX: Float, dominantZ: Float) {
            if (reference.kind == SurfaceKind.WALL) {
                // u は壁に沿った水平方向、v は真上。縦横が店の向きと揃う。
                ux = nz; uy = 0f; uz = -nx
                vx = 0f; vy = 1f; vz = 0f
            } else {
                ux = dominantX; uy = 0f; uz = dominantZ
                vx = -dominantZ; vy = 0f; vz = dominantX
            }
        }

        fun signedDistance(x: Float, y: Float, z: Float): Float =
            (x - ox) * nx + (y - oy) * ny + (z - oz) * nz

        fun addPoint(x: Float, y: Float, z: Float) {
            us.add((x - ox) * ux + (y - oy) * uy + (z - oz) * uz)
            vs.add((x - ox) * vx + (y - oy) * vy + (z - oz) * vz)
        }

        fun addFarPoint(x: Float, y: Float, z: Float) {
            farUs.add((x - ox) * ux + (y - oy) * uy + (z - oz) * uz)
            farVs.add((x - ox) * vx + (y - oy) * vy + (z - oz) * vz)
        }

        /** 各平面の多角形を u, v の座標に直す。 */
        fun seedPolygons(): List<FloatArray> = members.mapNotNull { spec ->
            val p = spec.polygon ?: return@mapNotNull null
            FloatArray(p.size / 3 * 2).also { out ->
                for (i in 0 until p.size / 3) {
                    val dx = p[i * 3] - ox
                    val dy = p[i * 3 + 1] - oy
                    val dz = p[i * 3 + 2] - oz
                    out[i * 2] = dx * ux + dy * uy + dz * uz
                    out[i * 2 + 1] = dx * vx + dy * vy + dz * vz
                }
            }
        }

        fun toSurface(grid: CellGrid, keep: BooleanArray): FittedSurface {
            val triangles = FloatList()
            val outline = FloatList()
            val used = BooleanArray(keep.size)
            var cells = 0
            var minCol = Int.MAX_VALUE
            var maxCol = Int.MIN_VALUE
            var minRow = Int.MAX_VALUE
            var maxRow = Int.MIN_VALUE
            for (row in 0 until grid.height) {
                for (col in 0 until grid.width) {
                    if (!keep[grid.index(col, row)]) continue
                    cells++
                    minCol = minOf(minCol, col)
                    maxCol = maxOf(maxCol, col)
                    minRow = minOf(minRow, row)
                    maxRow = maxOf(maxRow, row)
                    addOutline(grid, keep, col, row, outline)
                    if (used[grid.index(col, row)]) continue
                    addRectangle(grid, keep, used, col, row, triangles)
                }
            }
            return FittedSurface(
                reference, members, reference.kind, floatArrayOf(nx, ny, nz),
                triangles.toArray(), outline.toArray(), cells * CELL_M * CELL_M,
                SurfaceBasis(
                    floatArrayOf(ox, oy, oz), floatArrayOf(ux, uy, uz), floatArrayOf(vx, vy, vz),
                    (grid.minU + minCol) * CELL_M, (grid.minU + maxCol + 1) * CELL_M,
                    (grid.minV + minRow) * CELL_M, (grid.minV + maxRow + 1) * CELL_M,
                ),
            )
        }

        /** 隣り合うセルを長方形にまとめて、三角形の数を減らす。 */
        private fun addRectangle(grid: CellGrid, keep: BooleanArray, used: BooleanArray, col: Int, row: Int, out: FloatList) {
            var right = col
            while (right + 1 < grid.width && keep[grid.index(right + 1, row)] && !used[grid.index(right + 1, row)]) right++
            var top = row
            while (top + 1 < grid.height) {
                val next = top + 1
                if ((col..right).any { !keep[grid.index(it, next)] || used[grid.index(it, next)] }) break
                top = next
            }
            for (r in row..top) for (c in col..right) used[grid.index(c, r)] = true
            val u0 = (grid.minU + col) * CELL_M
            val u1 = (grid.minU + right + 1) * CELL_M
            val v0 = (grid.minV + row) * CELL_M
            val v1 = (grid.minV + top + 1) * CELL_M
            addVertex(out, u0, v0); addVertex(out, u1, v0); addVertex(out, u1, v1)
            addVertex(out, u0, v0); addVertex(out, u1, v1); addVertex(out, u0, v1)
        }

        private fun addOutline(grid: CellGrid, keep: BooleanArray, col: Int, row: Int, out: FloatList) {
            val u0 = (grid.minU + col) * CELL_M
            val u1 = u0 + CELL_M
            val v0 = (grid.minV + row) * CELL_M
            val v1 = v0 + CELL_M
            fun filled(c: Int, r: Int) = c in 0 until grid.width && r in 0 until grid.height && keep[grid.index(c, r)]
            if (!filled(col - 1, row)) { addVertex(out, u0, v0); addVertex(out, u0, v1) }
            if (!filled(col + 1, row)) { addVertex(out, u1, v0); addVertex(out, u1, v1) }
            if (!filled(col, row - 1)) { addVertex(out, u0, v0); addVertex(out, u1, v0) }
            if (!filled(col, row + 1)) { addVertex(out, u0, v1); addVertex(out, u1, v1) }
        }

        private fun addVertex(out: FloatList, u: Float, v: Float) {
            out.add(ox + u * ux + v * vx)
            out.add(oy + u * uy + v * vy)
            out.add(oz + u * uz + v * vz)
        }
    }

    /** 面の上の格子。点が十分に集まったセルだけを面として扱う。 */
    private class CellGrid(val minU: Int, val minV: Int, val width: Int, val height: Int) {
        val filled = BooleanArray(width * height)
        val seeded = BooleanArray(width * height)

        fun index(col: Int, row: Int) = row * width + col

        /** 間のセルが 1 つか 2 つだけ抜けているなら埋める。斜めから見た床は点がまばらになりやすい。 */
        fun bridgeGaps() {
            repeat(BRIDGE_PASSES) {
                val snapshot = filled.copyOf()
                for (row in 0 until height) {
                    for (col in 0 until width) {
                        if (snapshot[index(col, row)]) continue
                        val horizontal = anyFilled(snapshot, col, row, -1, 0) && anyFilled(snapshot, col, row, 1, 0)
                        val vertical = anyFilled(snapshot, col, row, 0, -1) && anyFilled(snapshot, col, row, 0, 1)
                        if (horizontal || vertical) filled[index(col, row)] = true
                    }
                }
            }
        }

        private fun anyFilled(cells: BooleanArray, col: Int, row: Int, dc: Int, dr: Int): Boolean {
            for (step in 1..BRIDGE_REACH_CELLS) {
                val c = col + dc * step
                val r = row + dr * step
                if (c in 0 until width && r in 0 until height && cells[index(c, r)]) return true
            }
            return false
        }

        fun componentsTouchingSeeds(): BooleanArray = selectComponents { cells, _ -> cells.any { seeded[it] } }

        /** 点だけで見つけた面は誤検知を防ぐため、広さと詰まり具合を確かめる。 */
        fun componentsPassingGates(gates: Gates): BooleanArray = selectComponents { cells, _ ->
            val cols = cells.map { it % width }
            val rows = cells.map { it / width }
            val spanU = cols.max() - cols.min() + 1
            val spanV = rows.max() - rows.min() + 1
            cells.size >= gates.minCells && spanU >= gates.minSpanU && spanV >= gates.minSpanV &&
                cells.size.toFloat() / (spanU * spanV) >= gates.minFill
        }

        private fun selectComponents(accept: (List<Int>, Int) -> Boolean): BooleanArray {
            val keep = BooleanArray(filled.size)
            val seen = BooleanArray(filled.size)
            val stack = IntArray(filled.size)
            for (start in filled.indices) {
                if (!filled[start] || seen[start]) continue
                val cells = ArrayList<Int>()
                var top = 0
                stack[top++] = start
                seen[start] = true
                while (top > 0) {
                    val cell = stack[--top]
                    cells += cell
                    val col = cell % width
                    val row = cell / width
                    for (dr in -1..1) for (dc in -1..1) {
                        val c = col + dc
                        val r = row + dr
                        if (c !in 0 until width || r !in 0 until height) continue
                        val next = index(c, r)
                        if (filled[next] && !seen[next]) {
                            seen[next] = true
                            stack[top++] = next
                        }
                    }
                }
                if (accept(cells, start)) cells.forEach { keep[it] = true }
            }
            return keep
        }

        companion object {
            fun build(group: Group, seeds: List<FloatArray>): CellGrid? {
                val us = group.us
                val vs = group.vs
                val farUs = group.farUs
                val farVs = group.farVs
                var minU = Int.MAX_VALUE
                var maxU = Int.MIN_VALUE
                var minV = Int.MAX_VALUE
                var maxV = Int.MIN_VALUE
                fun extend(u: Float, v: Float) {
                    val cu = cellOf(u)
                    val cv = cellOf(v)
                    if (cu < minU) minU = cu
                    if (cu > maxU) maxU = cu
                    if (cv < minV) minV = cv
                    if (cv > maxV) maxV = cv
                }
                for (i in 0 until us.size) extend(us[i], vs[i])
                for (i in 0 until farUs.size) extend(farUs[i], farVs[i])
                for (seed in seeds) for (i in 0 until seed.size / 2) extend(seed[i * 2], seed[i * 2 + 1])
                if (minU == Int.MAX_VALUE) return null
                val width = maxU - minU + 1
                val height = maxV - minV + 1
                if (width.toLong() * height > MAX_GRID_CELLS) return null

                val grid = CellGrid(minU, minV, width, height)
                val counts = IntArray(width * height)
                for (i in 0 until us.size) counts[grid.index(cellOf(us[i]) - minU, cellOf(vs[i]) - minV)]++
                for (i in 0 until farUs.size) {
                    counts[grid.index(cellOf(farUs[i]) - minU, cellOf(farVs[i]) - minV)] += FAR_POINT_WEIGHT
                }
                for (i in counts.indices) grid.filled[i] = counts[i] >= MIN_POINTS_PER_CELL
                for (seed in seeds) grid.markSeed(seed)
                return grid
            }

            private fun cellOf(value: Float) = floor(value / CELL_M).toInt()
        }

        /** ARCore が検出した範囲は点がなくても残す。今までより面が減らないようにする。 */
        private fun markSeed(polygon: FloatArray) {
            val count = polygon.size / 2
            if (count == 0) return
            var sumU = 0f
            var sumV = 0f
            for (i in 0 until count) {
                sumU += polygon[i * 2]
                sumV += polygon[i * 2 + 1]
            }
            // どんなに小さな面でも、中心のセルは必ず残す。
            setSeed(cellIndexOf(sumU / count, sumV / count))
            for (row in 0 until height) {
                for (col in 0 until width) {
                    val u = (minU + col + 0.5f) * CELL_M
                    val v = (minV + row + 0.5f) * CELL_M
                    if (insideConvex(polygon, u, v)) setSeed(index(col, row))
                }
            }
        }

        private fun cellIndexOf(u: Float, v: Float) =
            index(floor(u / CELL_M).toInt() - minU, floor(v / CELL_M).toInt() - minV)

        private fun setSeed(cell: Int) {
            seeded[cell] = true
            filled[cell] = true
        }

        private fun insideConvex(polygon: FloatArray, u: Float, v: Float): Boolean {
            val count = polygon.size / 2
            var sign = 0
            for (i in 0 until count) {
                val j = (i + 1) % count
                val cross = (polygon[j * 2] - polygon[i * 2]) * (v - polygon[i * 2 + 1]) -
                    (polygon[j * 2 + 1] - polygon[i * 2 + 1]) * (u - polygon[i * 2])
                val s = if (cross > 0) 1 else if (cross < 0) -1 else 0
                if (s == 0) continue
                if (sign != 0 && s != sign) return false
                sign = s
            }
            return true
        }
    }

    /** 点だけで面を認める条件。長さはセルの数で表す。 */
    private class Gates(val minCells: Int, val minSpanU: Int, val minSpanV: Int, val minFill: Float)

    /** ボクシングを避けた Float の可変長配列。 */
    private class FloatList {
        private var values = FloatArray(256)
        var size = 0
            private set

        operator fun get(index: Int) = values[index]

        fun add(value: Float) {
            if (size == values.size) values = values.copyOf(size * 2)
            values[size++] = value
        }

        fun add3(x: Float, y: Float, z: Float) {
            add(x); add(y); add(z)
        }

        fun toArray(): FloatArray = values.copyOf(size)

        inline fun filterTriples(keep: (Float, Float, Float) -> Boolean): FloatList {
            val out = FloatList()
            var i = 0
            while (i < size) {
                if (keep(values[i], values[i + 1], values[i + 2])) out.add3(values[i], values[i + 1], values[i + 2])
                i += 3
            }
            return out
        }
    }

    private companion object {
        const val CELL_M = 0.15f

        /** 5cm 単位の点なら、面に乗ったセルには最大 9 点入る。 */
        const val MIN_POINTS_PER_CELL = 3
        const val VOXEL_AREA_M2 = 0.0025f
        const val ON_PLANE_TOLERANCE_M = 0.07f
        const val SAME_PLANE_COSINE = 0.985f
        const val SAME_PLANE_OFFSET_M = 0.06f
        const val MIN_DOMINANT_WALL_AREA_M2 = 1f
        const val BRIDGE_PASSES = 2
        const val BRIDGE_REACH_CELLS = 2
        const val MAX_GRID_CELLS = 1_000_000L

        // 0.6 平方メートル以上で、縦横とも 45cm 以上、外接長方形の 55% 以上が埋まること。
        val NEAR_GATES = Gates(minCells = 27, minSpanU = 3, minSpanV = 3, minFill = 0.55f)

        // 遠くの深度は誤差が大きいので厳しくする。1.5 平方メートル以上で、幅 1m 以上、高さ 60cm 以上。
        val FAR_GATES = Gates(minCells = 67, minSpanU = 7, minSpanV = 4, minFill = 0.5f)

        /** 手がかりの点は 10cm 単位なので、面に乗ったセルに入る数が地図の点より少ない。2 点でセルを埋める。 */
        const val FAR_POINT_WEIGHT = 2
        const val FAR_ON_PLANE_TOLERANCE_M = 0.12f
        const val FAR_SAME_PLANE_OFFSET_M = 0.15f
        const val FAR_DISCOVERY_TOLERANCE_M = 0.12f
        const val MIN_FAR_DISCOVERY_POINTS = 150
        const val FAR_MIN_HEIGHT_ABOVE_FLOOR_M = 0.3f

        const val DISCOVERY_TOLERANCE_M = 0.05f
        const val MIN_DISCOVERY_POINTS = 200
        const val MIN_HEIGHT_ABOVE_FLOOR_M = 0.1f
        const val MIN_RAISED_HEIGHT_M = 0.25f
        const val MAX_DISCOVERY_ATTEMPTS = 4
        const val MAX_DISCOVERED_PER_FIT = 2
        const val MAX_RANSAC_SAMPLES = 20_000
        const val RANSAC_ITERATIONS = 200
        const val RANSAC_SEED = 20261106L
        const val RANSAC_MIN_SPAN_M = 0.4f
        const val RANSAC_MAX_SPAN_M = 6f
        const val SNAP_COSINE = 0.99f
        const val HEIGHT_BIN_M = 0.04f
        const val HEIGHT_BINS = 60
    }
}
