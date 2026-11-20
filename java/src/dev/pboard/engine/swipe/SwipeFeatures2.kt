package dev.pboard.engine.swipe

import dev.pboard.engine.TouchPoint
import kotlin.math.hypot
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min

/**
 * Layout-agnostic gesture features of the v2 swipe model: Kotlin port of `tools/swipe/swipefeat2.py` (keep in sync;
 * `NeuralSwipeTest` checks it against vectors written by that script). Built once per layout from the centres of
 * its keys (any layout, any size):
 *
 * 1. Key units: pixels divided by the median nearest-neighbour distance between key centres.
 * 2. Frames: one each time progress = arc / [stepS] + time / [stepT] passes an integer, plus the final sample.
 * 3. Per-frame features ([N_FEAT]): position in the layout frame (key-centre bounding box, [-1, 1]), motion and time
 *    since the previous frame (in steps), first/last flags, log frame size in key units.
 * 4. [keyX]/[keyY]: key centres in the frame ([0, 1] per axis), for the model's spatial head.
 */
class SwipeFeatures2(keyCx: FloatArray, keyCy: FloatArray, val stepS: Float = 0.5f, val stepT: Float = 70f, val maxFrames: Int = 160) {
    /** Pixels per key unit. */
    val unit: Float
    private val x0: Double; private val sx: Double; private val y0: Double; private val sy: Double
    val keyX: FloatArray
    val keyY: FloatArray
    val keyCount: Int get() = keyX.size

    init {
        val n = keyCx.size
        require(n >= 2) { "a layout needs at least two keys" }
        val nn = DoubleArray(n) { i ->
            var best = Double.POSITIVE_INFINITY
            for (j in 0 until n) if (j != i) best = min(best, hypot((keyCx[i] - keyCx[j]).toDouble(), (keyCy[i] - keyCy[j]).toDouble()))
            best
        }
        nn.sort()
        unit = (if (n % 2 == 1) nn[n / 2] else 0.5 * (nn[n / 2 - 1] + nn[n / 2])).toFloat()
        var minX = Double.POSITIVE_INFINITY; var maxX = Double.NEGATIVE_INFINITY
        var minY = Double.POSITIVE_INFINITY; var maxY = Double.NEGATIVE_INFINITY
        for (i in 0 until n) {
            val kx = keyCx[i].toDouble() / unit; val ky = keyCy[i].toDouble() / unit
            minX = min(minX, kx); maxX = max(maxX, kx); minY = min(minY, ky); maxY = max(maxY, ky)
        }
        x0 = minX; sx = max(1.0, maxX - minX); y0 = minY; sy = max(1.0, maxY - minY)
        keyX = FloatArray(n) { ((keyCx[it].toDouble() / unit - x0) / sx).toFloat() }
        keyY = FloatArray(n) { ((keyCy[it].toDouble() / unit - y0) / sy).toFloat() }
    }

    private var px = DoubleArray(256); private var py = DoubleArray(256); private var pt = DoubleArray(256); private var prog = DoubleArray(256)
    private var fx = DoubleArray(256); private var fy = DoubleArray(256); private var ft = DoubleArray(256)

    /** Frame count of the last [features] call. */
    var frames = 0; private set

    /** Features of [path] (pixels, ms) into [out] (row-major [frames × N_FEAT], grown when needed; returned). */
    fun features(path: List<TouchPoint>, out: FloatArray): FloatArray {
        val n = path.size
        if (px.size < n) { val c = n * 2; px = DoubleArray(c); py = DoubleArray(c); pt = DoubleArray(c); prog = DoubleArray(c) }
        val t0 = path[0].t
        var total = 0.0; var dur = 0.0
        for (i in 0 until n) {
            val p = path[i]
            px[i] = p.x.toDouble() / unit; py[i] = p.y.toDouble() / unit; pt[i] = (p.t - t0).toDouble()
            if (i > 0) { total += hypot(px[i] - px[i - 1], py[i] - py[i - 1]); dur += max(0.0, pt[i] - pt[i - 1]) }
        }
        var stepS = this.stepS.toDouble(); var stepT = this.stepT.toDouble()
        val est = total / stepS + dur / stepT
        if (est > maxFrames - 2) { val k = est / (maxFrames - 2); stepS *= k; stepT *= k }
        // Progress per sample; samples that add none are dropped (as np.interp needs increasing abscissae).
        var m = 0
        var tMax = Double.NEGATIVE_INFINITY
        var acc = 0.0
        for (i in 0 until n) {
            tMax = max(tMax, pt[i])
            if (i > 0) acc += hypot(px[i] - px[i - 1], py[i] - py[i - 1]) / stepS + max(0.0, pt[i] - pt[i - 1]) / stepT
            if (m > 0 && acc - prog[m - 1] <= 1e-9) continue
            prog[m] = acc; px[m] = px[i]; py[m] = py[i]; pt[m] = tMax; m++
        }
        val end = prog[m - 1]
        var count = 0
        fun emit(x: Double, y: Double, t: Double) {
            if (count == fx.size) { val c = count * 2; fx = fx.copyOf(c); fy = fy.copyOf(c); ft = ft.copyOf(c) }
            fx[count] = x; fy[count] = y; ft[count] = t; count++
        }
        if (m == 1) emit(px[0], py[0], pt[0])
        else {
            var j = 0
            var target = 0.0
            while (target < end) {
                while (j < m - 2 && prog[j + 1] < target) j++
                val f = ((target - prog[j]) / (prog[j + 1] - prog[j])).coerceIn(0.0, 1.0)
                emit(px[j] + f * (px[j + 1] - px[j]), py[j] + f * (py[j + 1] - py[j]), pt[j] + f * (pt[j + 1] - pt[j]))
                target += 1.0
            }
            if (count == 0 || end - (target - 1.0) > 1e-6) emit(px[m - 1], py[m - 1], pt[m - 1])
        }
        frames = count
        val res = if (out.size >= count * N_FEAT) out else FloatArray(count * N_FEAT)
        java.util.Arrays.fill(res, 0, count * N_FEAT, 0f)
        val lsx = ln(sx).toFloat(); val lsy = ln(sy).toFloat()
        for (k in 0 until count) {
            val o = k * N_FEAT
            res[o] = ((fx[k] - x0) / sx * 2 - 1).toFloat()
            res[o + 1] = ((fy[k] - y0) / sy * 2 - 1).toFloat()
            if (k > 0) {
                res[o + 2] = ((fx[k] - fx[k - 1]) / stepS).toFloat()
                res[o + 3] = ((fy[k] - fy[k - 1]) / stepS).toFloat()
                res[o + 4] = (min(ft[k] - ft[k - 1], 4 * stepT) / stepT).toFloat()
            }
            if (k == 0) res[o + 5] = 1f
            if (k == count - 1) res[o + 6] = 1f
            res[o + 7] = lsx; res[o + 8] = lsy
        }
        return res
    }

    companion object {
        const val N_FEAT = 9
    }
}
