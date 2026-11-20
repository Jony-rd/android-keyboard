package dev.pboard.engine.swipe

import dev.pboard.engine.TouchPoint
import dev.pboard.engine.decode.KeySpace
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Gesture features of the neural swipe model: Kotlin port of `tools/swipe/swipefeat.py` (keep in sync;
 * `SwipeFeaturesTest` checks it against vectors written by that script).
 *
 * 1. Grid units: the 26 letter keys are fitted per axis to the canonical staggered QWERTY grid
 *    (q = (0.5, 0.5), a = (1.0, 1.5), z = (2.0, 2.5)), so u is in key pitches and v in row pitches. This removes
 *    screen size, keyboard height and key aspect ratio.
 * 2. Frames: one every [stepS] grid units of arc length, or every [stepT] ms while the finger (nearly) rests,
 *    linearly interpolated, so the result does not depend on the touch sampling rate.
 * 3. Per-frame features ([N_FEAT]): position, motion since the previous frame, elapsed time, first/last flags and a
 *    Gaussian proximity to each letter key.
 */
class SwipeFeatures(val stepS: Float = 0.30f, val stepT: Float = 40f, val maxFrames: Int = 192, val proxSigma: Float = 0.55f) {

    /** Least-squares map from keyboard pixels to grid units: u = (x − ox) / px, v = (y − oy) / py. */
    class GridFit(val ox: Float, val px: Float, val oy: Float, val py: Float, val maxResidual: Float) {
        fun u(x: Float) = (x - ox) / px
        fun v(y: Float) = (y - oy) / py
    }

    /** Resampled frames (grid units, ms); arrays are reused between calls. */
    class Frames(capacity: Int) {
        var u = FloatArray(capacity); var v = FloatArray(capacity); var t = FloatArray(capacity)
        var size = 0
        internal fun add(fu: Float, fv: Float, ft: Float) {
            if (size == u.size) { u = u.copyOf(size * 2); v = v.copyOf(size * 2); t = t.copyOf(size * 2) }
            u[size] = fu; v[size] = fv; t[size] = ft; size++
        }
    }

    /**
     * Frames of a stroke given in grid units ([u], [v]) and ms ([t], non-decreasing), first [n] samples. Same
     * arithmetic as the Python reference (doubles internally, floats out).
     */
    fun resample(u: FloatArray, v: FloatArray, t: FloatArray, n: Int, out: Frames) {
        out.size = 0
        if (n == 0) return
        var total = 0.0
        for (i in 1 until n) total += hypot((u[i] - u[i - 1]).toDouble(), (v[i] - v[i - 1]).toDouble())
        val dur = if (n > 1) (t[n - 1] - t[0]).toDouble() else 0.0
        var stepS = this.stepS.toDouble(); var stepT = this.stepT.toDouble()
        val est = total / stepS + dur / stepT
        if (est > maxFrames - 2) { val k = est / (maxFrames - 2); stepS *= k; stepT *= k }
        out.add(u[0], v[0], t[0])
        var sSince = 0.0
        var tLast = t[0].toDouble()
        for (i in 1 until n) {
            val au = u[i - 1].toDouble(); val av = v[i - 1].toDouble(); val at = t[i - 1].toDouble()
            val bu = u[i].toDouble(); val bv = v[i].toDouble(); val bt = t[i].toDouble()
            val len = hypot(bu - au, bv - av)
            val d = maxOf(0.0, bt - at)
            var pos = 0.0
            while (true) {
                val fs = if (len > 1e-9) pos + (stepS - sSince) / len else Double.POSITIVE_INFINITY
                val tc = at + pos * d
                val ft = if (d > 1e-9) pos + (stepT - (tc - tLast)) / d else Double.POSITIVE_INFINITY
                val f = min(fs, ft)
                if (f > 1.0) { sSince += (1.0 - pos) * len; break }
                out.add((au + f * (bu - au)).toFloat(), (av + f * (bv - av)).toFloat(), (at + f * d).toFloat())
                sSince = 0.0
                tLast = at + f * d
                pos = f
            }
        }
        val lt = maxOf(t[n - 1], out.t[out.size - 1])
        if (sSince > 1e-6 || lt - tLast > 1e-6 || out.size == 1) out.add(u[n - 1], v[n - 1], lt)
    }

    /** Writes [frames]'s features into [x] (row-major [frames.size × N_FEAT]); grows it when needed. */
    fun features(frames: Frames, x: FloatArray): FloatArray {
        val n = frames.size
        val out = if (x.size >= n * N_FEAT) x else FloatArray(n * N_FEAT)
        java.util.Arrays.fill(out, 0, n * N_FEAT, 0f)
        val inv2s2 = 1f / (2f * proxSigma * proxSigma)
        for (j in 0 until n) {
            val o = j * N_FEAT
            val fu = frames.u[j]; val fv = frames.v[j]
            out[o] = (fu - 5f) / 3f
            out[o + 1] = (fv - 1.5f) / 1.5f
            if (j > 0) {
                out[o + 2] = (fu - frames.u[j - 1]) / stepS
                out[o + 3] = (fv - frames.v[j - 1]) / stepS
                out[o + 4] = min(frames.t[j] - frames.t[j - 1], 4 * stepT) / stepT
            }
            if (j == 0) out[o + 5] = 1f
            if (j == n - 1) out[o + 6] = 1f
            for (k in 0 until 26) {
                val du = fu - KEY_U[k]; val dv = fv - KEY_V[k]
                out[o + N_BASE + k] = exp(-(du * du + dv * dv) * inv2s2)
            }
        }
        return out
    }

    companion object {
        const val N_BASE = 7
        const val N_FEAT = N_BASE + 26
        const val N_CLASSES = 27
        const val LETTERS = "abcdefghijklmnopqrstuvwxyz"
        private val ROWS = arrayOf("qwertyuiop", "asdfghjkl", "zxcvbnm")
        private val ROW_SHIFT = floatArrayOf(0f, 0.5f, 1.5f)
        /** Canonical grid position of each letter a..z. */
        val KEY_U = FloatArray(26)
        val KEY_V = FloatArray(26)
        init {
            for ((r, row) in ROWS.withIndex()) for ((i, c) in row.withIndex()) {
                KEY_U[c - 'a'] = i + 0.5f + ROW_SHIFT[r]; KEY_V[c - 'a'] = r + 0.5f
            }
        }

        /** The letters at the ends of the QWERTY rows: their hit boxes may run to the keyboard edge. */
        private const val ROW_ENDS = "qpalzm"

        /**
         * Fits [keys]' letter keys to the QWERTY grid; null unless all of a..z exist and sit where QWERTY puts them.
         * The fit and the [tolerance] check (grid units) use the interior letters. The row-end letters only have to be
         * in their row and within [edgeTolerance] horizontally: the IME describes keys by their hit boxes, and a row's
         * outer keys extend to the screen edge (Gboard's `a` and `l`: centres 0.28 key from the drawn key), while a
         * different layout (QWERTZ, AZERTY) moves letters by whole keys and is still rejected.
         */
        fun fitQwerty(keys: KeySpace, tolerance: Float = 0.2f, edgeTolerance: Float = 0.45f): GridFit? {
            val xs = FloatArray(26); val ys = FloatArray(26)
            for (k in 0 until 26) {
                val i = keys.keyOf('a' + k)
                if (i < 0) return null
                xs[k] = keys.cx[i]; ys[k] = keys.cy[i]
            }
            val core = (0 until 26).filter { ('a' + it) !in ROW_ENDS }
            val cu = FloatArray(core.size) { KEY_U[core[it]] }; val cv = FloatArray(core.size) { KEY_V[core[it]] }
            val (ox, px) = line(cu, FloatArray(core.size) { xs[core[it]] })
            val (oy, py) = line(cv, FloatArray(core.size) { ys[core[it]] })
            if (px <= 0.0 || py <= 0.0) return null
            var worst = 0.0
            for (k in 0 until 26) {
                val du = (xs[k] - ox) / px - KEY_U[k]; val dv = (ys[k] - oy) / py - KEY_V[k]
                if (('a' + k) in ROW_ENDS) {
                    if (abs(du) > edgeTolerance || abs(dv) > tolerance) return null
                } else worst = maxOf(worst, sqrt(du * du + dv * dv))
            }
            if (worst > tolerance) return null
            return GridFit(ox.toFloat(), px.toFloat(), oy.toFloat(), py.toFloat(), worst.toFloat())
        }

        /** Least-squares y = a + b·x. */
        private fun line(x: FloatArray, y: FloatArray): Pair<Double, Double> {
            val n = x.size
            var mx = 0.0; var my = 0.0
            for (i in 0 until n) { mx += x[i]; my += y[i] }
            mx /= n; my /= n
            var sxx = 0.0; var sxy = 0.0
            for (i in 0 until n) { sxx += (x[i] - mx) * (x[i] - mx); sxy += (x[i] - mx) * (y[i] - my) }
            if (abs(sxx) < 1e-12) return 0.0 to 0.0
            val b = sxy / sxx
            return (my - b * mx) to b
        }

        /**
         * The path in grid units, time in ms from its first sample (kept non-decreasing). Every sample is kept, as in
         * training: a resting finger's repeated positions are what produce dwell frames.
         */
        fun toGrid(path: List<TouchPoint>, fit: GridFit, u: FloatArray, v: FloatArray, t: FloatArray): Int {
            var n = 0
            val t0 = if (path.isEmpty()) 0L else path[0].t
            for (p in path) {
                u[n] = fit.u(p.x); v[n] = fit.v(p.y)
                t[n] = maxOf((p.t - t0).toFloat(), if (n > 0) t[n - 1] else 0f)
                n++
            }
            return n
        }
    }
}
