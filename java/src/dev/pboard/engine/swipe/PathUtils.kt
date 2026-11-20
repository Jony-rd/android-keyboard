package dev.pboard.engine.swipe

import kotlin.math.max
import kotlin.math.sqrt

/** Geometry helpers for gesture paths. Points are stored as separate x/y float arrays. */
object PathUtils {

    fun length(xs: FloatArray, ys: FloatArray, n: Int = xs.size): Float {
        var l = 0f
        for (i in 1 until n) l += dist(xs[i - 1], ys[i - 1], xs[i], ys[i])
        return l
    }

    fun dist(x0: Float, y0: Float, x1: Float, y1: Float): Float {
        val dx = x1 - x0; val dy = y1 - y0
        return sqrt(dx * dx + dy * dy)
    }

    /**
     * Resamples the polyline ([xs], [ys]) (first [n] points) to [count] points equally spaced along its
     * arc length, writing into [outX]/[outY]. A zero-length polyline yields [count] copies of its first point.
     */
    fun resample(xs: FloatArray, ys: FloatArray, n: Int, count: Int, outX: FloatArray, outY: FloatArray) {
        if (n == 0) return
        val total = length(xs, ys, n)
        if (n == 1 || total <= 1e-6f) {
            for (k in 0 until count) { outX[k] = xs[0]; outY[k] = ys[0] }
            return
        }
        val step = total / (count - 1)
        outX[0] = xs[0]; outY[0] = ys[0]
        var seg = 1
        var segStart = 0f // arc length at point seg-1
        var segLen = dist(xs[0], ys[0], xs[1], ys[1])
        for (k in 1 until count - 1) {
            val target = k * step
            while (seg < n - 1 && segStart + segLen < target) {
                segStart += segLen
                seg++
                segLen = dist(xs[seg - 1], ys[seg - 1], xs[seg], ys[seg])
            }
            val t = if (segLen > 1e-6f) ((target - segStart) / segLen).coerceIn(0f, 1f) else 0f
            outX[k] = xs[seg - 1] + (xs[seg] - xs[seg - 1]) * t
            outY[k] = ys[seg - 1] + (ys[seg] - ys[seg - 1]) * t
        }
        outX[count - 1] = xs[n - 1]; outY[count - 1] = ys[n - 1]
    }

    /**
     * Shape normalisation (SHARK²): translate the centroid to the origin and scale so the larger bounding
     * box side is 1. Returns the scale factor applied (0 for a degenerate path).
     */
    fun normalise(xs: FloatArray, ys: FloatArray, count: Int, outX: FloatArray, outY: FloatArray): Float {
        var cx = 0f; var cy = 0f
        var minX = Float.POSITIVE_INFINITY; var maxX = Float.NEGATIVE_INFINITY
        var minY = Float.POSITIVE_INFINITY; var maxY = Float.NEGATIVE_INFINITY
        for (i in 0 until count) {
            cx += xs[i]; cy += ys[i]
            if (xs[i] < minX) minX = xs[i]; if (xs[i] > maxX) maxX = xs[i]
            if (ys[i] < minY) minY = ys[i]; if (ys[i] > maxY) maxY = ys[i]
        }
        cx /= count; cy /= count
        val side = max(maxX - minX, maxY - minY)
        val s = if (side > 1e-6f) 1f / side else 0f
        for (i in 0 until count) { outX[i] = (xs[i] - cx) * s; outY[i] = (ys[i] - cy) * s }
        return s
    }

    /** Mean squared distance between two equally sampled point sequences. */
    fun meanSquaredDistance(ax: FloatArray, ay: FloatArray, bx: FloatArray, by: FloatArray, count: Int): Float {
        var s = 0f
        for (i in 0 until count) {
            val dx = ax[i] - bx[i]; val dy = ay[i] - by[i]
            s += dx * dx + dy * dy
        }
        return s / count
    }
}
