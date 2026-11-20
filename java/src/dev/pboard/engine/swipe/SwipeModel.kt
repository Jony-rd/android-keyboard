package dev.pboard.engine.swipe

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * The neural gesture model (`.pbsw`, written by `tools/swipe/export.py` / `export2.py`; see docs/swipe-neural.md).
 *
 * Version 1 (QWERTY): touch frames ([SwipeFeatures]) → per-frame letter log-posteriors (class 0 = CTC blank,
 * 1 + i = letter 'a' + i), from a linear output layer.
 * Version 2 (any layout): touch frames ([SwipeFeatures2]) → per frame an intention gate g and [nBasis]² spatial
 * coefficients c; with the layout's keys given at run time ([Encoder.setKeys]), class 0 = blank with probability
 * 1 − σ(g) and class 1 + k = key k with σ(g) · softmax_k(c · φ(key k)), φ(x, y)[a·N + b] = cos(πax)·cos(πby).
 *
 * Network: strided 1-D convolution + ReLU over the frame features (halves the frame rate), [layers] bidirectional
 * LSTMs of [hid] units per direction, then the output layer. Hand-written for the JVM/ART (no runtime
 * dependency): fp16 weights in the file, fp32 in memory and arithmetic, blocked matrix products for everything that
 * does not depend on the recurrence.
 *
 * Not thread-safe: one [Encoder] per decoding thread ([newEncoder]).
 */
class SwipeModel private constructor(buf: ByteBuffer) {
    val nFeat: Int
    val conv: Int
    val kernel: Int
    val stride: Int
    val hid: Int
    val layers: Int
    /** File format: 1 = fixed QWERTY letter classes, 2 = layout-agnostic spatial head. */
    val version: Int
    /** Version 1: output classes (27). Version 2: 0 (1 + the number of keys given to [Encoder.setKeys]). */
    val nClasses: Int
    /** Version 2: cosine basis frequencies per axis. */
    val nBasis: Int
    /** Version 1 feature extraction (version 2 uses [SwipeFeatures2] built per layout with [stepS]/[stepT]/[maxFrames]). */
    val features: SwipeFeatures
    val stepS: Float
    val stepT: Float
    val maxFrames: Int
    /**
     * Version 2: an optional second network stored in the same file (same features), trained with layout
     * co-augmentation; [NeuralSwipeDecoder] uses it to score very short words (see [SwipeParams.neuralAuxKeys]).
     */
    val aux: SwipeModel?
    /** Training metadata (JSON) stored with the weights. */
    val metadata: String

    private val convW: FloatArray // [conv][kernel·nFeat], frame-major windows
    private val convB: FloatArray
    private val wih: Array<FloatArray> // [layer·2 + dir] → [4·hid × in]
    private val whh: Array<FloatArray> // [4·hid × hid]
    private val bias: Array<FloatArray> // [4·hid]
    private val outW: FloatArray // [nClasses × 2·hid]
    private val outB: FloatArray

    init {
        buf.order(ByteOrder.LITTLE_ENDIAN)
        val magic = ByteArray(4) { buf.get(it) }
        require(String(magic, Charsets.US_ASCII) == "PBSW") { "not a swipe model" }
        version = buf.getShort(4).toInt()
        require(version == 1 || version == 2) { "unsupported .pbsw version" }
        require(buf.getShort(6).toInt() == 1) { "unsupported weight type" }
        nFeat = buf.getInt(8); conv = buf.getInt(12); kernel = buf.getInt(16); stride = buf.getInt(20)
        hid = buf.getInt(24); layers = buf.getInt(28)
        stepS = buf.getFloat(36); stepT = buf.getFloat(40); maxFrames = buf.getInt(44)
        if (version == 1) {
            nClasses = buf.getInt(32); nBasis = 0
            features = SwipeFeatures(stepS, stepT, maxFrames, buf.getFloat(48))
            require(nFeat == SwipeFeatures.N_FEAT && nClasses == SwipeFeatures.N_CLASSES) { "model does not match the feature set" }
        } else {
            nClasses = 0; nBasis = buf.getInt(32)
            features = SwipeFeatures()
            require(nFeat == SwipeFeatures2.N_FEAT && nBasis in 1..32) { "model does not match the feature set" }
        }
        val metaLen = buf.getInt(52)
        val mb = ByteArray(metaLen)
        buf.position(64); buf.get(mb)
        metadata = String(mb, Charsets.UTF_8)
        buf.position((64 + metaLen + 3) and 3.inv())
        fun read(n: Int) = FloatArray(n) { halfToFloat(buf.getShort().toInt()) }
        val cw = read(conv * nFeat * kernel) // torch layout [conv][nFeat][kernel]
        convW = FloatArray(cw.size)
        for (o in 0 until conv) for (f in 0 until nFeat) for (k in 0 until kernel) convW[o * kernel * nFeat + k * nFeat + f] = cw[(o * nFeat + f) * kernel + k]
        convB = read(conv)
        wih = Array(layers * 2) { FloatArray(0) }; whh = Array(layers * 2) { FloatArray(0) }; bias = Array(layers * 2) { FloatArray(0) }
        for (l in 0 until layers) for (d in 0 until 2) {
            val inp = if (l == 0) conv else 2 * hid
            wih[l * 2 + d] = read(4 * hid * inp); whh[l * 2 + d] = read(4 * hid * hid); bias[l * 2 + d] = read(4 * hid)
        }
        val outRows = if (version == 1) nClasses else nBasis * nBasis + 1
        outW = read(outRows * 2 * hid); outB = read(outRows)
        val auxOff = if (version == 2) buf.getInt(48) else 0
        aux = if (auxOff > 0) SwipeModel(buf.duplicate().also { (it as java.nio.Buffer).position(auxOff) }.slice()).also {
            require(it.version == 2 && it.stepS == stepS && it.stepT == stepT && it.maxFrames == maxFrames) { "auxiliary model must share the features" }
        } else null
    }

    /** Rows of the output layer: classes (v1) or basis coefficients + gate (v2). */
    private val outRows: Int get() = if (version == 1) nClasses else nBasis * nBasis + 1

    /** Multiply-adds of one encode for [frames] input frames (for latency budgeting). */
    fun macs(frames: Int): Long {
        val t = ((frames + stride - 1) / stride).toLong()
        var m = t * conv * kernel * nFeat
        for (l in 0 until layers) m += 2 * t * 4 * hid * ((if (l == 0) conv else 2 * hid) + hid)
        return m + t * outRows * 2 * hid
    }

    fun newEncoder(): Encoder = Encoder()

    /** Per-thread scratch space and the forward pass. */
    inner class Encoder {
        private var cap = 0
        private var xpad = FloatArray(0)
        private var a = FloatArray(0) // layer input [T' × max(conv, 2·hid)]
        private var b = FloatArray(0) // layer output [T' × 2·hid]
        private var g = FloatArray(0) // input projections [T' × 4·hid]
        private val gates = FloatArray(4 * hid)
        private val h = FloatArray(hid)
        private val c = FloatArray(hid)
        // Version 2: the layout's basis [keys × nBasis²] and scratch for the head output and key scores.
        private var phi = FloatArray(0)
        private var keys = 0
        private var zeroBias = FloatArray(0)
        private var head = FloatArray(0)
        private var z = FloatArray(0)

        /** Output classes per frame: v1 [nClasses], v2 1 + the number of keys set with [setKeys]. */
        val classes: Int get() = if (version == 1) nClasses else 1 + keys

        /** Version 2: the layout's key centres in the feature frame ([SwipeFeatures2.keyX]/[keyY], [0, 1]). */
        fun setKeys(kx: FloatArray, ky: FloatArray) {
            check(version == 2)
            val nb = nBasis; val c2 = nb * nb
            keys = kx.size
            phi = FloatArray(keys * c2)
            for (k in 0 until keys) for (a in 0 until nb) for (b in 0 until nb)
                phi[k * c2 + a * nb + b] = (kotlin.math.cos(Math.PI * a * kx[k]) * kotlin.math.cos(Math.PI * b * ky[k])).toFloat()
            zeroBias = FloatArray(keys)
        }

        private fun ensure(tIn: Int) {
            val t = (tIn + stride - 1) / stride
            if (t <= cap && xpad.size >= (tIn + kernel + stride) * nFeat) return
            cap = maxOf(t, 16)
            xpad = FloatArray((cap * stride + kernel + stride) * nFeat)
            a = FloatArray(cap * maxOf(conv, 2 * hid)); b = FloatArray(cap * 2 * hid); g = FloatArray(cap * 4 * hid)
        }

        /**
         * Log-posteriors for [n] feature frames [x] (row-major [n × nFeat]) into [lastOutput] (T' × nClasses).
         * Returns the output frame count T', or -1 when [cancelled] reported (between layer passes) that the result
         * is no longer wanted.
         */
        var lastOutput = FloatArray(0)
            private set

        fun encode(x: FloatArray, n: Int, cancelled: (() -> Boolean)? = null): Int {
            ensure(n)
            val t = (n + stride - 1) / stride
            val left = (kernel - stride) / 2
            // Zero-padded input: `left` frames before, enough after for the last window.
            java.util.Arrays.fill(xpad, 0, ((t - 1) * stride + kernel) * nFeat, 0f)
            System.arraycopy(x, 0, xpad, left * nFeat, n * nFeat)
            // Convolution as a matrix product over overlapping windows (stride·nFeat apart), then ReLU.
            gemm(xpad, 0, stride * nFeat, t, convW, conv, kernel * nFeat, convB, a, conv)
            for (i in 0 until t * conv) if (a[i] < 0f) a[i] = 0f
            var inp = conv
            var src = a; var dst = b
            for (l in 0 until layers) {
                for (d in 0 until 2) {
                    val k = l * 2 + d
                    if (cancelled?.invoke() == true) return -1
                    gemm(src, 0, inp, t, wih[k], 4 * hid, inp, bias[k], g, 4 * hid)
                    recur(t, whh[k], dst, d)
                }
                inp = 2 * hid
                val tmp = src; src = dst; dst = tmp
            }
            if (version == 1) {
                if (lastOutput.size < t * nClasses) lastOutput = FloatArray(t * nClasses)
                val out = lastOutput
                gemm(src, 0, 2 * hid, t, outW, nClasses, 2 * hid, outB, out, nClasses)
                for (i in 0 until t) logSoftmax(out, i * nClasses, nClasses)
                return t
            }
            check(keys > 0) { "setKeys first" }
            val rows = outRows; val c2 = nBasis * nBasis; val nc = 1 + keys
            if (head.size < t * rows) head = FloatArray(t * rows)
            if (z.size < t * keys) z = FloatArray(t * keys)
            if (lastOutput.size < t * nc) lastOutput = FloatArray(t * nc)
            gemm(src, 0, 2 * hid, t, outW, rows, 2 * hid, outB, head, rows)
            gemm(head, 0, rows, t, phi, keys, c2, zeroBias, z, keys)
            val out = lastOutput
            for (i in 0 until t) {
                val gt = head[i * rows + c2]
                // log σ(g) and log σ(−g), numerically stable.
                val lsp = -softplus(-gt); val lsn = -softplus(gt)
                logSoftmax(z, i * keys, keys)
                out[i * nc] = lsn
                for (k in 0 until keys) out[i * nc + 1 + k] = lsp + z[i * keys + k]
            }
            return t
        }

        /** One LSTM direction over [t] steps, input projections in [g]; writes h into [y] columns [dir·hid, …). */
        private fun recur(t: Int, w: FloatArray, y: FloatArray, dir: Int) {
            java.util.Arrays.fill(h, 0f); java.util.Arrays.fill(c, 0f)
            val h4 = 4 * hid
            for (s in 0 until t) {
                val step = if (dir == 0) s else t - 1 - s
                val go = step * h4
                for (r in 0 until h4) {
                    val ro = r * hid
                    var a0 = 0f; var a1 = 0f; var a2 = 0f; var a3 = 0f
                    var j = 0
                    while (j + 3 < hid) {
                        a0 += w[ro + j] * h[j]; a1 += w[ro + j + 1] * h[j + 1]
                        a2 += w[ro + j + 2] * h[j + 2]; a3 += w[ro + j + 3] * h[j + 3]
                        j += 4
                    }
                    while (j < hid) { a0 += w[ro + j] * h[j]; j++ }
                    gates[r] = g[go + r] + (a0 + a1) + (a2 + a3)
                }
                val yo = step * 2 * hid + dir * hid
                for (j in 0 until hid) {
                    val ig = sigmoid(gates[j]); val fg = sigmoid(gates[hid + j])
                    val gg = tanh(gates[2 * hid + j]); val og = sigmoid(gates[3 * hid + j])
                    val cj = fg * c[j] + ig * gg
                    c[j] = cj
                    val hj = og * tanh(cj)
                    h[j] = hj
                    y[yo + j] = hj
                }
            }
        }
    }

    companion object {
        fun read(bytes: ByteArray): SwipeModel = SwipeModel(ByteBuffer.wrap(bytes))
        fun read(file: File): SwipeModel = read(file.readBytes())

        /** Resource path of the model bundled with the engine. */
        const val BUNDLED = "/dev/pboard/engine/swipe/qwerty.pbsw"

        /** The model bundled with the engine (null if this build has none). */
        fun bundled(): SwipeModel? = SwipeModel::class.java.getResourceAsStream(BUNDLED)?.use { read(it.readBytes()) }

        /**
         * C[i, r] = bias[r] + Σ_c W[r, c] · A[aOff + i·aStride + c] for i < n, r < rows (C row stride [cStride]).
         * Blocked 4 inputs × 2 rows: six loads per eight multiply-adds.
         */
        internal fun gemm(A: FloatArray, aOff: Int, aStride: Int, n: Int, W: FloatArray, rows: Int, cols: Int, bias: FloatArray, C: FloatArray, cStride: Int) {
            var i = 0
            while (i + 3 < n) {
                val p0 = aOff + i * aStride; val p1 = p0 + aStride; val p2 = p1 + aStride; val p3 = p2 + aStride
                var r = 0
                while (r + 1 < rows) {
                    val w0 = r * cols; val w1 = w0 + cols
                    var s00 = 0f; var s01 = 0f; var s02 = 0f; var s03 = 0f
                    var s10 = 0f; var s11 = 0f; var s12 = 0f; var s13 = 0f
                    for (c in 0 until cols) {
                        val x0 = A[p0 + c]; val x1 = A[p1 + c]; val x2 = A[p2 + c]; val x3 = A[p3 + c]
                        val u = W[w0 + c]; val v = W[w1 + c]
                        s00 += u * x0; s01 += u * x1; s02 += u * x2; s03 += u * x3
                        s10 += v * x0; s11 += v * x1; s12 += v * x2; s13 += v * x3
                    }
                    val b0 = bias[r]; val b1 = bias[r + 1]
                    C[i * cStride + r] = s00 + b0; C[(i + 1) * cStride + r] = s01 + b0
                    C[(i + 2) * cStride + r] = s02 + b0; C[(i + 3) * cStride + r] = s03 + b0
                    C[i * cStride + r + 1] = s10 + b1; C[(i + 1) * cStride + r + 1] = s11 + b1
                    C[(i + 2) * cStride + r + 1] = s12 + b1; C[(i + 3) * cStride + r + 1] = s13 + b1
                    r += 2
                }
                while (r < rows) {
                    val w0 = r * cols
                    var s0 = 0f; var s1 = 0f; var s2 = 0f; var s3 = 0f
                    for (c in 0 until cols) {
                        val u = W[w0 + c]
                        s0 += u * A[p0 + c]; s1 += u * A[p1 + c]; s2 += u * A[p2 + c]; s3 += u * A[p3 + c]
                    }
                    C[i * cStride + r] = s0 + bias[r]; C[(i + 1) * cStride + r] = s1 + bias[r]
                    C[(i + 2) * cStride + r] = s2 + bias[r]; C[(i + 3) * cStride + r] = s3 + bias[r]
                    r++
                }
                i += 4
            }
            while (i < n) {
                val p0 = aOff + i * aStride
                for (r in 0 until rows) {
                    val w0 = r * cols
                    var s0 = 0f; var s1 = 0f
                    var c = 0
                    while (c + 1 < cols) { s0 += W[w0 + c] * A[p0 + c]; s1 += W[w0 + c + 1] * A[p0 + c + 1]; c += 2 }
                    if (c < cols) s0 += W[w0 + c] * A[p0 + c]
                    C[i * cStride + r] = s0 + s1 + bias[r]
                }
                i++
            }
        }

        private fun softplus(x: Float): Float = if (x > 20f) x else kotlin.math.ln(1.0 + kotlin.math.exp(x.toDouble())).toFloat()

        private fun logSoftmax(z: FloatArray, off: Int, n: Int) {
            var m = Float.NEGATIVE_INFINITY
            for (k in off until off + n) if (z[k] > m) m = z[k]
            var s = 0.0
            for (k in off until off + n) s += kotlin.math.exp((z[k] - m).toDouble())
            val lz = m + kotlin.math.ln(s).toFloat()
            for (k in off until off + n) z[k] -= lz
        }

        // tanh by table + linear interpolation on [-TANH_MAX, TANH_MAX] (abs error < 3e-5); sigmoid via tanh.
        private const val TANH_MAX = 8f
        private const val TANH_RES = 64
        private val TANH = FloatArray((2 * TANH_MAX * TANH_RES).toInt() + 2) { kotlin.math.tanh(it.toDouble() / TANH_RES - TANH_MAX).toFloat() }

        internal fun tanh(x: Float): Float {
            if (x >= TANH_MAX) return 1f
            if (x <= -TANH_MAX) return -1f
            val p = (x + TANH_MAX) * TANH_RES
            val i = p.toInt()
            val f = p - i
            val a = TANH[i]
            return a + (TANH[i + 1] - a) * f
        }

        internal fun sigmoid(x: Float): Float = 0.5f + 0.5f * tanh(0.5f * x)

        /** IEEE 754 half → float. */
        internal fun halfToFloat(hbits: Int): Float {
            val s = (hbits ushr 15) and 1
            val e = (hbits ushr 10) and 0x1f
            val m = hbits and 0x3ff
            val bits = when {
                e == 0 -> if (m == 0) s shl 31 else {
                    var mm = m; var ee = -14
                    while (mm and 0x400 == 0) { mm = mm shl 1; ee-- }
                    (s shl 31) or ((ee + 127) shl 23) or ((mm and 0x3ff) shl 13)
                }
                e == 31 -> (s shl 31) or (0xff shl 23) or (m shl 13)
                else -> (s shl 31) or ((e - 15 + 127) shl 23) or (m shl 13)
            }
            return java.lang.Float.intBitsToFloat(bits)
        }
    }
}
