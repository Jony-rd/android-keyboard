package dev.pboard.engine.lm

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.tanh

/**
 * Word-level LSTM language model ("PBNN" `.nnlm`, see docs/lm-format.md), hand-written for the JVM/ART:
 * int8 weights with per-row fp32 scales, fp32 activations, no runtime dependency.
 *
 * Vocabulary = the base pack's n-gram ids below [nv] (classes such as {color, colour}); larger ids are
 * `<rare>`. Output: adaptive softmax. The head (shortlist + one logit per tail cluster) is evaluated eagerly
 * per state; tail clusters only when a word inside them is scored.
 *
 * Cost per [step]: 4·hid·(emb+hid) + hid·emb multiply-adds (≈ 1.5M for the shipped 512/192 model); per
 * [Output]: (shortlist + clusters)·emb (≈ 0.4M). [State] and [Output] are immutable and can be cached.
 */
class NeuralLm private constructor(buf: ByteBuffer, threads: Int) {
    /** Optional helper thread: the large matrix-vector products are split in two row halves. */
    private val helper: java.util.concurrent.ExecutorService? = if (threads >= 2)
        java.util.concurrent.Executors.newSingleThreadExecutor { r -> Thread(r, "pboard-nnlm").apply { isDaemon = true } }
    else null

    val nv: Int
    val emb: Int
    val hid: Int
    val baseBuildId: Long
    /** Interpolation weight tuned on dev data when the model was exported (see [NeuralRescoringLm]). */
    val recommendedAlpha: Float
    /** cutoffs[0] = shortlist size; cluster i covers ids [cutoffs[i], cutoffs[i+1]). */
    val cutoffs: IntArray

    private class Mat(val rows: Int, val cols: Int, val q: ByteArray, val scale: FloatArray)

    private val embM: Mat
    private val wih: Mat
    private val whh: Mat
    private val bias: FloatArray
    private val projW: Mat
    private val projB: FloatArray
    private val head: Mat
    private val tailProj: Array<Mat>
    private val tailOut: Array<Mat>

    init {
        buf.order(ByteOrder.LITTLE_ENDIAN)
        val magic = ByteArray(4) { buf.get(it) }
        require(String(magic, Charsets.US_ASCII) == "PBNN") { "not a neural LM" }
        require((buf.getShort(4).toInt() and 0xFFFF) == 1) { "unsupported .nnlm version" }
        nv = buf.getInt(12)
        emb = buf.getInt(16)
        hid = buf.getInt(20)
        val nClusters = buf.getInt(24)
        cutoffs = IntArray(nClusters + 1) { buf.getInt(28 + 4 * it) }
        baseBuildId = buf.getLong(64)
        recommendedAlpha = buf.getFloat(72).takeIf { it > 0f && it < 1f } ?: 0.5f
        val mats = HashMap<Int, Mat>()
        val vecs = HashMap<Int, FloatArray>()
        for (i in 0 until buf.getInt(124)) {
            val p = 128 + 24 * i
            val id = buf.getInt(p)
            val kind = buf.getInt(p + 4)
            val rows = buf.getInt(p + 8)
            val cols = buf.getInt(p + 12)
            val off = buf.getLong(p + 16).toInt()
            if (kind == 1) {
                val q = ByteArray(rows * cols)
                buf.position(off)
                buf.get(q)
                val sOff = off + ((rows * cols + 3) and 3.inv())
                mats[id] = Mat(rows, cols, q, FloatArray(rows) { buf.getFloat(sOff + 4 * it) })
            } else {
                vecs[id] = FloatArray(cols) { buf.getFloat(off + 4 * it) }
            }
        }
        embM = mats.getValue(1); wih = mats.getValue(2); whh = mats.getValue(3); bias = vecs.getValue(4)
        projW = mats.getValue(5); projB = vecs.getValue(6); head = mats.getValue(7)
        tailProj = Array(nClusters) { mats.getValue(10 + 2 * it) }
        tailOut = Array(nClusters) { mats.getValue(11 + 2 * it) }
        require(wih.rows == 4 * hid && whh.cols == hid && head.rows == cutoffs[0] + nClusters)
    }

    /** Recurrent state (h, c) after a token sequence. */
    class State internal constructor(internal val h: FloatArray, internal val c: FloatArray)

    fun initialState(): State = State(FloatArray(hid), FloatArray(hid))

    /** Feeds n-gram id [token] (ids >= [nv] are `<rare>`). */
    fun step(s: State, token: Int): State {
        val tok = if (token in 0 until nv) token else LmPack.RARE
        val x = FloatArray(emb)
        val sc = embM.scale[tok]
        val base = tok * emb
        for (j in 0 until emb) x[j] = embM.q[base + j] * sc
        val g = FloatArray(4 * hid)
        split(4 * hid) { r0, r1 ->
            matVec(wih, x, g, false, r0, r1)
            matVec(whh, s.h, g, true, r0, r1)
        }
        val h = FloatArray(hid)
        val c = FloatArray(hid)
        for (j in 0 until hid) {
            val ig = sigmoid(g[j] + bias[j])
            val fg = sigmoid(g[hid + j] + bias[hid + j])
            val gg = tanh(g[2 * hid + j] + bias[2 * hid + j])
            val og = sigmoid(g[3 * hid + j] + bias[3 * hid + j])
            val cj = fg * s.c[j] + ig * gg
            c[j] = cj
            h[j] = og * tanh(cj)
        }
        return State(h, c)
    }

    /** Next-token distribution of a state. Thread-safe for reads; tail clusters are computed on first use. */
    inner class Output internal constructor(s: State) {
        private val e = FloatArray(emb)
        /** log-softmax of the head: shortlist ids, then one entry per tail cluster. */
        internal val headLp: FloatArray
        /** Tail-cluster log-softmaxes, computed on first use; atomic so they publish safely across threads. */
        private val tailLp = java.util.concurrent.atomic.AtomicReferenceArray<FloatArray?>(cutoffs.size - 1)

        init {
            matVec(projW, s.h, e, false)
            for (j in 0 until emb) e[j] += projB[j]
            val z = FloatArray(head.rows)
            split(head.rows) { r0, r1 -> matVec(head, e, z, false, r0, r1) }
            headLp = logSoftmax(z)
        }

        /** Natural-log probability of n-gram id [id] (must be < [nv]). */
        fun logProb(id: Int): Float {
            if (id < cutoffs[0]) return headLp[id]
            var k = 0
            while (id >= cutoffs[k + 1]) k++
            val t = tail(k)
            return t[id - cutoffs[k]] + headLp[cutoffs[0] + k]
        }

        private fun tail(k: Int): FloatArray {
            tailLp.get(k)?.let { return it }
            synchronized(this) {
                tailLp.get(k)?.let { return it }
                val p = FloatArray(tailProj[k].rows)
                matVec(tailProj[k], e, p, false)
                val z = FloatArray(tailOut[k].rows)
                matVec(tailOut[k], p, z, false)
                return logSoftmax(z).also { tailLp.set(k, it) }
            }
        }

        /** The [k] most likely shortlist ids that pass [accept], best first. */
        fun topShortlist(k: Int, accept: (Int) -> Boolean): IntArray {
            val ids = IntArray(k) { -1 }
            val sc = FloatArray(k) { Float.NEGATIVE_INFINITY }
            for (id in 0 until cutoffs[0]) {
                val v = headLp[id]
                if (v <= sc[k - 1] || !accept(id)) continue
                var i = k - 1
                while (i > 0 && sc[i - 1] < v) { sc[i] = sc[i - 1]; ids[i] = ids[i - 1]; i-- }
                sc[i] = v; ids[i] = id
            }
            return ids.filter { it >= 0 }.toIntArray()
        }
    }

    fun output(s: State): Output = Output(s)

    /** Runs [body] over rows [0, rows): the upper half on the helper thread when there is one. */
    private fun split(rows: Int, body: (Int, Int) -> Unit) {
        val h = helper
        if (h == null) { body(0, rows); return }
        val mid = rows / 2
        val f = h.submit(Runnable { body(mid, rows) })
        body(0, mid)
        f.get()
    }

    companion object {
        /**
         * [threads] = 2 splits each step across the caller and one helper thread. Keep the default of 1: on the
         * dev phone (Snapdragon 732G) the split cut the bare step from 3.1 to 2.3 ms but made `predict` 3-4x
         * slower (next-word p99 4.5 -> 21 ms), most likely because the caller, blocking on the helper, is moved
         * to a little core.
         */
        fun read(bytes: ByteArray, threads: Int = 1): NeuralLm = NeuralLm(ByteBuffer.wrap(bytes), threads)
        fun read(file: File, threads: Int = 1): NeuralLm = read(file.readBytes(), threads)

        private fun sigmoid(v: Float): Float = 1f / (1f + exp(-v))

        private fun logSoftmax(z: FloatArray): FloatArray {
            var m = Float.NEGATIVE_INFINITY
            for (v in z) if (v > m) m = v
            var s = 0.0
            for (v in z) s += exp((v - m).toDouble())
            val lz = m + ln(s).toFloat()
            return FloatArray(z.size) { z[it] - lz }
        }

        /** out[r] (+)= scale[r] · Σ_c q[r,c]·x[c]; 4 independent accumulators for instruction-level parallelism. */
        private fun matVec(m: Mat, x: FloatArray, out: FloatArray, accumulate: Boolean, r0: Int = 0, r1: Int = m.rows) {
            val q = m.q
            val cols = m.cols
            val c4 = cols and 3.inv()
            for (r in r0 until r1) {
                var a0 = 0f; var a1 = 0f; var a2 = 0f; var a3 = 0f
                var i = r * cols
                var j = 0
                while (j < c4) {
                    a0 += q[i] * x[j]
                    a1 += q[i + 1] * x[j + 1]
                    a2 += q[i + 2] * x[j + 2]
                    a3 += q[i + 3] * x[j + 3]
                    i += 4; j += 4
                }
                while (j < cols) { a0 += q[i] * x[j]; i++; j++ }
                val v = (a0 + a1 + a2 + a3) * m.scale[r]
                if (accumulate) out[r] += v else out[r] = v
            }
        }
    }
}
