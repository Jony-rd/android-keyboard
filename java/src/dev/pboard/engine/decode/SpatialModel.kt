package dev.pboard.engine.decode

import dev.pboard.engine.KeyboardGeometry
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Parameters of the per-key touch model, in units of the typical key width so they are resolution
 * independent. Defaults follow the thumb-typing literature (touch SD ≈ 0.25–0.4 key widths, touches land
 * slightly below the key centre).
 */
data class SpatialParams(
    val sigmaX: Float = 0.42f,
    val sigmaY: Float = 0.46f,
    /**
     * Expected downward offset of touches from the key centre (positive = towards the bottom). Device
     * dependent (Gboard/Pixel: slightly up; Nexus 4: down), so 0 by default and learned per user.
     */
    val biasY: Float = 0.0f,
    /** Cost cap (nats) for a tap nowhere near the intended key: a "cognitive" substitution. */
    val maxCost: Float = 8.5f,
)

/**
 * The typeable keys of a layout, derived purely from [KeyboardGeometry]: every key whose label is a
 * single word character (letters of any script, apostrophe…). Nothing here assumes QWERTY or Latin.
 */
class KeySpace(val geometry: KeyboardGeometry, rules: TextRules) {
    val keyWidth: Float = geometry.keyWidth.takeIf { it > 0f } ?: 1f
    val labels: CharArray
    val cx: FloatArray
    val cy: FloatArray
    private val byLabel = HashMap<Char, Int>()
    private val byAlternate = HashMap<Char, Int>()

    init {
        val l = ArrayList<Char>(); val x = ArrayList<Float>(); val y = ArrayList<Float>()
        for (k in geometry.keys) {
            val lab = rules.lowercase(k.label)
            if (lab.length != 1 || !rules.isWordChar(lab[0]) || byLabel.containsKey(lab[0])) continue
            val idx = l.size
            byLabel[lab[0]] = idx
            l.add(lab[0]); x.add(k.cx); y.add(k.cy)
            for (alt in k.alternates) {
                val a = rules.lowercase(alt)
                if (a.length == 1 && rules.isWordChar(a[0])) byAlternate.putIfAbsent(a[0], idx)
            }
        }
        labels = l.toCharArray(); cx = x.toFloatArray(); cy = y.toFloatArray()
    }

    val size: Int get() = labels.size

    /** Key index producing lower-case character [c] directly, or -1. */
    fun keyOf(c: Char): Int = byLabel[c] ?: -1

    /** Key whose long-press alternates include [c] ([dev.pboard.engine.KeyGeom.alternates]), or -1. */
    fun alternateKeyOf(c: Char): Int = byAlternate[c] ?: -1
}

/**
 * Learned per-key touch offsets (key-width units) for this user, added to key centres. Keyed by key
 * label so they survive geometry changes of the same layout.
 */
interface TouchOffsets {
    fun dx(label: Char): Float
    fun dy(label: Char): Float
    /** Bumped whenever the offsets change so cached spatial tables are rebuilt. */
    val version: Int
}

/**
 * Online estimate of systematic per-key touch offsets from taps of committed words (on-device only).
 * A shrunk, slowly decaying mean: a handful of taps cannot move a key far, posture changes are tracked.
 */
class AdaptiveTouchOffsets(private val maxOffset: Float = 0.35f, private val priorWeight: Float = 12f) : TouchOffsets {
    private class Acc { var sx = 0f; var sy = 0f; var n = 0f }
    private val acc = HashMap<Char, Acc>()
    override var version: Int = 0; private set

    override fun dx(label: Char): Float = acc[label]?.let { clamp(it.sx / (it.n + priorWeight)) } ?: 0f
    override fun dy(label: Char): Float = acc[label]?.let { clamp(it.sy / (it.n + priorWeight)) } ?: 0f

    /** A touch intended for [label] landed ([ox], [oy]) key widths from its modelled centre. */
    fun observe(label: Char, ox: Float, oy: Float) {
        if (ox * ox + oy * oy > 1.0f) return // more likely an error than a systematic offset
        val a = acc.getOrPut(label) { Acc() }
        val decay = 0.995f
        a.sx = a.sx * decay + ox; a.sy = a.sy * decay + oy; a.n = a.n * decay + 1f
        version++
    }

    fun reset() { acc.clear(); version++ }

    /** Accumulators per key label: (sum dx, sum dy, weight), for persistence. */
    fun snapshot(): Map<Char, FloatArray> = acc.mapValues { floatArrayOf(it.value.sx, it.value.sy, it.value.n) }

    fun restore(data: Map<Char, FloatArray>) {
        acc.clear()
        for ((k, v) in data) if (v.size == 3 && v.all { it.isFinite() }) acc[k] = Acc().also { it.sx = v[0]; it.sy = v[1]; it.n = v[2] }
        version++
    }

    private fun clamp(v: Float) = max(-maxOffset, min(maxOffset, v))
}

/**
 * Per-key 2-D Gaussian touch likelihood. For each touch it fills a cost vector over [KeySpace] keys: the
 * negative log-likelihood relative to the most likely key (best key costs 0), capped at
 * [SpatialParams.maxCost].
 */
class SpatialModel(val keys: KeySpace, val params: SpatialParams = SpatialParams(), private val offsets: TouchOffsets? = null) {
    val keyWidth: Float get() = keys.keyWidth
    private val meanX = FloatArray(keys.size)
    private val meanY = FloatArray(keys.size)
    private var offsetsVersion = -1
    private val invVarX = 1f / (2f * sq(params.sigmaX * keyWidth))
    private val invVarY = 1f / (2f * sq(params.sigmaY * keyWidth))

    init { refreshMeans() }

    private fun refreshMeans() {
        offsetsVersion = offsets?.version ?: 0
        for (i in 0 until keys.size) {
            val l = keys.labels[i]
            meanX[i] = keys.cx[i] + (offsets?.dx(l) ?: 0f) * keyWidth
            meanY[i] = keys.cy[i] + (params.biasY + (offsets?.dy(l) ?: 0f)) * keyWidth
        }
    }

    /** Fills [out] (size = key count) with per-key costs for a touch at ([x], [y]); returns the best key or -1. */
    fun costs(x: Float, y: Float, out: FloatArray): Int {
        if (offsets != null && offsets.version != offsetsVersion) refreshMeans()
        var best = Float.POSITIVE_INFINITY
        var bestI = -1
        for (i in 0 until keys.size) {
            val dx = x - meanX[i]; val dy = y - meanY[i]
            val c = dx * dx * invVarX + dy * dy * invVarY
            out[i] = c
            if (c < best) { best = c; bestI = i }
        }
        val cap = params.maxCost
        for (i in 0 until keys.size) out[i] = min(cap, out[i] - best)
        return bestI
    }

    /** Where an ideal touch on key [k] lands under this model (centre + bias + learned offset). */
    fun idealX(k: Int): Float { if (offsets != null && offsets.version != offsetsVersion) refreshMeans(); return meanX[k] }
    fun idealY(k: Int): Float { if (offsets != null && offsets.version != offsetsVersion) refreshMeans(); return meanY[k] }

    /** Offset of a touch from key [k]'s centre in key widths, net of the global bias (for [AdaptiveTouchOffsets]). */
    fun residual(k: Int, x: Float, y: Float): Pair<Float, Float> =
        (x - keys.cx[k]) / keyWidth to (y - keys.cy[k]) / keyWidth - params.biasY

    fun distanceToKey(k: Int, x: Float, y: Float): Float {
        val dx = x - keys.cx[k]; val dy = y - keys.cy[k]
        return sqrt(dx * dx + dy * dy) / keyWidth
    }

    private fun sq(v: Float) = v * v
}

/**
 * Maps a lexicon's characters to the keys of a layout: the key labelled with the character (cost 0),
 * and/or the key of its folded form ('é' on the 'e' key) at [foldPenalty]. Silent characters (apostrophe,
 * hyphen) may additionally be skipped by the decoders; characters with no key at all can only be reached
 * through a substitution.
 */
class CharKeyMap(val lexicon: Lexicon, val keys: KeySpace, foldPenalty: Float) {
    private val a = lexicon.alphabet.size
    /** Up to two (key, penalty) routes per symbol; key -1 = none. */
    val key1 = IntArray(a) { -1 }
    val pen1 = FloatArray(a)
    val key2 = IntArray(a) { -1 }
    val pen2 = FloatArray(a)
    val silent = BooleanArray(a)
    /** The key a gesture passes over for each symbol, or -1 (silent / unreachable). */
    val gestureKey = IntArray(a) { -1 }

    init {
        val rules = lexicon.rules
        for (s in 0 until a) {
            val c = lexicon.alphabet[s]
            silent[s] = rules.isSilent(c)
            val direct = keys.keyOf(c)
            val f = rules.fold(c)
            // Without a dedicated key, a character is typed on its base letter's key (folding) or on the key
            // that offers it as a long-press alternate; both cost the fold penalty when tapped plainly.
            val folded = (if (f != null) keys.keyOf(f) else -1).takeIf { it >= 0 } ?: keys.alternateKeyOf(c)
            if (direct >= 0) { key1[s] = direct; pen1[s] = 0f; if (folded >= 0 && folded != direct) { key2[s] = folded; pen2[s] = foldPenalty } }
            else if (folded >= 0) { key1[s] = folded; pen1[s] = foldPenalty }
            gestureKey[s] = if (silent[s]) -1 else if (direct >= 0) direct else folded
        }
    }

    /** Per-symbol costs from per-key costs; unreachable symbols cost [unreachable]. */
    fun symbolCosts(keyCosts: FloatArray, out: FloatArray, unreachable: Float, cap: Float) {
        for (s in 0 until a) {
            var c = unreachable
            val k1 = key1[s]
            if (k1 >= 0) c = keyCosts[k1] + pen1[s]
            val k2 = key2[s]
            if (k2 >= 0) c = min(c, keyCosts[k2] + pen2[s])
            out[s] = min(c, cap)
        }
    }
}
