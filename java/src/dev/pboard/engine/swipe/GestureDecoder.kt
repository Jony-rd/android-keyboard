package dev.pboard.engine.swipe

import dev.pboard.engine.LanguageModel
import dev.pboard.engine.TouchPoint
import dev.pboard.engine.decode.CharKeyMap
import dev.pboard.engine.decode.Lexicon
import dev.pboard.engine.decode.WordHypothesis
import java.util.PriorityQueue
import kotlin.math.abs
import kotlin.math.min

/**
 * Gesture decoder tunables (one place, so they can be re-fit on human traces). Channel costs are
 * nats-like and add to −[lmWeight]·ln P(word | context). Distances are in key widths.
 *
 * The channel weights below were fitted by separable CMA-ES on the human TUNE swipes (FUTO + HWS, production
 * LM; `./gradlew :engine:tune -Ptune.args="--mode swipe ..."`): two runs, 1,346 candidates. Compared with the
 * hand-set weights the fit relies far less on the scale-free shape channel (6 -> 0.34) and the LM (1.0 -> 0.74),
 * and far more on where the gesture starts and ends.
 */
data class SwipeParams(
    /** Equidistant resampling resolution for the location/shape channels. */
    val samples: Int = 32,
    val lmWeight: Float = 0.7361f,
    /** Weight of the mean squared point-to-point distance between path and word template (location channel). */
    val locationWeight: Float = 3.327f,
    /** Weight of the shape channel (translation/scale-invariant mean squared distance, scaled back to the path size). */
    val shapeWeight: Float = 0.3418f,
    /** Weight of the ordered key-alignment channel: each letter's closest in-order approach of the path. */
    val keyWeight: Float = 1.332f,
    /** Weight of the squared distance from the gesture's first point to the first letter's key. */
    val startWeight: Float = 2.63f,
    /**
     * Same for the last point and the last key. Human end points scatter about twice as much as start
     * points (so ~1/4 of the weight) and overshoot along the last segment.
     */
    val endWeight: Float = 2.065f,
    /** Overshoot past the last key along the final segment that costs nothing (key widths). */
    val endOvershoot: Float = 0.7474f,
    /**
     * First/last keys considered must lie within these distances of the gesture's first/last point. Soft
     * likelihoods decide among them: ~30 % of human swipes start or end nearest a different key.
     */
    val startRadius: Float = 2.0f,
    val endRadius: Float = 2.8f,
    /** Template length vs. path length gate: |Lt − Lp| ≤ ratio·Lp + slack (key widths). */
    val lengthRatio: Float = 0.6f,
    val lengthSlack: Float = 2.0f,
    /** Candidates kept after the cheap location pass, then rescored with every channel + context LM. */
    val coarseKeep: Int = 80,
    /** Weight of the unigram prior in the coarse pass. */
    val coarsePriorWeight: Float = 0.6512f,
    /** Paths shorter than this (key widths) are treated as taps. */
    val minGestureLength: Float = 0.6f,
    /**
     * Glides at most this long (key widths) over 2-5 keys decode to the keys they cover or the word those spell
     * ("ty" for a drag from t to the edge of y, as Gboard 18.6), unless the decoder's best word fits the same keys.
     */
    val shortGlideMax: Float = 2.2f,
    /**
     * The decoder's best word of two or more letters stays first for a short glide even when the keys disagree. Human
     * TUNE (FUTO / HWS 1-3 letter words, top-1): off 95.91 / 82.36 %, on 96.01 / 82.36 %; without this switch the keys
     * outranked right words: 95.45 / 82.02 %.
     */
    val shortGlideKeepsWords: Boolean = true,
    val maxResults: Int = 6,
    /**
     * Resampling density: N = clamp(pathLength × [samplesPerKw], [samples], [maxSamples]). 0 keeps N fixed at
     * [samples]; long paths then get sparser samples.
     */
    val samplesPerKw: Float = 0f,
    val maxSamples: Int = 128,
    /** Location-channel weight is divided by 1 + decay·max(0, keys − 5): proportional alignment drifts on long words. */
    val longWordLocationDecay: Float = 0.6003f,
    /**
     * Weight of the path-fit channel: mean squared distance (key widths²) from each path sample to the
     * template polyline. The mirror image of the key channel — it penalises words that skip keys the path
     * clearly visits ("salary" for a "salivary" gesture) and is insensitive to speed and corner cutting.
     */
    val pathFitWeight: Float = 10.57f,
    /**
     * Scale the path-fit weight with template length (×clamp((keys − 2)/4, 0.3, 1.5)): long words need it
     * most, short words are better served by the location/shape channels.
     */
    val pathFitScaleByLength: Boolean = true,
    /**
     * Cap on one interior key's squared distance (key widths²) in the ordered key alignment: a key the path
     * never reaches costs at most this, so a skipped back-and-forth ("tomorrow" glided t-o-m-o-w) degrades the
     * match instead of ruling the word out. Infinity: no cap. Human TUNE, words of 7+ keys: targets lost in the
     * coarse cut 2.75 -> 1.67 % (FUTO) and 9.9 -> 6.5 % (HWS) at 1.5; top-3 +0.4 (FUTO), top-1 unchanged.
     */
    val keySkipCost: Float = 1.573f,
    /**
     * Decode with [NeuralSwipeDecoder] when a [SwipeModel] is available and the layout is QWERTY; this template
     * matcher stays the fallback (other layouts, no model), the second opinion ([neuralSharkWeight]) and the
     * reference for comparisons. On by default: human TEST top-1 85.4 → 92.5 (FUTO) and 74.5 → 80.5 (HWS),
     * dev-phone p99 21 ms (docs/swipe-neural.md).
     */
    val neural: Boolean = true,
    // Neural decoder settings: tuned on the human TUNE splits with the production LM (docs/swipe-neural.md).
    /** Neural decoder: CTC prefix-beam width. */
    val neuralBeam: Int = 64,
    /** Neural decoder: a letter is not entered at a frame whose log posterior for it is below this. */
    val neuralPrune: Float = -12f,
    /** Neural decoder: weight of the best unigram prior below a prefix when pruning the beam. */
    val neuralLookahead: Float = 0.3f,
    /** Neural decoder: weight of −log P_ctc(keys | gesture) in the final cost. */
    val neuralCtcWeight: Float = 1f,
    /**
     * Neural decoder: weight of ln P(word | context) in the final cost (also used when the engine's neural LM
     * re-ranks the final candidates). 1.0 with that re-rank on, 0.8 without (TUNE).
     */
    val neuralLmWeight: Float = 1.0f,
    /** Neural decoder: words returned per decode. */
    val neuralMaxWords: Int = 40,
    /**
     * Neural decoder: weight of this template matcher's channel cost added to each neural candidate (0 = off). The
     * two decoders err differently: with the CMA-ES template weights, TUNE top-1 +0.3 (FUTO) and +1.7 (HWS) at 0.5.
     */
    val neuralSharkWeight: Float = 0.5f,
    /** Neural decoder: cap on that channel cost (one bad channel cannot veto a word). */
    val neuralSharkCap: Float = 30f,
    /** Neural decoder: candidates of at most this many keys get [neuralSharkWeightShort] instead of [neuralSharkWeight]. */
    val neuralShortKeys: Int = 2,
    /** Neural decoder: template weight for very short candidates, where the template matcher is strong. */
    val neuralSharkWeightShort: Float = 1.0f,
    /**
     * Neural decoder, models with an auxiliary network ([SwipeModel.aux]): words of at most this many keys are scored
     * by that network (0 = never), on strokes of at most [neuralAuxMaxFrames] input frames.
     */
    val neuralAuxKeys: Int = 2,
    val neuralAuxMaxFrames: Int = 50,
    /** Neural decoder: weight of the (blended) −log P_ctc for those words. */
    val neuralAuxWeight: Float = 0.9f,
    /** Neural decoder: share of the auxiliary network in that score (1 = auxiliary only, else blended with the main). */
    val neuralAuxMix: Float = 1.0f,
    /** Neural decoder: the auxiliary network's best short words added to the candidates. */
    val neuralAuxAdd: Int = 4,
)

/**
 * Diagnostics for one decode: where a [target] word (lower-case spelling) was lost. Stages in order:
 * no template (not in the lexicon / not typeable), first key outside the start radius, last key outside
 * the end radius, length gate, coarse cut (not among the [SwipeParams.coarseKeep] survivors).
 */
class SwipeTrace(val target: String) {
    var hasTemplate = false
    var startOk = false
    var endOk = false
    var lengthOk = false
    var keptCoarse = false
    val stage: String get() = when {
        !hasTemplate -> "no-template"
        !startOk -> "start-radius"
        !endOk -> "end-radius"
        !lengthOk -> "length-gate"
        !keptCoarse -> "coarse-cut"
        else -> "kept"
    }
}

/**
 * SHARK²-style gesture decoder (Kristensson & Zhai 2004) with an extra ordered key-alignment channel.
 * Layout-agnostic: templates come from the lexicon's characters mapped to keys by [CharKeyMap].
 *
 * 1. The path is resampled to N equidistant points.
 * 2. Candidates are pruned by start/end key proximity (words are bucketed by first/last key) and by
 *    template length vs. path length.
 * 3. Location channel: mean squared distance between the path and the word template (polyline through
 *    its keys, repeated keys collapsed, resampled to N points). Cheap; keeps the best [SwipeParams.coarseKeep].
 * 4. Shape channel (translation/scale-normalised distance) and key channel (every key must be approached
 *    by the path in order — robust to corner cutting and speed changes) refine the survivors, which are
 *    then fused with the context language model.
 */
class GestureDecoder(private val lexicon: Lexicon, private val map: CharKeyMap, private val params: SwipeParams = SwipeParams()) {
    private val keys = map.keys
    private val kw = keys.keyWidth
    private val nKeys = keys.size
    private val keyX = keys.cx
    private val keyY = keys.cy

    // Templates: one per terminal trie node whose characters all map to keys (silent ones are skipped).
    private val tplNode: IntArray
    private val tplStart: IntArray
    private val tplLen: IntArray
    private val tplKeys: ShortArray
    private val tplLength: FloatArray // key widths
    private val tplPrior: FloatArray
    /** Templates by (first key, last key), each sorted by template length (for the length-gate window). */
    private val buckets: Array<IntArray>
    private val bucketLengths: Array<FloatArray>
    /** Best unigram prior in each bucket: bounds a whole bucket's coarse cost. */
    private val bucketMaxPrior: FloatArray

    init {
        val nodes = ArrayList<Int>()
        val starts = ArrayList<Int>()
        val lens = ArrayList<Int>()
        var letters = ShortArray(1 shl 16)
        var used = 0
        val lengths = ArrayList<Float>()
        val bucketLists = Array(nKeys * nKeys) { ArrayList<Int>() }
        val buf = IntArray(Lexicon.MAX_DEPTH)
        val symBuf = IntArray(Lexicon.MAX_DEPTH)
        for (node in 0 until lexicon.nodeCount) {
            if (!lexicon.isTerminal(node)) continue
            val depth = lexicon.depthOf(node)
            var d = depth
            var x = node
            while (d > 0) { symBuf[--d] = lexicon.symOf(x); x = lexicon.parentOf(x) }
            var m = 0
            var ok = true
            for (j in 0 until depth) {
                val sym = symBuf[j]
                val k = map.gestureKey[sym]
                if (k < 0) { if (map.silent[sym]) continue; ok = false; break }
                if (m > 0 && buf[m - 1] == k) continue
                buf[m++] = k
            }
            if (!ok || m == 0) continue
            var len = 0f
            for (j in 1 until m) len += PathUtils.dist(keyX[buf[j - 1]], keyY[buf[j - 1]], keyX[buf[j]], keyY[buf[j]])
            val idx = nodes.size
            nodes.add(node); starts.add(used); lens.add(m); lengths.add(len / kw)
            while (used + m > letters.size) letters = letters.copyOf(letters.size * 2)
            for (j in 0 until m) letters[used++] = buf[j].toShort()
            bucketLists[buf[0] * nKeys + buf[m - 1]].add(idx)
        }
        tplNode = nodes.toIntArray(); tplStart = starts.toIntArray(); tplLen = lens.toIntArray()
        tplKeys = letters.copyOf(used); tplLength = lengths.toFloatArray()
        tplPrior = FloatArray(tplNode.size) { lexicon.prior(lexicon.wordsStart(tplNode[it])) }
        buckets = Array(bucketLists.size) { b -> bucketLists[b].sortedBy { tplLength[it] }.toIntArray() }
        bucketLengths = Array(buckets.size) { b -> FloatArray(buckets[b].size) { tplLength[buckets[b][it]] } }
        bucketMaxPrior = FloatArray(buckets.size) { b -> buckets[b].maxOfOrNull { tplPrior[it] } ?: Float.NEGATIVE_INFINITY }
    }

    /** First index in sorted [a] whose value is >= [v]. */
    private fun lowerBound(a: FloatArray, v: Float): Int {
        var lo = 0; var hi = a.size
        while (lo < hi) { val mid = (lo + hi) ushr 1; if (a[mid] < v) lo = mid + 1 else hi = mid }
        return lo
    }

    /** Returns null when the path is too short to be a gesture (the caller should treat it as a tap). */
    private val templateOfNode: HashMap<Int, Int> by lazy { HashMap<Int, Int>(tplNode.size * 2).also { m -> for (t in tplNode.indices) m[tplNode[t]] = t } }

    private fun locationWeightFor(t: Int): Float {
        val extra = tplLen[t] - 5
        return if (extra > 0 && params.longWordLocationDecay > 0f) params.locationWeight / (1f + params.longWordLocationDecay * extra) else params.locationWeight
    }

    /**
     * Decodes [path]. Returns null when it is not a gesture (too short) or when [cancelled] reported, while
     * decoding, that the result is no longer wanted (a newer request superseded it).
     */
    fun decode(
        path: List<TouchPoint>, lm: LanguageModel, lmContext: List<String>, lang: Int = 0, langLogPrior: Float = 0f,
        trace: SwipeTrace? = null, cancelled: (() -> Boolean)? = null,
    ): List<WordHypothesis>? {
        if (path.isEmpty() || nKeys == 0) return null
        val rawX = FloatArray(path.size)
        val rawY = FloatArray(path.size)
        var n = 0
        for (p in path) {
            if (n > 0 && rawX[n - 1] == p.x && rawY[n - 1] == p.y) continue
            rawX[n] = p.x; rawY[n] = p.y; n++
        }
        val pathLen = PathUtils.length(rawX, rawY, n) / kw
        if (pathLen < params.minGestureLength) return null

        val N = if (params.samplesPerKw > 0f) (pathLen * params.samplesPerKw).toInt().coerceIn(params.samples, params.maxSamples) else params.samples
        val px = FloatArray(N); val py = FloatArray(N)
        PathUtils.resample(rawX, rawY, n, N, px, py)
        val pnx = FloatArray(N); val pny = FloatArray(N)
        val pScale = PathUtils.normalise(px, py, N, pnx, pny)
        val pathExtent = if (pScale > 0f) 1f / pScale / kw else 0f

        // Start / end keys (soft: every key within the radius is a candidate, scored by distance).
        val rs2 = params.startRadius * params.startRadius
        val re2 = params.endRadius * params.endRadius
        val startD2 = FloatArray(nKeys) { Float.POSITIVE_INFINITY }
        val endD2 = FloatArray(nKeys) { Float.POSITIVE_INFINITY }
        for (l in 0 until nKeys) {
            val ds = sq((px[0] - keyX[l]) / kw) + sq((py[0] - keyY[l]) / kw)
            val de = sq((px[N - 1] - keyX[l]) / kw) + sq((py[N - 1] - keyY[l]) / kw)
            if (ds <= rs2) startD2[l] = ds
            if (de <= re2) endD2[l] = de
        }
        val endX = px[N - 1]; val endY = py[N - 1]

        // Squared key-to-sample distances (key widths²), shared by every template's key alignment.
        val d2 = FloatArray(nKeys * N)
        val inv = 1f / (kw * kw)
        val cap = params.keySkipCost
        for (l in 0 until nKeys) for (i in 0 until N) d2[l * N + i] = min((sq(px[i] - keyX[l]) + sq(py[i] - keyY[l])) * inv, cap)
        // Closest approach of the path to each key (key widths²): a lower bound per key of the alignment cost.
        val minD2 = FloatArray(nKeys) { l -> var m = Float.POSITIVE_INFINITY; for (i in 0 until N) if (d2[l * N + i] < m) m = d2[l * N + i]; m }

        // Coarse pass: endpoints + ordered key alignment (abandoned early once it cannot make the cut)
        // + location channel + unigram prior.
        val tx = FloatArray(N); val ty = FloatArray(N)
        val prev = FloatArray(N); val cur = FloatArray(N)
        val keep = params.coarseKeep
        val heap = PriorityQueue<Long>(keep + 1, compareByDescending<Long> { Float.fromBits((it ushr 32).toInt()) })
        // Visit (first key, last key) buckets best bound first: endpoint costs (the end with its overshoot
        // allowance) minus the bucket's best prior. Once a bound cannot beat the cut, no later bucket can.
        val pairB = ArrayList<Int>(); val pairBound = ArrayList<Float>()
        for (s in 0 until nKeys) {
            if (startD2[s].isInfinite()) continue
            for (e in 0 until nKeys) {
                if (endD2[e].isInfinite()) continue
                val b = s * nKeys + e
                if (buckets[b].isEmpty()) continue
                val endLow = (kotlin.math.sqrt(endD2[e]) - params.endOvershoot).coerceAtLeast(0f)
                pairB.add(b)
                pairBound.add(params.startWeight * startD2[s] + params.endWeight * endLow * endLow -
                    params.lmWeight * params.coarsePriorWeight * bucketMaxPrior[b])
            }
        }
        val order = pairB.indices.sortedBy { pairBound[it] }
        val lenLo = pathLen - (params.lengthRatio * pathLen + params.lengthSlack)
        val lenHi = pathLen + (params.lengthRatio * pathLen + params.lengthSlack)
        var visited = 0
        for (oi in order) {
            if (heap.size >= keep && pairBound[oi] >= Float.fromBits((heap.peek() ushr 32).toInt())) break
            if (cancelled?.invoke() == true) return null
            val b = pairB[oi]
            val s = b / nKeys
            run {
                val startCost = params.startWeight * startD2[s]
                val list = buckets[b]
                var k = lowerBound(bucketLengths[b], lenLo)
                while (k < list.size) {
                    val t = list[k++]
                    if (tplLength[t] > lenHi) break
                    if (++visited and 1023 == 0 && cancelled?.invoke() == true) return null
                    val base = startCost + params.endWeight * endDistance2(t, endX, endY) - params.lmWeight * params.coarsePriorWeight * tplPrior[t]
                    val worst = if (heap.size >= keep) Float.fromBits((heap.peek() ushr 32).toInt()) else Float.POSITIVE_INFINITY
                    if (base >= worst) continue
                    // Cheap lower bound of the ordered key alignment: each interior key's closest approach.
                    if (worst.isFinite()) {
                        val ts = tplStart[t]; val tm = tplLen[t]
                        var lb = 0f
                        for (j in 1 until tm - 1) lb += minD2[tplKeys[ts + j].toInt()]
                        if (base + params.keyWeight * lb >= worst) continue
                    }
                    val keys = keyAlignment(t, d2, N, prev, cur, (worst - base) / params.keyWeight)
                    var c = base + params.keyWeight * keys
                    if (c >= worst) continue
                    templatePoints(t, N, tx, ty)
                    c += locationWeightFor(t) * PathUtils.meanSquaredDistance(px, py, tx, ty, N) * inv
                    if (c >= worst) continue
                    heap.add((c.toRawBits().toLong() shl 32) or t.toLong())
                    if (heap.size > keep) heap.poll()
                }
            }
        }
        if (trace != null) {
            val t = lexicon.find(trace.target).let { if (it > 0) templateOfNode[it] else null }
            if (t != null) {
                trace.hasTemplate = true
                trace.startOk = startD2[tplKeys[tplStart[t]].toInt()].isFinite()
                trace.endOk = endD2[tplKeys[tplStart[t] + tplLen[t] - 1].toInt()].isFinite()
                trace.lengthOk = abs(tplLength[t] - pathLen) <= params.lengthRatio * pathLen + params.lengthSlack
                trace.keptCoarse = heap.any { (it and 0xffffffffL).toInt() == t }
            }
        }
        if (heap.isEmpty()) return emptyList()
        if (cancelled?.invoke() == true) return null

        // Fine pass.
        val tnx = FloatArray(N); val tny = FloatArray(N)
        val out = ArrayList<WordHypothesis>(heap.size * 2)
        for (e in heap) {
            val t = (e and 0xffffffffL).toInt()
            templatePoints(t, N, tx, ty)
            val loc = PathUtils.meanSquaredDistance(px, py, tx, ty, N) / (kw * kw)
            PathUtils.normalise(tx, ty, N, tnx, tny)
            val shape = PathUtils.meanSquaredDistance(pnx, pny, tnx, tny, N) * pathExtent * pathExtent
            val keyCost = keyAlignment(t, d2, N, prev, cur, Float.POSITIVE_INFINITY)
            val first = tplKeys[tplStart[t]].toInt()
            val fit = if (params.pathFitWeight > 0f) pathFit(t, px, py, N) else 0f
            val fitWeight = if (params.pathFitScaleByLength) params.pathFitWeight * ((tplLen[t] - 2) / 4f).coerceIn(0.3f, 1.5f) else params.pathFitWeight
            val channel = fitWeight * fit + locationWeightFor(t) * loc + params.shapeWeight * shape + params.keyWeight * keyCost +
                params.startWeight * startD2[first] + params.endWeight * endDistance2(t, endX, endY)
            for (id in lexicon.wordsAt(tplNode[t])) {
                val word = lexicon.word(id)
                val lp = lm.logProb(lmContext, word) + langLogPrior
                out.add(WordHypothesis(lang, id, word, channel, lp, channel - params.lmWeight * lp, false, 0))
            }
        }
        out.sortBy { it.total }
        return out
    }

    /**
     * The fine-pass channel cost (no language model) of each lexicon node `lexNodes[0, n)` for [path], into [out]
     * (+inf for nodes without a template). Lets [NeuralSwipeDecoder] combine both decoders' evidence. Returns
     * false for a path too short to be a gesture.
     */
    fun channelCosts(path: List<TouchPoint>, lexNodes: IntArray, n: Int, out: FloatArray): Boolean {
        if (path.isEmpty() || nKeys == 0) return false
        val rawX = FloatArray(path.size); val rawY = FloatArray(path.size)
        var m = 0
        for (p in path) {
            if (m > 0 && rawX[m - 1] == p.x && rawY[m - 1] == p.y) continue
            rawX[m] = p.x; rawY[m] = p.y; m++
        }
        val pathLen = PathUtils.length(rawX, rawY, m) / kw
        if (pathLen < params.minGestureLength) return false
        val N = if (params.samplesPerKw > 0f) (pathLen * params.samplesPerKw).toInt().coerceIn(params.samples, params.maxSamples) else params.samples
        val px = FloatArray(N); val py = FloatArray(N)
        PathUtils.resample(rawX, rawY, m, N, px, py)
        val pnx = FloatArray(N); val pny = FloatArray(N)
        val pScale = PathUtils.normalise(px, py, N, pnx, pny)
        val pathExtent = if (pScale > 0f) 1f / pScale / kw else 0f
        val inv = 1f / (kw * kw)
        val d2 = FloatArray(nKeys * N)
        for (l in 0 until nKeys) for (i in 0 until N) d2[l * N + i] = min((sq(px[i] - keyX[l]) + sq(py[i] - keyY[l])) * inv, params.keySkipCost)
        val tx = FloatArray(N); val ty = FloatArray(N); val tnx = FloatArray(N); val tny = FloatArray(N)
        val prev = FloatArray(N); val cur = FloatArray(N)
        for (k in 0 until n) {
            val t = templateOfNode[lexNodes[k]]
            if (t == null) { out[k] = Float.POSITIVE_INFINITY; continue }
            templatePoints(t, N, tx, ty)
            val loc = PathUtils.meanSquaredDistance(px, py, tx, ty, N) * inv
            PathUtils.normalise(tx, ty, N, tnx, tny)
            val shape = PathUtils.meanSquaredDistance(pnx, pny, tnx, tny, N) * pathExtent * pathExtent
            val keyCost = keyAlignment(t, d2, N, prev, cur, Float.POSITIVE_INFINITY)
            val first = tplKeys[tplStart[t]].toInt()
            val start = sq((px[0] - keyX[first]) / kw) + sq((py[0] - keyY[first]) / kw)
            val fit = if (params.pathFitWeight > 0f) pathFit(t, px, py, N) else 0f
            val fitWeight = if (params.pathFitScaleByLength) params.pathFitWeight * ((tplLen[t] - 2) / 4f).coerceIn(0.3f, 1.5f) else params.pathFitWeight
            out[k] = fitWeight * fit + locationWeightFor(t) * loc + params.shapeWeight * shape + params.keyWeight * keyCost +
                params.startWeight * start + params.endWeight * endDistance2(t, px[N - 1], py[N - 1])
        }
        return true
    }

    /**
     * Squared distance (key widths²) from the gesture's end point to template [t]'s last key, not counting an
     * overshoot of up to [SwipeParams.endOvershoot] along the final segment (fingers keep going past the key).
     */
    private fun endDistance2(t: Int, ex: Float, ey: Float): Float {
        val s = tplStart[t]; val m = tplLen[t]
        val last = tplKeys[s + m - 1].toInt()
        var dx = (ex - keyX[last]) / kw; var dy = (ey - keyY[last]) / kw
        if (m >= 2) {
            val prev = tplKeys[s + m - 2].toInt()
            var ux = keyX[last] - keyX[prev]; var uy = keyY[last] - keyY[prev]
            val l = kotlin.math.sqrt(ux * ux + uy * uy)
            if (l > 1e-3f) {
                ux /= l; uy /= l
                val along = dx * ux + dy * uy
                if (along > 0f) { val cut = kotlin.math.min(along, params.endOvershoot); dx -= cut * ux; dy -= cut * uy }
            }
        }
        return dx * dx + dy * dy
    }

    /** Mean squared distance (key widths²) from the path samples to template [t]'s polyline. */
    private fun pathFit(t: Int, px: FloatArray, py: FloatArray, n: Int): Float {
        val s = tplStart[t]; val m = tplLen[t]
        var sum = 0f
        for (i in 0 until n) {
            var best = Float.POSITIVE_INFINITY
            if (m == 1) {
                val k = tplKeys[s].toInt()
                best = sq(px[i] - keyX[k]) + sq(py[i] - keyY[k])
            } else for (j in 0 until m - 1) {
                val a = tplKeys[s + j].toInt(); val b = tplKeys[s + j + 1].toInt()
                val ax = keyX[a]; val ay = keyY[a]
                val dx = keyX[b] - ax; val dy = keyY[b] - ay
                val l2 = dx * dx + dy * dy
                val u = if (l2 > 1e-6f) (((px[i] - ax) * dx + (py[i] - ay) * dy) / l2).coerceIn(0f, 1f) else 0f
                val d = sq(px[i] - (ax + u * dx)) + sq(py[i] - (ay + u * dy))
                if (d < best) best = d
            }
            sum += best
        }
        return sum / n / (kw * kw)
    }

    private fun templatePoints(t: Int, count: Int, outX: FloatArray, outY: FloatArray) {
        val s = tplStart[t]; val m = tplLen[t]
        if (m == 1) {
            val l = tplKeys[s].toInt()
            for (k in 0 until count) { outX[k] = keyX[l]; outY[k] = keyY[l] }
            return
        }
        val total = tplLength[t] * kw
        val step = total / (count - 1)
        var seg = 1
        var a = tplKeys[s].toInt(); var b = tplKeys[s + 1].toInt()
        var segStart = 0f
        var segLen = PathUtils.dist(keyX[a], keyY[a], keyX[b], keyY[b])
        outX[0] = keyX[a]; outY[0] = keyY[a]
        for (k in 1 until count - 1) {
            val target = k * step
            while (seg < m - 1 && segStart + segLen < target) {
                segStart += segLen
                seg++
                a = b; b = tplKeys[s + seg].toInt()
                segLen = PathUtils.dist(keyX[a], keyY[a], keyX[b], keyY[b])
            }
            val f = if (segLen > 1e-6f) ((target - segStart) / segLen).coerceIn(0f, 1f) else 0f
            outX[k] = keyX[a] + (keyX[b] - keyX[a]) * f
            outY[k] = keyY[a] + (keyY[b] - keyY[a]) * f
        }
        val z = tplKeys[s + m - 1].toInt()
        outX[count - 1] = keyX[z]; outY[count - 1] = keyY[z]
    }

    /**
     * Minimum over monotonic alignments of the interior keys to path samples of the summed squared
     * key-to-path distance (key widths², from [d2]). First/last keys are covered by the endpoint channel.
     * Returns +inf as soon as the partial cost exceeds [bound].
     */
    private fun keyAlignment(t: Int, d2: FloatArray, n: Int, prev: FloatArray, cur: FloatArray, bound: Float): Float {
        val s = tplStart[t]; val m = tplLen[t]
        if (m <= 2) return 0f
        prev.fill(Float.POSITIVE_INFINITY); prev[0] = 0f
        var p = prev; var c = cur
        for (j in 1 until m - 1) {
            val off = tplKeys[s + j] * n
            var running = Float.POSITIVE_INFINITY
            var rowMin = Float.POSITIVE_INFINITY
            for (i in 0 until n) {
                running = min(running, p[i])
                val v = running + d2[off + i]
                c[i] = v
                if (v < rowMin) rowMin = v
            }
            if (rowMin > bound) return Float.POSITIVE_INFINITY
            val tmp = p; p = c; c = tmp
        }
        var best = Float.POSITIVE_INFINITY
        for (i in 0 until n) best = min(best, p[i])
        return best
    }

    private fun sq(v: Float) = v * v
}
