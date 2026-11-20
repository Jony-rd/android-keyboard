package dev.pboard.engine.swipe

import dev.pboard.engine.LanguageModel
import dev.pboard.engine.TouchPoint
import dev.pboard.engine.decode.CharKeyMap
import dev.pboard.engine.decode.KeySpace
import dev.pboard.engine.decode.Lexicon
import dev.pboard.engine.decode.WordHypothesis
import kotlin.math.exp
import kotlin.math.ln

/**
 * Runs the [SwipeModel] on a gesture once and shares the per-frame log-posteriors between the decoders of every
 * active language (the network does not depend on the lexicon). Bound to one layout. A version 1 model exists only
 * for layouts whose letter keys fit the QWERTY grid it was trained on ([SwipeFeatures.fitQwerty]); a version 2
 * model scores whatever keys the layout has ([SwipeFeatures2]).
 */
class NeuralSwipeEncoder private constructor(
    val model: SwipeModel, val keys: KeySpace, private val fit: SwipeFeatures.GridFit?, private val feat2: SwipeFeatures2?,
) {
    private val encoder = model.newEncoder().also { e -> feat2?.let { e.setKeys(it.keyX, it.keyY) } }
    private val auxEncoder = model.aux?.newEncoder()?.also { e -> feat2?.let { e.setKeys(it.keyX, it.keyY) } }
    private var lastFeatures = 0
    private var auxFor: List<TouchPoint>? = null

    /** Whether the model carries an auxiliary network ([SwipeModel.aux]). */
    val hasAux: Boolean get() = auxEncoder != null
    /** Input frames of the last encoded path (before the network's stride). */
    val inputFrames: Int get() = lastFeatures
    /** Auxiliary log-posteriors of the last path ([encodeAux]): [auxFrameCount] × [classes]. */
    var auxLogProbs = FloatArray(0); private set
    var auxFrameCount = 0; private set

    /** Runs the auxiliary network on the last encoded path's features; false if [cancelled]. */
    fun encodeAux(cancelled: (() -> Boolean)? = null): Boolean {
        val e = auxEncoder ?: return false
        if (auxFor === lastPath && lastPath != null) return true
        val t = e.encode(x, lastFeatures, cancelled)
        if (t < 0) return false
        auxFrameCount = t; auxLogProbs = e.lastOutput; auxFor = lastPath
        return true
    }
    private val frames = SwipeFeatures.Frames(256)
    private var u = FloatArray(256); private var v = FloatArray(256); private var t = FloatArray(256)
    private var x = FloatArray(256 * SwipeFeatures.N_FEAT)

    /** Model class of each [KeySpace] key (v1: 1 + letter index; v2: 1 + key index), or -1 for keys it does not know. */
    val classOfKey = IntArray(keys.size) { k ->
        if (feat2 != null) 1 + k else keys.labels[k].let { c -> if (c in 'a'..'z') 1 + (c - 'a') else -1 }
    }

    /** Classes per frame (blank + keys). */
    val classes: Int get() = encoder.classes

    private var lastPath: List<TouchPoint>? = null
    private var lastSize = -1
    /** Log-posteriors of the last encoded path: [frameCount] × [classes]. */
    var logProbs = FloatArray(0); private set
    var frameCount = 0; private set

    /** Encodes [path] unless it is the path encoded last (same list, same size); false if [cancelled]. */
    fun encode(path: List<TouchPoint>, cancelled: (() -> Boolean)? = null): Boolean {
        if (path === lastPath && path.size == lastSize) return true
        evalEmissions?.invoke(path)?.let { e ->
            logProbs = e; frameCount = e.size / classes; lastPath = path; lastSize = path.size
            return true
        }
        lastPath = null
        val n: Int
        if (feat2 != null) {
            x = feat2.features(path, x)
            n = feat2.frames
        } else {
            if (path.size > u.size) { val c = path.size * 2; u = FloatArray(c); v = FloatArray(c); t = FloatArray(c) }
            val m = SwipeFeatures.toGrid(path, fit!!, u, v, t)
            model.features.resample(u, v, t, m, frames)
            x = model.features.features(frames, x)
            n = frames.size
        }
        lastFeatures = n
        auxFor = null
        val t = encoder.encode(x, n, cancelled)
        if (t < 0) return false
        frameCount = t
        logProbs = encoder.lastOutput
        lastPath = path; lastSize = path.size
        return true
    }

    companion object {
        /**
         * Evaluation only (never set in production): per-frame log-scores for a path computed elsewhere, e.g. another
         * model's emissions ([frames] × [classes], blank first), decoded instead of [model]'s; null falls through to
         * the model. See NeuralSwipeEval --emissions.
         */
        @Volatile internal var evalEmissions: ((List<TouchPoint>) -> FloatArray?)? = null

        /** An encoder for [keys], or null when the model cannot serve this layout. */
        fun create(model: SwipeModel, keys: KeySpace): NeuralSwipeEncoder? {
            if (model.version == 2) {
                if (keys.size < 2) return null
                val f = SwipeFeatures2(keys.cx, keys.cy, model.stepS, model.stepT, model.maxFrames)
                if (f.keyY.maxOrNull() == f.keyY.minOrNull()) return null // a single row: no 2-D layout
                return NeuralSwipeEncoder(model, keys, null, f)
            }
            return SwipeFeatures.fitQwerty(keys)?.let { NeuralSwipeEncoder(model, keys, it, null) }
        }
    }
}

/**
 * The lexicon as a trie over gesture key sequences: each word's letters mapped to keys ([CharKeyMap.gestureKey];
 * silent characters skipped, accented letters on their base key), consecutive repeats merged ("letter" -> l-e-t-e-r)
 * because a gesture cannot show them, and keys numbered by model class. Several spellings can share a sequence
 * ("its"/"it's", "leter"/"letter"); the language model separates them.
 */
internal class KeySequenceTrie(lexicon: Lexicon, map: CharKeyMap, classOfKey: IntArray) {
    val label: ByteArray
    val firstChild: IntArray
    val childCount: ByteArray
    /** Lexicon nodes (spellings) ending at each trie node: [tplStart, tplStart + tplCount). */
    val tplStart: IntArray
    val tplCount: IntArray
    val tplLexNodes: IntArray
    /** Best unigram log prior below each node (beam look-ahead). */
    val maxPrior: FloatArray
    /** Keys from the root to each node. */
    val depth: ByteArray
    val parent: IntArray

    /** Terminal nodes with at most [maxKeys] keys (cached per value). */
    private var shortNodes: IntArray? = null
    private var shortKeys = -1
    fun terminalsUpTo(maxKeys: Int): IntArray {
        if (shortKeys == maxKeys) return shortNodes!!
        val out = ArrayList<Int>()
        for (n in 1 until label.size) if (depth[n] <= maxKeys && tplCount[n] > 0) out.add(n)
        shortKeys = maxKeys; shortNodes = out.toIntArray()
        return shortNodes!!
    }
    val nodeCount: Int get() = label.size

    init {
        val seqs = ArrayList<String>()
        val lexNodes = ArrayList<Int>()
        val symBuf = IntArray(Lexicon.MAX_DEPTH)
        val sb = StringBuilder()
        for (node in 0 until lexicon.nodeCount) {
            if (!lexicon.isTerminal(node)) continue
            val depth = lexicon.depthOf(node)
            var d = depth; var x = node
            while (d > 0) { symBuf[--d] = lexicon.symOf(x); x = lexicon.parentOf(x) }
            sb.setLength(0)
            var ok = true
            for (j in 0 until depth) {
                val sym = symBuf[j]
                val k = map.gestureKey[sym]
                if (k < 0) { if (map.silent[sym]) continue; ok = false; break }
                val c = classOfKey[k]
                if (c < 0) { ok = false; break }
                val ch = ('a' + c - 1)
                if (sb.isNotEmpty() && sb[sb.length - 1] == ch) continue
                sb.append(ch)
            }
            if (!ok || sb.isEmpty()) continue
            seqs.add(sb.toString()); lexNodes.add(node)
        }
        val order = seqs.indices.sortedWith(compareBy<Int> { seqs[it] })
        val paths = Array(order.size) { seqs[order[it]] }
        val tpl = IntArray(order.size) { lexNodes[order[it]] }
        // Breadth-first construction over sorted ranges (children contiguous, sorted by label).
        var cap = 1024
        var nLab = ByteArray(cap); var nFirst = IntArray(cap); var nCount = ByteArray(cap)
        var nTs = IntArray(cap); var nTc = IntArray(cap); var nDepth = IntArray(cap); var rLo = IntArray(cap); var rHi = IntArray(cap)
        fun grow() {
            cap *= 2
            nLab = nLab.copyOf(cap); nFirst = nFirst.copyOf(cap); nCount = nCount.copyOf(cap); nTs = nTs.copyOf(cap)
            nTc = nTc.copyOf(cap); nDepth = nDepth.copyOf(cap); rLo = rLo.copyOf(cap); rHi = rHi.copyOf(cap)
        }
        var count = 1
        rLo[0] = 0; rHi[0] = paths.size
        var id = 0
        while (id < count) {
            val lo = rLo[id]; val hi = rHi[id]; val d = nDepth[id]
            var j = lo
            while (j < hi && paths[j].length == d) j++
            nTs[id] = lo; nTc[id] = j - lo
            nFirst[id] = count
            while (j < hi) {
                val c = paths[j][d]
                var k = j
                while (k < hi && paths[k][d] == c) k++
                if (count == cap) grow()
                nLab[count] = (c - 'a' + 1).toByte(); nDepth[count] = d + 1; rLo[count] = j; rHi[count] = k
                count++
                j = k
            }
            nCount[id] = (count - nFirst[id]).toByte()
            id++
        }
        label = nLab.copyOf(count); firstChild = nFirst.copyOf(count); childCount = nCount.copyOf(count)
        tplStart = nTs.copyOf(count); tplCount = nTc.copyOf(count); tplLexNodes = tpl
        depth = ByteArray(count) { minOf(nDepth[it], 127).toByte() }
        maxPrior = FloatArray(count) { Float.NEGATIVE_INFINITY }
        parent = IntArray(count) { -1 }
        for (n in 0 until count) for (k in 0 until childCount[n]) parent[firstChild[n] + k] = n
        for (n in count - 1 downTo 0) {
            for (k in 0 until tplCount[n]) {
                val p = lexicon.prior(lexicon.wordsStart(tpl[tplStart[n] + k]))
                if (p > maxPrior[n]) maxPrior[n] = p
            }
            val p = parent[n]
            if (p >= 0 && maxPrior[n] > maxPrior[p]) maxPrior[p] = maxPrior[n]
        }
    }
}

/**
 * Neural gesture decoder (docs/swipe-neural.md), after Alsharif et al. (ICASSP 2015) and Ouyang et al. (2017): an
 * LSTM reads the touch frames and emits per-frame letter posteriors trained with CTC ([SwipeModel]); a prefix beam
 * search over the lexicon's key-sequence trie finds the words whose CTC likelihood is highest (with a unigram
 * look-ahead keeping frequent words in the beam), and the survivors are ranked by
 * `−log P_ctc(keys | gesture) − lmWeight · ln P(word | context)`.
 *
 * Same contract as [GestureDecoder.decode], so the engine can use either; not thread-safe.
 */
class NeuralSwipeDecoder(
    private val lexicon: Lexicon, map: CharKeyMap, private val encoder: NeuralSwipeEncoder, private val params: SwipeParams = SwipeParams(),
    /** The template matcher over the same lexicon: second opinion ([SwipeParams.neuralSharkWeight]) and fallback. */
    private val templates: (() -> GestureDecoder?)? = null,
) {
    private val trie = KeySequenceTrie(lexicon, map, encoder.classOfKey)
    private val kw = map.keys.keyWidth
    private val nc get() = encoder.classes
    private var allowed = BooleanArray(0)

    // Beam (current frame) and the merge table for the next frame.
    private var cap = 0
    private var bNode = IntArray(0); private var bPb = FloatArray(0); private var bPnb = FloatArray(0); private var bN = 0
    private var sNode = IntArray(0); private var sPos = IntArray(0); private var sPb = FloatArray(0); private var sPnb = FloatArray(0); private var sScore = FloatArray(0); private var sN = 0
    private var hKey = IntArray(0); private var hVal = IntArray(0); private var hMask = 0
    private var idx = IntArray(0)

    private var capClasses = 0

    private fun ensure(beam: Int, nc: Int) {
        if (beam <= cap && nc <= capClasses) return
        cap = beam; capClasses = nc
        bNode = IntArray(beam); bPb = FloatArray(beam); bPnb = FloatArray(beam)
        val maxSlots = beam * (2 + nc)
        sNode = IntArray(maxSlots); sPos = IntArray(maxSlots); sPb = FloatArray(maxSlots); sPnb = FloatArray(maxSlots); sScore = FloatArray(maxSlots); idx = IntArray(maxSlots)
        var h = 1
        while (h < 2 * maxSlots) h = h shl 1
        hKey = IntArray(h) { -1 }; hVal = IntArray(h); hMask = h - 1
    }

    private fun slot(node: Int): Int {
        var p = (node * -0x61c88647) ushr 7 and hMask
        while (true) {
            val k = hKey[p]
            if (k == node) return hVal[p]
            if (k < 0) {
                hKey[p] = node; hVal[p] = sN; sPos[sN] = p
                sNode[sN] = node; sPb[sN] = NEG; sPnb[sN] = NEG
                return sN++
            }
            p = (p + 1) and hMask
        }
    }

    /**
     * Decodes [path]. Returns null when it is not a gesture (too short) or when [cancelled] reported that the result
     * is no longer wanted (the encoder checks between layer passes, the search every 16 frames).
     */
    fun decode(
        path: List<TouchPoint>, lm: LanguageModel, lmContext: List<String>, lang: Int = 0, langLogPrior: Float = 0f,
        cancelled: (() -> Boolean)? = null,
    ): List<WordHypothesis>? {
        if (path.isEmpty()) return null
        var len = 0f
        for (i in 1 until path.size) len += PathUtils.dist(path[i - 1].x, path[i - 1].y, path[i].x, path[i].y)
        if (len / kw < params.minGestureLength) return null
        if (!encoder.encode(path, cancelled) || cancelled?.invoke() == true) return null
        val lp = encoder.logProbs
        val frames = encoder.frameCount
        val beam = params.neuralBeam
        val nc = this.nc
        ensure(beam, nc)
        if (allowed.size < nc) allowed = BooleanArray(nc)
        val la = params.neuralLookahead
        val prune = params.neuralPrune
        // Start: the empty prefix, all mass "ending in blank".
        bN = 1; bNode[0] = 0; bPb[0] = 0f; bPnb[0] = NEG
        for (t in 0 until frames) {
            if (t and 15 == 15 && cancelled?.invoke() == true) return null
            val base = t * nc
            var any = false
            for (c in 1 until nc) { val ok = lp[base + c] > prune; allowed[c] = ok; any = any or ok }
            val blank = lp[base]
            sN = 0
            for (i in 0 until bN) {
                val node = bNode[i]; val pb = bPb[i]; val pnb = bPnb[i]
                val tot = lse(pb, pnb)
                var s = slot(node)
                sPb[s] = lse(sPb[s], tot + blank)
                if (node != 0) {
                    val lab = trie.label[node].toInt()
                    s = slot(node)
                    sPnb[s] = lse(sPnb[s], pnb + lp[base + lab])
                }
                if (!any) continue
                val f = trie.firstChild[node]
                for (k in 0 until trie.childCount[node]) {
                    val ch = f + k
                    val lab = trie.label[ch].toInt()
                    if (!allowed[lab]) continue
                    s = slot(ch)
                    sPnb[s] = lse(sPnb[s], tot + lp[base + lab])
                }
            }
            // Clear the merge table and keep the best `beam` entries (score = log P + look-ahead prior).
            for (i in 0 until sN) hKey[sPos[i]] = -1
            for (i in 0 until sN) { sScore[i] = lse(sPb[i], sPnb[i]) + la * trie.maxPrior[sNode[i]]; idx[i] = i }
            val keep = minOf(beam, sN)
            if (sN > keep) selectTop(sScore, idx, sN, keep)
            for (j in 0 until keep) { val i = idx[j]; bNode[j] = sNode[i]; bPb[j] = sPb[i]; bPnb[j] = sPnb[i] }
            bN = keep
        }
        // Spellings: terminal nodes of the final beam (trie node, lexicon node, key count, -log P_ctc).
        val cNode = ArrayList<Int>(); val cLex = ArrayList<Int>(); val cKeys = ArrayList<Int>(); val cChan = ArrayList<Float>()
        val byLex = HashMap<Int, Int>()
        for (j in 0 until bN) {
            val node = bNode[j]
            val ll = lse(bPb[j], bPnb[j])
            for (k in 0 until trie.tplCount[node]) {
                val lex = trie.tplLexNodes[trie.tplStart[node] + k]
                byLex[lex] = cLex.size
                cNode.add(node); cLex.add(lex); cKeys.add(trie.depth[node].toInt()); cChan.add(-params.neuralCtcWeight * ll)
            }
        }
        // Very short words: when the model carries an auxiliary network (co-augmented, so without a spelling prior) and
        // the stroke is short enough to be one, words of at most neuralAuxKeys keys are scored by forced alignment under
        // that network instead, and its best short words join the candidates.
        if (params.neuralAuxKeys > 0 && encoder.hasAux && encoder.inputFrames <= params.neuralAuxMaxFrames) {
            if (!encoder.encodeAux(cancelled)) return null
            val alp = encoder.auxLogProbs; val at = encoder.auxFrameCount
            val shortNodes = trie.terminalsUpTo(params.neuralAuxKeys)
            val mix = params.neuralAuxMix
            val mlp = encoder.logProbs; val mt = encoder.frameCount
            // Blend of the auxiliary and main networks' scores (forced alignment under each).
            val sc = FloatArray(shortNodes.size) {
                val a = forcedCtc(shortNodes[it], alp, at, nc)
                if (mix >= 1f) a else mix * a + (1 - mix) * forcedCtc(shortNodes[it], mlp, mt, nc)
            }
            val scoreOf = HashMap<Int, Float>(shortNodes.size * 2)
            for (i in shortNodes.indices) scoreOf[shortNodes[i]] = sc[i]
            for (k in cNode.indices) if (cKeys[k] <= params.neuralAuxKeys) scoreOf[cNode[k]]?.let { cChan[k] = -params.neuralAuxWeight * it }
            for (i in shortNodes.indices.sortedByDescending { sc[it] }.take(params.neuralAuxAdd)) {
                val node = shortNodes[i]
                for (k in 0 until trie.tplCount[node]) {
                    val lex = trie.tplLexNodes[trie.tplStart[node] + k]
                    if (byLex.containsKey(lex)) continue
                    byLex[lex] = cLex.size
                    cNode.add(node); cLex.add(lex); cKeys.add(trie.depth[node].toInt()); cChan.add(-params.neuralAuxWeight * sc[i])
                }
            }
        }
        val ns = cLex.size
        val lexNodes = IntArray(ns) { cLex[it] }; val channel = FloatArray(ns) { cChan[it] }
        // Optionally add the template matcher's channel cost for the same spellings (both decoders' evidence).
        val sw = params.neuralSharkWeight
        if (sw > 0f && ns > 0) {
            val g = templates?.invoke()
            val cost = FloatArray(ns)
            if (g != null && g.channelCosts(path, lexNodes, ns, cost)) for (k in 0 until ns) {
                val w = if (cKeys[k] <= params.neuralShortKeys) params.neuralSharkWeightShort else sw
                channel[k] += w * minOf(cost[k], params.neuralSharkCap)
            }
        }
        val out = ArrayList<WordHypothesis>()
        val lmW = params.neuralLmWeight
        for (k in 0 until ns) {
            for (id in lexicon.wordsAt(lexNodes[k])) {
                val word = lexicon.word(id)
                val lpw = lm.logProb(lmContext, word) + langLogPrior
                out.add(WordHypothesis(lang, id, word, channel[k], lpw, channel[k] - lmW * lpw, false, 0))
            }
        }
        // No complete word survived the beam (an unusual path): let the template matcher answer instead.
        if (out.isEmpty()) return templates?.invoke()?.decode(path, lm, lmContext, lang, langLogPrior, cancelled = cancelled) ?: out
        out.sortBy { it.total }
        return if (out.size > params.neuralMaxWords) out.subList(0, params.neuralMaxWords).toList() else out
    }

    /** log P_ctc of the key sequence ending at trie [node] under log-posteriors [lp] ([t] frames × [nc] classes). */
    private fun forcedCtc(node: Int, lp: FloatArray, t: Int, nc: Int): Float {
        val d = trie.depth[node].toInt()
        val lab = IntArray(d)
        var x = node
        for (i in d - 1 downTo 0) { lab[i] = trie.label[x].toInt(); x = trie.parent[x] }
        val sN = 2 * d + 1 // blank, l1, blank, l2, ..., blank
        var a = FloatArray(sN) { NEG }; var b = FloatArray(sN)
        a[0] = lp[0]; if (d > 0) a[1] = lp[lab[0]]
        for (f in 1 until t) {
            val base = f * nc
            for (st in 0 until sN) {
                var v = a[st]
                if (st >= 1) v = lse(v, a[st - 1])
                // Skip over a blank between two different labels (keys sequences never repeat a key).
                if (st >= 2 && st % 2 == 1) v = lse(v, a[st - 2])
                b[st] = if (v == NEG) NEG else v + lp[base + if (st % 2 == 0) 0 else lab[st / 2]]
            }
            val tmp = a; a = b; b = tmp
        }
        return if (d == 0) a[0] else lse(a[sN - 1], a[sN - 2])
    }

    /** Partial sort: moves the indices of the [k] largest [score]s to idx[0, k). */
    private fun selectTop(score: FloatArray, idx: IntArray, n: Int, k: Int) {
        var lo = 0; var hi = n - 1
        while (lo < hi) {
            val pivot = score[idx[(lo + hi) ushr 1]]
            var i = lo; var j = hi
            while (i <= j) {
                while (score[idx[i]] > pivot) i++
                while (score[idx[j]] < pivot) j--
                if (i <= j) { val tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp; i++; j-- }
            }
            if (k - 1 <= j) hi = j else if (k - 1 >= i) lo = i else return
        }
    }

    companion object {
        private const val NEG = Float.NEGATIVE_INFINITY

        // log(exp(a) + exp(b)) with a table for log1p(exp(-d)), d in [0, 16] (abs error < 1e-4).
        private const val LSE_RES = 64
        private const val LSE_MAX = 16f
        private val LSE = FloatArray((LSE_MAX * LSE_RES).toInt() + 2) { ln(1.0 + exp(-it.toDouble() / LSE_RES)).toFloat() }

        internal fun lse(a: Float, b: Float): Float {
            if (a == NEG) return b
            if (b == NEG) return a
            val m: Float; val d: Float
            if (a > b) { m = a; d = a - b } else { m = b; d = b - a }
            if (d >= LSE_MAX) return m
            val p = d * LSE_RES
            val i = p.toInt()
            val f = p - i
            return m + LSE[i] + (LSE[i + 1] - LSE[i]) * f
        }
    }
}
