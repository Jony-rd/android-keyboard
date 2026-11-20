package dev.pboard.engine.decode

import dev.pboard.engine.Context
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min

/**
 * TF-01 tap-time key resolution: which key a single tap types. Among the hit key and its neighbouring keys,
 * picks the argmax of
 *
 *     −spatialCost(key, touch) + priorWeight · log P(key | word so far, context)
 *
 * where the spatial cost is the per-key Gaussian of [SpatialModel] (centres moved by the user's learned
 * offsets) and the letter prior mixes the lexicon (the best word below each trie child of the prefix), the
 * language model's predictions for the context (words starting with the prefix) and, when the prefix is in no
 * lexicon, the letter-trigram model. Taps inside the hit key's centre zone always type that key.
 *
 * The space bar and the delete key take part as two more keys ([TapResolveParams.spaceAndDelete]) inside a
 * word: the space bar's prior is that the word ends here (the word itself as a lexicon entry or prediction, or
 * the letter model's end of word), the delete key's a fixed share; their spatial cost is the same Gaussian
 * around their centre line (a wide key is a row of key centres). Other keys next to letters (".", ",", shift:
 * [TapResolveParams.otherKeys]) take part for taps on them inside a word only (such a tap never becomes anything but
 * a letter or the key itself): "." and "," with the space bar's prior (they end the word too), shift with a fixed
 * share. A letter key's tap never becomes the delete key: a tap decides only itself. Exception: a tap on "." or ","
 * next to the space bar, close to the edge they share, may be a space that missed the bar
 * ([TapResolveParams.punctuationSpace]): the bar's prior is the chance that the word ends here and the sentence goes
 * on, the key's that it ends (at least a small fixed share); the sentence-end lookup allocates.
 *
 * Built and replaced (never mutated) by the engine on its worker thread; [resolve] only reads immutable
 * tables and allocates nothing. Its scratch arrays make it single-caller: call it from one thread (the UI
 * thread) only.
 */
class TapResolver internal constructor(
    private val labels: CharArray,
    private val lower: Array<String>,
    private val upper: Array<String>,
    private val cx: FloatArray,
    private val cy: FloatArray,
    private val meanX: FloatArray,
    private val meanY: FloatArray,
    private val invVarX: Float,
    private val invVarY: Float,
    private val keyWidth: Float,
    /**
     * Per key (letter keys, then [space], then [delete]): itself, then the keys within the neighbour radius.
     * Sized [labels].size + 2.
     */
    private val neighbours: Array<IntArray>,
    private val langs: Array<Lang>,
    private val rules: TextRules,
    private val params: TapResolveParams,
    /** Hit rectangles (left, top, right, bottom) of the space bar and the delete key, or null. */
    private val spaceRect: FloatArray? = null,
    private val deleteRect: FloatArray? = null,
    /** Mean touch offset from key centres (applied to the space bar and the delete key). */
    private val offX: Float = 0f,
    private val offY: Float = 0f,
    /**
     * Other keys next to letter keys ([TapResolveParams.otherKeys]): their labels (as passed to [resolve]), hit
     * rectangles, the vertical shift of their centre line (key widths) and, per key, which of its edges face letter
     * keys (top, left, right): a tap deeper inside than [TapResolveParams.functionEdge] from those always types it.
     * Pseudo-keys n + 2 + i; [neighbours] covers them.
     */
    private val otherLabels: Array<String> = emptyArray(),
    private val otherRects: Array<FloatArray> = emptyArray(),
    private val otherShift: FloatArray = FloatArray(0),
    private val otherEdges: Array<BooleanArray> = emptyArray(),
    /**
     * Per other key, the side on which it borders the space bar in the same row (1: the bar is on its left, 2: on its
     * right, 0: not next to the bar) for [TapResolveParams.punctuationSpace].
     */
    private val otherSpaceSide: IntArray = IntArray(otherLabels.size),
    /**
     * The chance that [label] ("." or ",") follows [word] after [context] (the sentence, or clause, ends there), or
     * null without a language model to ask.
     */
    private val punctuationProb: ((context: Context, word: String, label: String) -> Float)? = null,
) {
    /** One language's view: its lexicon, the lexicon symbol of each key's letter (or -1) and its log weight. */
    class Lang(val lexicon: Lexicon, val letters: LetterModel?, val keySym: IntArray, val logWeight: Float)

    /** The language model's next words for one context (lower case, probabilities), published by the engine. */
    class ContextPrior(val sentenceStart: Boolean, val words: List<String>, val preds: Array<String>, val probs: FloatArray)

    @Volatile internal var contextPrior: ContextPrior? = null

    private val n = labels.size
    /** Pseudo-key indices of the space bar and the delete key. */
    private val space = n
    private val delete = n + 1
    private val lexP = FloatArray(n + 2 + otherLabels.size)
    private val ctxP = FloatArray(n + 2 + otherLabels.size)
    private val dotted = rules.locale.language == "tr" || rules.locale.language == "az"
    /** Pseudo-keys that end the word like the space bar ("." and ","): their prior is the word ending here. */
    private val endsWordKey = BooleanArray(n + 2 + otherLabels.size) { it == space || (it > delete && otherLabels[it - delete - 1] in WORD_ENDERS) }

    /** See [dev.pboard.engine.KeyboardEngine.resolveTap]. */
    fun resolve(context: Context, typedSoFar: String, x: Float, y: Float, rectKey: String): String {
        if (!params.enabled || !context.autoCorrectAllowed || context.exactKeys) return rectKey
        val inWord = typedSoFar.isNotEmpty()
        // At a word's start only the space bar, "." and shift, only near their edge and with a strong prior for the key
        // itself ([TapResolveParams.wordStart]): a deliberate space or period after a word must stay one.
        val startEdge = params.wordStartEdge * keyWidth
        val k: Int
        when {
            rectKey == SPACE || rectKey == " " -> {
                val r = spaceRect ?: return rectKey
                if (!params.spaceAndDelete) return rectKey
                if (inWord) { if (y - r[1] > params.functionEdge * keyWidth) return rectKey }
                else if (!params.wordStart || y - r[1] > startEdge) return rectKey
                k = space
            }
            rectKey == DELETE -> {
                val r = deleteRect ?: return rectKey
                if (!params.spaceAndDelete || !inWord || min(y - r[1], x - r[0]) > params.functionEdge * keyWidth) return rectKey
                k = delete
            }
            rectKey.length == 1 && Character.isLetter(rectKey[0]) -> {
                k = indexOf(lowerChar(rectKey[0]))
                if (k < 0) return rectKey
                val zx = (x - cx[k]) / keyWidth; val zy = (y - cy[k]) / keyWidth
                if (zx * zx + zy * zy <= params.centreZone * params.centreZone) return rectKey
            }
            else -> {
                val o = otherLabels.indexOf(rectKey)
                if (o < 0 || !params.otherKeys) return rectKey
                if (inWord) {
                    val rival = spaceRival(o, x, typedSoFar)
                    if (otherEdgeDistance(o, x, y) > params.functionEdge * keyWidth) {
                        return if (rival) againstSpace(n + 2 + o, context, typedSoFar, x, y, rectKey) else rectKey
                    }
                    // Near the letters too: a letter first, else the bar against the key.
                    if (rival) {
                        val r = resolveAmong(n + 2 + o, context, typedSoFar, x, y, rectKey)
                        return if (r == rectKey) againstSpace(n + 2 + o, context, typedSoFar, x, y, rectKey) else r
                    }
                }
                else if (!params.wordStart || rectKey !in WORD_START_KEYS || otherEdgeDistance(o, x, y) > startEdge) return rectKey
                k = n + 2 + o
            }
        }
        return resolveAmong(k, context, typedSoFar, x, y, rectKey)
    }

    /** [resolve] once the hit key [k] is known to take part: the best of it and its neighbours. */
    private fun resolveAmong(k: Int, context: Context, typedSoFar: String, x: Float, y: Float, rectKey: String): String {
        val inWord = typedSoFar.isNotEmpty()
        val cand = neighbours[k]
        if (cand.size < 2) return rectKey
        if (!lexiconPrior(typedSoFar, cand) && !letterPrior(typedSoFar, cand)) return rectKey
        val ctxWeight = if (contextPrior(context, typedSoFar, cand)) params.contextWeight else 0f
        // A tap decides only itself: a letter key's tap never becomes the delete key (that would remove a letter
        // already typed); only a tap on the delete key may delete.
        val allowDelete = k == delete
        // A word-start tap on the space bar, "." or shift: the key keeps a fixed (large) share of the prior.
        val startKey = !inWord && k >= n
        var fixedShare = 0f
        var wide = k >= n
        for (j in cand) {
            if (startKey && j == k) { fixedShare += params.wordStartKeyPrior; continue }
            if (j == delete && !allowDelete) continue
            if (j == delete) fixedShare += params.deletePrior
            if (j > delete && !endsWordKey[j]) fixedShare += params.otherPrior
            if (j >= n && params.spaceAndDelete && inWord) wide = true
        }
        val keep = 1f - fixedShare
        val priorWeight = if (wide) params.wideKeyPriorWeight else params.priorWeight
        var best = k
        var bestScore = Float.NEGATIVE_INFINITY
        for (j in cand) {
            if (j >= n && !(params.spaceAndDelete && inWord) && !(startKey && j == k)) continue
            if (j == delete && !allowDelete) continue
            val p = if (startKey && j == k) params.wordStartKeyPrior else if (j == delete) params.deletePrior
                else if (j > delete && !endsWordKey[j]) params.otherPrior
                else keep * ((1f - ctxWeight) * lexP[j] + ctxWeight * ctxP[j])
            val score = -spatialCost(j, x, y) + priorWeight * ln(max(p, params.minPrior))
            if (score > bestScore) { bestScore = score; best = j }
        }
        if (best == k) return rectKey
        return when {
            best == space -> SPACE
            best == delete -> DELETE
            best > delete -> otherLabels[best - delete - 1]
            k < n && Character.isUpperCase(rectKey[0]) -> upper[best]
            else -> lower[best]
        }
    }

    /** Squared-distance cost of a touch for key [j]: a Gaussian around its centre (the centre line for the wide keys). */
    private fun spatialCost(j: Int, x: Float, y: Float): Float {
        if (j < n) {
            val dx = x - meanX[j]; val dy = y - meanY[j]
            return dx * dx * invVarX + dy * dy * invVarY
        }
        val r = (if (j == space) spaceRect else if (j == delete) deleteRect else otherRects[j - delete - 1]) ?: return Float.POSITIVE_INFINITY
        val half = keyWidth / 2
        val left = min(r[0] + half, (r[0] + r[2]) / 2) + offX
        val right = max(r[2] - half, (r[0] + r[2]) / 2) + offX
        val dx = if (x < left) left - x else if (x > right) x - right else 0f
        val shift = (if (j == space) params.spaceShift else if (j == delete) params.deleteShift else otherShift[j - delete - 1]) * keyWidth
        val dy = y - ((r[1] + r[3]) / 2 + offY + shift)
        return dx * dx * invVarX + dy * dy * invVarY
    }

    /**
     * Whether the space bar competes with a tap at [x] on other key [o] inside the word [typed]: the key borders the bar
     * ([otherSpaceSide]), the tap is within [TapResolveParams.punctuationSpaceEdge] of that edge, and the word so far is
     * letters (a number keeps its decimal point).
     */
    private fun spaceRival(o: Int, x: Float, typed: String): Boolean {
        if (!params.punctuationSpace || spaceRect == null) return false
        val r = otherRects[o]
        val d = when (otherSpaceSide[o]) { 1 -> x - r[0]; 2 -> r[2] - x; else -> return false }
        if (d < 0f || d > params.punctuationSpaceEdge * keyWidth) return false
        for (c in typed) if (!Character.isLetter(c) && c != '\'' && c != '\u2019') return false
        return true
    }

    /**
     * A tap on punctuation key [k] near the space bar after [typed]: the bar against the key, with the wide keys'
     * spatial model. Priors: the word ends here ([wordEndProb]) and the sentence goes on (bar) or ends there (key, at
     * least [TapResolveParams.punctuationSpaceKeyPrior]), the latter from [punctuationProb].
     */
    private fun againstSpace(k: Int, context: Context, typed: String, x: Float, y: Float, rectKey: String): String {
        val pe = wordEndProb(typed)
        val s = punctuationProb?.invoke(context, typed, rectKey)?.takeIf { !it.isNaN() }?.coerceIn(0f, 1f) ?: 0f
        val w = params.wideKeyPriorWeight
        val bar = -spatialCost(space, x, y) + w * ln(max(pe * (1f - s), params.minPrior))
        val key = -spatialCost(k, x, y) + w * ln(max(pe * s, params.punctuationSpaceKeyPrior))
        return if (bar > key) SPACE else rectKey
    }

    /**
     * How likely [typed] is a whole word rather than the start of a longer one: the best lexicon word equal to it
     * against the best word at or below its trie node (1 for "little" or "the", 0 for "litt").
     */
    private fun wordEndProb(typed: String): Float {
        var best = 0f
        for (lang in langs) {
            val lex = lang.lexicon
            val node = walk(lex, typed)
            if (node < 0) continue
            val w0 = lex.wordsStart(node)
            for (id in w0 until w0 + lex.wordsCount(node)) best = max(best, exp(lex.prior(id) - lex.maxLogPrior(node)))
        }
        return best.coerceAtMost(1f)
    }

    /** Distance of a touch from the edges of other key [o] that face letter keys (infinite when none does). */
    private fun otherEdgeDistance(o: Int, x: Float, y: Float): Float {
        val r = otherRects[o]; val e = otherEdges[o]
        var d = Float.POSITIVE_INFINITY
        if (e[0]) d = min(d, y - r[1])
        if (e[1]) d = min(d, x - r[0])
        if (e[2]) d = min(d, r[2] - x)
        return d
    }

    /**
     * Fills [lexP] for [cand] from the best word below each trie child of the prefix (the space bar: the prefix
     * itself as a word); false when no lexicon has the prefix.
     */
    private fun lexiconPrior(typed: String, cand: IntArray): Boolean {
        for (j in cand) lexP[j] = 0f
        var any = false
        for (lang in langs) {
            val lex = lang.lexicon
            val node = walk(lex, typed)
            if (node < 0) continue
            any = true
            for (j in cand) {
                if (endsWordKey[j]) {
                    val w0 = lex.wordsStart(node)
                    for (id in w0 until w0 + lex.wordsCount(node)) lexP[j] = max(lexP[j], exp(lex.prior(id) + lang.logWeight))
                    continue
                }
                if (j >= n) continue
                val s = lang.keySym[j]
                if (s < 0) continue
                val child = childOrAfterSilent(lex, node, s)
                if (child >= 0) lexP[j] = max(lexP[j], exp(lex.maxLogPrior(child) + lang.logWeight))
            }
        }
        if (!any) return false
        return normalise(lexP, cand)
    }

    /** Fills [lexP] from the letter-trigram model (the prefix is in no lexicon: a typo or a new word). */
    private fun letterPrior(typed: String, cand: IntArray): Boolean {
        val lang = langs.firstOrNull { it.letters != null } ?: return false
        val lm = lang.letters!!
        for (j in cand) lexP[j] = when {
            endsWordKey[j] -> exp(lm.endLogProb(typed))
            j >= n -> 0f
            lang.keySym[j] >= 0 -> exp(lm.nextLogProb(typed, labels[j]))
            else -> 0f
        }
        return normalise(lexP, cand)
    }

    /** Fills [ctxP] from the published predictions for this context; false when none applies. */
    private fun contextPrior(context: Context, typed: String, cand: IntArray): Boolean {
        val cp = contextPrior ?: return false
        if (cp.sentenceStart != context.sentenceStart || !sameWords(cp.words, context.words)) return false
        for (j in cand) ctxP[j] = 0f
        var spaceCand = false
        for (j in cand) if (endsWordKey[j]) spaceCand = true
        val plen = typed.length
        for (i in cp.preds.indices) {
            val w = cp.preds[i]
            if (w.length < plen || (w.length == plen && !spaceCand)) continue
            var match = true
            for (c in 0 until plen) if (w[c] != lowerChar(typed[c])) { match = false; break }
            if (!match) continue
            if (w.length == plen) { for (j in cand) if (endsWordKey[j]) ctxP[j] += cp.probs[i]; continue }
            var next = w[plen]
            if (rules.isSilent(next) && w.length > plen + 1) next = w[plen + 1]
            for (j in cand) if (j < n && labels[j] == next) { ctxP[j] += cp.probs[i]; break }
        }
        return normalise(ctxP, cand)
    }

    private fun normalise(p: FloatArray, cand: IntArray): Boolean {
        var z = 0f
        for (j in cand) z += p[j]
        if (z <= 0f) return false
        for (j in cand) p[j] /= z
        return true
    }

    private fun walk(lex: Lexicon, typed: String): Int {
        var node = lex.root
        for (ch in typed) {
            val s = lex.symbolOf(lowerChar(ch))
            if (s < 0) return -1
            node = lex.child(node, s)
            if (node < 0) return -1
        }
        return node
    }

    /** The child of [node] for symbol [s], also through one silent character ("don" + 't' -> "don't"). */
    private fun childOrAfterSilent(lex: Lexicon, node: Int, s: Int): Int {
        val c = lex.child(node, s)
        if (c >= 0) return c
        val first = lex.firstChildOf(node)
        for (i in first until first + lex.childCountOf(node)) {
            if (rules.isSilent(lex.alphabet[lex.symOf(i)])) {
                val cc = lex.child(i, s)
                if (cc >= 0) return cc
            }
        }
        return -1
    }

    private fun sameWords(a: List<String>, b: List<String>): Boolean {
        if (a.size != b.size) return false
        for (i in a.indices) if (a[i] != b[i]) return false
        return true
    }

    private fun indexOf(c: Char): Int {
        for (i in 0 until n) if (labels[i] == c) return i
        return -1
    }

    private fun lowerChar(c: Char): Char = when {
        dotted && c == 'I' -> 'ı'
        dotted && c == 'İ' -> 'i'
        else -> Character.toLowerCase(c)
    }

    companion object {
        /** Labels of the space bar and the delete key, as in [dev.pboard.engine.KeyboardGeometry] and [resolve]'s results. */
        const val SPACE = "space"
        const val DELETE = "delete"
        /** Other keys that end a word as the space bar does. */
        private val WORD_ENDERS = setOf(".", ",")
        /** Other keys resolved at a word's start too ([TapResolveParams.wordStart]). */
        private val WORD_START_KEYS = setOf(".", "shift")
    }
}
