package dev.pboard.engine.decode

import kotlin.math.ln
import kotlin.math.min

/**
 * Letter-trigram model of a lexicon's spellings (frequency-weighted, padded with word start and end), read
 * off the trie: the probability mass of the words below each node flows along its edge. It judges whether an
 * unknown typed string is a plausible spelling of a word of the language (a name such as "Mikey") rather
 * than a garbled one ("Hrllo": no common word starts "hr"). Language-agnostic: it only uses the lexicon's
 * alphabet, keeping the [MAX_SYMBOLS] most used characters and pooling the rest (rare accented letters).
 */
class LetterModel(lexicon: Lexicon) {
    private val rules = lexicon.rules
    private val sym: (Char) -> Int = lexicon::symbolOf
    /** Lexicon symbol -> model symbol (the rarest symbols share one). */
    private val map: IntArray
    private val start: Int
    private val end: Int
    private val dim: Int
    private val tri: FloatArray
    private val triCtx: FloatArray
    private val bi: FloatArray
    private val biCtx: FloatArray

    init {
        val nodes = lexicon.nodeCount
        val root = lexicon.root
        val own = FloatArray(nodes)
        val mass = FloatArray(nodes)
        val symMass = DoubleArray(lexicon.alphabet.size)
        // Children come after their parent (BFS order): a node's subtree mass is complete when it is visited.
        for (node in nodes - 1 downTo 1) {
            if (lexicon.isTerminal(node)) {
                var m = 0f
                for (id in lexicon.wordsAt(node)) m += Math.exp(lexicon.prior(id).toDouble()).toFloat()
                own[node] = m
                mass[node] += m
            }
            symMass[lexicon.symOf(node)] += mass[node].toDouble()
            mass[lexicon.parentOf(node)] += mass[node]
        }
        val kept = symMass.indices.sortedByDescending { symMass[it] }.take(MAX_SYMBOLS)
        val other = kept.size
        map = IntArray(symMass.size) { other }
        kept.forEachIndexed { i, s -> map[s] = i }
        start = other + 1
        end = other + 2
        dim = other + 3
        tri = FloatArray(dim * dim * dim)
        triCtx = FloatArray(dim * dim)
        bi = FloatArray(dim * dim)
        biCtx = FloatArray(dim)
        for (node in 1 until nodes) {
            val p = lexicon.parentOf(node)
            val c = map[lexicon.symOf(node)]
            val b = if (p == root) start else map[lexicon.symOf(p)]
            val a = if (p == root || lexicon.parentOf(p) == root) start else map[lexicon.symOf(lexicon.parentOf(p))]
            add(a, b, c, mass[node])
            if (own[node] > 0f) add(b, c, end, own[node])
        }
    }

    private fun add(a: Int, b: Int, c: Int, m: Float) {
        if (m <= 0f) return
        val ctx = a * dim + b
        tri[ctx * dim + c] += m
        triCtx[ctx] += m
        bi[b * dim + c] += m
        biCtx[b] += m
    }

    /** log P(c | a, b): the trigram estimate interpolated with the bigram one (unseen contexts get the bigram share only). */
    private fun logP(a: Int, b: Int, c: Int): Float {
        val ctx = triCtx[a * dim + b]
        val pTri = if (ctx > 0f) tri[(a * dim + b) * dim + c].toDouble() / ctx else 0.0
        val pBi = if (biCtx[b] > 0f) bi[b * dim + c].toDouble() / biCtx[b] else 0.0
        return ln(maxOf(LAMBDA * pTri + (1 - LAMBDA) * pBi, 1e-12)).toFloat()
    }

    /**
     * The least likely letter transition in [word] (from the word start through to its end), in nats; a
     * character the lexicon never uses counts as impossible. Silent characters (apostrophes, hyphens) are skipped.
     */
    fun weakestTransition(word: String): Float {
        var a = start
        var b = start
        var worst = 0f
        for (ch in rules.lowercase(word)) {
            if (rules.isSilent(ch)) continue
            val s = sym(ch)
            if (s < 0) return Float.NEGATIVE_INFINITY
            val m = map[s]
            worst = min(worst, logP(a, b, m))
            a = b; b = m
        }
        return min(worst, logP(a, b, end))
    }

    /** log P([next] | the last two letters of [prefix]); allocation-free (lower-cases per character). */
    fun nextLogProb(prefix: CharSequence, next: Char): Float {
        val ctx = context(prefix)
        val s = sym(Character.toLowerCase(next))
        if (s < 0) return Float.NEGATIVE_INFINITY
        return logP(ctx / dim, ctx % dim, map[s])
    }

    /** log P(the word ends | the last two letters of [prefix]); allocation-free. */
    fun endLogProb(prefix: CharSequence): Float {
        val ctx = context(prefix)
        return logP(ctx / dim, ctx % dim, end)
    }

    /** The (a, b) letter context after [prefix], as a * dim + b. */
    private fun context(prefix: CharSequence): Int {
        var a = start
        var b = start
        for (i in prefix.indices) {
            val ch = Character.toLowerCase(prefix[i])
            if (rules.isSilent(ch)) continue
            val s = sym(ch)
            a = b
            b = if (s < 0) start - 1 else map[s] // unknown characters share the pooled "other" symbol
        }
        return a * dim + b
    }

    private companion object {
        const val LAMBDA = 0.99
        /** 48 + other + start + end = 51 symbols: 51³ floats ≈ 0.5 MB. */
        const val MAX_SYMBOLS = 48
    }
}
