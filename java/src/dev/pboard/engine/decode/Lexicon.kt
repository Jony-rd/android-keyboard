package dev.pboard.engine.decode

import java.util.PriorityQueue

/** The vocabulary entry type of the API ([dev.pboard.engine.LexiconEntry]); kept here for existing imports. */
typealias LexiconEntry = dev.pboard.engine.LexiconEntry

/**
 * Compact, immutable trie over the locale-lower-cased spellings of a lexicon ("I'm" -> "i'm",
 * "iPhone" -> "iphone", "Déjà" -> "déjà"). It is independent of any keyboard layout: the decoders map
 * lexicon characters to keys at run time ([CharKeyMap]), so one trie serves QWERTY, AZERTY, … alike.
 *
 * Nodes are numbered breadth-first so each node's children are contiguous; per-node data lives in
 * primitive arrays (≈ 17 bytes/node). Edge labels are indices into this lexicon's [alphabet]. Each node
 * stores the best prior in its subtree ([maxLogPrior]), used as an admissible LM look-ahead and for
 * best-first completion. Terminal nodes own a run of canonical words with that lower-case spelling, best
 * prior first ("us" -> [us, US]).
 */
class Lexicon private constructor(
    val rules: TextRules,
    private val words: Array<String>,
    private val priors: FloatArray,
    /** Distinct lower-case characters used by the lexicon; edge labels index into it. */
    val alphabet: CharArray,
    private val nodeSym: ShortArray,
    private val firstChild: IntArray,
    private val childCount: ShortArray,
    private val parent: IntArray,
    private val depth: ByteArray,
    private val maxPrior: FloatArray,
    private val wordStart: IntArray,
    private val nodeWordCount: ByteArray,
    private val variantGroupOf: IntArray,
    private val variantGroups: Array<IntArray>,
    private val offensiveIds: java.util.BitSet,
) {
    /** Whether word [id] is flagged potentially offensive. */
    fun isOffensive(id: Int): Boolean = offensiveIds.get(id)

    private val byCanonical: HashMap<String, Int> = HashMap<String, Int>(words.size * 2).also { m ->
        for (i in words.indices) m[words[i]] = i
    }
    private val symIndex: HashMap<Char, Int> = HashMap<Char, Int>().also { m -> for (i in alphabet.indices) m[alphabet[i]] = i }
    private val skeletons = arrayOfNulls<String>(words.size)

    val size: Int get() = words.size
    val nodeCount: Int get() = nodeSym.size
    val root: Int get() = 0

    fun word(id: Int): String = words[id]
    fun prior(id: Int): Float = priors[id]

    /** Word id of an exact canonical spelling, or -1. */
    fun idOf(canonical: String): Int = byCanonical[canonical] ?: -1
    fun contains(canonical: String): Boolean = byCanonical.containsKey(canonical)

    /** Alphabet index of a lower-case character, or -1 if no word uses it. */
    fun symbolOf(c: Char): Int = symIndex[c] ?: -1

    fun symOf(node: Int): Int = nodeSym[node].toInt()
    fun depthOf(node: Int): Int = depth[node].toInt()
    fun parentOf(node: Int): Int = parent[node]
    fun firstChildOf(node: Int): Int = firstChild[node]
    fun childCountOf(node: Int): Int = childCount[node].toInt()
    fun maxLogPrior(node: Int): Float = maxPrior[node]
    fun isTerminal(node: Int): Boolean = nodeWordCount[node] > 0
    fun wordsStart(node: Int): Int = wordStart[node]
    fun wordsCount(node: Int): Int = nodeWordCount[node].toInt()

    fun child(node: Int, sym: Int): Int {
        val f = firstChild[node]
        val n = childCount[node].toInt()
        for (k in 0 until n) {
            val c = nodeSym[f + k].toInt()
            if (c == sym) return f + k
            if (c > sym) return -1
        }
        return -1
    }

    /** Node reached by the lower-case spelling [path], or -1. */
    fun find(path: CharSequence): Int {
        var node = 0
        for (ch in path) {
            val s = symbolOf(ch)
            if (s < 0) return -1
            node = child(node, s)
            if (node < 0) return -1
        }
        return node
    }

    /** The lower-case spelling of the path from the root to [node]. */
    fun pathOf(node: Int): String {
        val d = depth[node].toInt()
        val buf = CharArray(d)
        var n = node
        for (k in d - 1 downTo 0) { buf[k] = alphabet[nodeSym[n].toInt()]; n = parent[n] }
        return String(buf)
    }

    /** Word ids at [node] (same lower-case spelling), best prior first. */
    fun wordsAt(node: Int): IntRange {
        val s = wordStart[node]
        return s until s + nodeWordCount[node]
    }

    /** [TextRules.skeleton] of word [id] (cached). */
    /** Other-dialect spellings of word [id] (different lower-case spelling, same variant key). */
    fun variantsOf(id: Int): IntArray {
        val g = variantGroupOf[id]
        if (g < 0) return EMPTY
        val lower = rules.lowercase(words[id])
        return variantGroups[g].filter { rules.lowercase(words[it]) != lower }.toIntArray()
    }

    fun skeletonOf(id: Int): String = skeletons[id] ?: rules.skeleton(words[id]).also { skeletons[id] = it }

    /**
     * Best-first enumeration of the [limit] highest-prior words strictly below [node] (longer than its
     * path). Uses [maxLogPrior] so only O(limit · depth · fan-out) nodes are touched.
     */
    fun topCompletions(node: Int, limit: Int, out: MutableList<Int> = ArrayList(limit)): MutableList<Int> {
        if (limit <= 0) return out
        val pq = PriorityQueue<Long>(32) { a, b -> java.lang.Float.compare(keyPrior(b), keyPrior(a)) }
        val f = firstChild[node]
        for (k in 0 until childCount[node]) pq.add((f + k).toLong())
        while (pq.isNotEmpty() && out.size < limit) {
            val e = pq.poll()
            val v = (e and 0xffffffffL).toInt()
            if (e ushr 63 == 1L) { out.add(v); continue }
            for (w in wordsAt(v)) pq.add((1L shl 63) or w.toLong())
            val fc = firstChild[v]
            for (k in 0 until childCount[v]) pq.add((fc + k).toLong())
        }
        return out
    }

    private fun keyPrior(e: Long): Float {
        val v = (e and 0xffffffffL).toInt()
        return if (e ushr 63 == 1L) priors[v] else maxPrior[v]
    }

    companion object {
        const val MAX_DEPTH = 48
        private val EMPTY = IntArray(0)

        /**
         * Builds the trie. Duplicate canonical spellings keep their best prior; words containing characters
         * outside the Basic Multilingual Plane or longer than [MAX_DEPTH] are skipped.
         */
        fun build(entries: Iterable<LexiconEntry>, rules: TextRules): Lexicon {
            val best = HashMap<String, LexiconEntry>()
            for (e in entries) {
                if (e.word.isEmpty()) continue
                val prev = best[e.word]
                if (prev == null || e.logPrior > prev.logPrior) best[e.word] = e
            }
            class Item(val word: String, val path: String, val prior: Float, val variantKey: String?, val offensive: Boolean)
            val items = ArrayList<Item>(best.size)
            val chars = java.util.TreeSet<Char>()
            for ((w, e) in best) {
                val path = rules.lowercase(w)
                if (path.length > MAX_DEPTH || path.any { Character.isSurrogate(it) || it.isWhitespace() }) continue
                items.add(Item(w, path, e.logPrior, e.variantKey, e.offensive))
                for (c in path) chars.add(c)
            }
            val alphabet = chars.toCharArray()
            require(alphabet.size < Short.MAX_VALUE) { "alphabet too large" }
            val symOf = HashMap<Char, Int>().also { m -> for (i in alphabet.indices) m[alphabet[i]] = i }
            items.sortWith(compareBy<Item> { it.path }.thenByDescending { it.prior }.thenBy { it.word })
            val paths = Array(items.size) { items[it].path }

            var cap = 1024
            var nSym = ShortArray(cap); var nFirst = IntArray(cap); var nCount = ShortArray(cap)
            var nParent = IntArray(cap); var nDepth = ByteArray(cap); var nWStart = IntArray(cap); var nWCount = ByteArray(cap)
            var rLo = IntArray(cap); var rHi = IntArray(cap)
            fun grow() {
                cap *= 2
                nSym = nSym.copyOf(cap); nFirst = nFirst.copyOf(cap); nCount = nCount.copyOf(cap)
                nParent = nParent.copyOf(cap); nDepth = nDepth.copyOf(cap); nWStart = nWStart.copyOf(cap)
                nWCount = nWCount.copyOf(cap); rLo = rLo.copyOf(cap); rHi = rHi.copyOf(cap)
            }
            var count = 1
            rLo[0] = 0; rHi[0] = paths.size; nParent[0] = -1; nSym[0] = -1
            var id = 0
            while (id < count) {
                val lo = rLo[id]; val hi = rHi[id]; val d = nDepth[id].toInt()
                var j = lo
                while (j < hi && paths[j].length == d) j++
                nWStart[id] = lo
                nWCount[id] = minOf(j - lo, 127).toByte()
                nFirst[id] = count
                while (j < hi) {
                    val c = paths[j][d]
                    var k = j
                    while (k < hi && paths[k][d] == c) k++
                    if (count == cap) grow()
                    nSym[count] = symOf[c]!!.toShort(); nParent[count] = id; nDepth[count] = (d + 1).toByte()
                    rLo[count] = j; rHi[count] = k
                    count++
                    j = k
                }
                nCount[id] = (count - nFirst[id]).toShort()
                id++
            }
            val priors = FloatArray(items.size) { items[it].prior }
            val maxP = FloatArray(count) { Float.NEGATIVE_INFINITY }
            for (n in count - 1 downTo 0) {
                if (nWCount[n] > 0) maxP[n] = maxOf(maxP[n], priors[nWStart[n]])
                if (n > 0) { val p = nParent[n]; if (maxP[n] > maxP[p]) maxP[p] = maxP[n] }
            }
            // Dialect-variant groups: words sharing a variant key but spelt differently.
            val byKey = HashMap<String, MutableList<Int>>()
            for ((i, it) in items.withIndex()) byKey.getOrPut(it.variantKey ?: rules.variantKey(it.word)) { ArrayList(1) }.add(i)
            val groupOf = IntArray(items.size) { -1 }
            val groups = ArrayList<IntArray>()
            for (members in byKey.values) {
                if (members.size < 2 || members.map { items[it].path }.distinct().size < 2) continue
                for (m in members) groupOf[m] = groups.size
                groups.add(members.toIntArray())
            }
            return Lexicon(
                rules, Array(items.size) { items[it].word }, priors, alphabet,
                nSym.copyOf(count), nFirst.copyOf(count), nCount.copyOf(count), nParent.copyOf(count),
                nDepth.copyOf(count), maxP, nWStart.copyOf(count), nWCount.copyOf(count), groupOf, groups.toTypedArray(),
                java.util.BitSet(items.size).also { bs -> for ((i, it) in items.withIndex()) if (it.offensive) bs.set(i) },
            )
        }
    }
}
