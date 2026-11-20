package dev.pboard.engine.lm

import dev.pboard.engine.LanguageModel
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max

/**
 * Back-off n-gram [LanguageModel] over a memory-mapped [LmPack], with an optional regional [DialectOverlay]
 * (en_GB / en_US spelling and vocabulary preferences) and an optional on-device [UserNgramModel].
 *
 * Words are case-insensitive on input ("london" scores as "London"; a capitalised sentence-initial word
 * scores as its lowercase form when that is more common) and predictions are returned in canonical
 * capitalisation. Probabilities of a word w of n-gram class g (a word or a spelling class such as
 * {color, colour}) are `log P(g | context) + inClass(w) + overlay.adjust(w) - overlay.logZ`, linearly
 * interpolated with the user model when it has evidence.
 *
 * Context entries may carry punctuation ("Hello,", "today.", ","): sentence-final punctuation starts a new
 * sentence; ',', ';', ':' become the clause token. A context shorter than order-1 words is taken to start at
 * the beginning of the field (sentence start).
 *
 * Offensive words (pack/overlay flag) are never returned by [predict] unless [blockOffensive] is false, the
 * user typed the complete word as the prefix, or the user has used the word repeatedly ([UserNgramModel]).
 * [logProb] stays an honest probability for all words so that decoders keep literally typed words.
 *
 * Thread-safety: queries may run concurrently; [observe]/[setDialect] should come from one thread.
 */
class NgramLanguageModel(
    val pack: LmPack,
    overlay: DialectOverlay? = null,
    val user: UserNgramModel? = null,
) : LanguageModel {

    @Volatile var overlay: DialectOverlay? = null
        private set

    /** Gboard's "Block offensive words" setting. */
    @Volatile var blockOffensive: Boolean = true

    private val order = pack.order
    private val rules = pack.rules

    fun setDialect(o: DialectOverlay?) {
        if (o != null) {
            require(o.baseBuildId == pack.buildId) { "overlay ${o.locale} was built for a different base pack" }
            require(o.size == pack.nLex) { "overlay size mismatch" }
        }
        overlay = o
        bounds = computeBounds(o)
    }

    /**
     * Per-class score bounds for the top-k search, for one overlay. A word's score is
     * classLp + inClass(word) + adjust(word) - logZ, so classBest[g] = max over the members of class g of
     * inClass + adjust bounds every member: a class with classLp + classBest[g] - logZ at or below the heap
     * minimum cannot enter, and is skipped without visiting its members. unigramBound[i] is the largest
     * unigram logp + classBest over topUnigrams[i..], so the unigram fallback loop can stop as soon as no
     * remaining class can enter. Skipping only offers the heap would reject keeps results bit-identical.
     */
    private class Bounds(val overlay: DialectOverlay?, val classBest: FloatArray, val unigramBound: FloatArray)

    @Volatile private var bounds: Bounds? = null

    private fun computeBounds(o: DialectOverlay?): Bounds {
        val best = FloatArray(pack.nIds) { Float.NEGATIVE_INFINITY }
        for (lex in 0 until pack.nLex) {
            val g = pack.ngramIdOf(lex)
            val v = pack.inClassLogProb(lex) + (o?.adjust(lex) ?: 0f)
            if (v > best[g]) best[g] = v
        }
        val top = topUnigrams
        val ub = FloatArray(top.size + 1) { Float.NEGATIVE_INFINITY }
        for (i in top.size - 1 downTo 0) ub[i] = maxOf(ub[i + 1], pack.unigramLogProb(top[i]) + best[top[i]])
        return Bounds(o, best, ub)
    }

    private fun boundsFor(o: DialectOverlay?): Bounds {
        bounds?.let { if (it.overlay === o) return it }
        return computeBounds(o).also { bounds = it }
    }

    /** n-gram ids (words only) by decreasing unigram probability: the fallback list for next-word prediction. */
    private val topUnigrams: IntArray by lazy {
        val keys = LongArray(pack.nIds - LmPack.NSPECIAL) { i ->
            val g = i + LmPack.NSPECIAL
            // sort ascending on -logp (fixed point), id in the low bits
            ((-pack.unigramLogProb(g) * 1000f).toLong() shl 20) or g.toLong()
        }
        keys.sort()
        IntArray(keys.size) { (keys[it] and 0xFFFFF).toInt() }
    }

    // after every property setDialect touches (bounds, topUnigrams) has been initialised
    init {
        setDialect(overlay)
    }

    private class Scratch(nIds: Int) {
        val sb = StringBuilder(32)
        val sb2 = StringBuilder(32)
        val ctx = IntArray(16)
        val lastIds = IntArray(16)
        var lastN = 0
        var lastContext: List<String>? = null
        var lastOverlay: DialectOverlay? = null
        val stamp = IntArray(nIds)
        var gen = 0
        val heapScore = FloatArray(MAX_HEAP)
        val heapId = IntArray(MAX_HEAP)
        var heapSize = 0
        var heapCap = 0
    }

    private val scratch = ThreadLocal.withInitial { Scratch(pack.nIds) }

    // ============================================================================================ words

    /**
     * Lexicon id of [word]: exact spelling if present (a capitalised sentence-initial word prefers its more
     * frequent lowercase variant), otherwise the most frequent case variant; -1 if unknown.
     */
    fun lexIdOf(word: String, sentenceInitial: Boolean = false): Int {
        if (word.isEmpty()) return -1
        val s = scratch.get()
        val lower = rules.lower(word)
        val g = pack.findLowerGroup(lower, s.sb)
        if (g < 0) return -1
        var id = g
        var exact = -1
        while (id < pack.nLex && pack.lowerEquals(id, lower)) {
            if (pack.wordEquals(id, word)) {
                exact = id
                break
            }
            id++
        }
        if (exact < 0) return g
        if (sentenceInitial && exact != g && Character.isUpperCase(word.codePointAt(0)) && pack.wordEquals(g, lower)) return g
        return exact
    }

    fun word(lexId: Int): String = pack.word(lexId)

    fun isOffensive(word: String): Boolean {
        val id = lexIdOf(word)
        return id >= 0 && isOffensiveId(id)
    }

    /** Offensive flag of lexicon entry [lexId] (pack flag or dialect overlay flag). */
    fun isOffensiveId(lexId: Int): Boolean =
        (pack.flags(lexId) or (overlay?.extraFlags(lexId) ?: 0)) and LmPack.F_OFFENSIVE != 0

    /**
     * Context-free natural-log prior of lexicon entry [lexId] in the active dialect: its relative frequency in
     * the training data with the dialect's spelling and vocabulary preferences applied.
     */
    fun wordLogPrior(lexId: Int): Float {
        val o = overlay
        return pack.logFreq(lexId) + (if (o == null) 0f else o.adjust(lexId) - o.logZ)
    }

    /** True if committing [word] should not be followed by a space (elided clitics such as French "l'"). */
    fun attachesToNext(word: String): Boolean {
        val id = lexIdOf(word)
        return id >= 0 && pack.flags(id) and LmPack.F_NO_SPACE_AFTER != 0
    }

    override fun isKnown(word: String): Boolean {
        if (word.isEmpty()) return false
        if (isNumber(word)) return true
        if (lexIdOf(word) >= 0) return true
        if (user != null && user.isKnown(word)) return true
        return pack.isCompounding && decompose(word) != null
    }

    // ============================================================================================ context

    /** Fills s.ctx with up to order-1 n-gram ids (oldest first); returns their count. */
    private fun contextIds(context: List<String>, s: Scratch): Int {
        // decoders score many candidate words in the same context: reuse the last resolution
        val last = s.lastContext
        if (last != null && last == context && s.lastOverlay === overlay) {
            System.arraycopy(s.lastIds, 0, s.ctx, 0, s.lastN)
            return s.lastN
        }
        val n = resolveContext(context, s)
        s.lastContext = ArrayList(context)
        s.lastOverlay = overlay
        System.arraycopy(s.ctx, 0, s.lastIds, 0, n)
        s.lastN = n
        return n
    }

    private fun resolveContext(context: List<String>, s: Scratch): Int {
        val maxLen = order - 1
        val rev = ArrayList<String>(maxLen + 1)  // most recent first
        var boundary = false
        var i = context.size - 1
        outer@ while (i >= 0 && rev.size < maxLen) {
            val tokens = splitPunct(context[i])
            for (t in tokens.indices.reversed()) {
                val tok = tokens[t]
                when {
                    tok.isEmpty() -> {}
                    isSentenceEnd(tok) -> { boundary = true; break@outer }
                    tok == "," || tok == ";" || tok == ":" -> if (rev.isEmpty() || rev.last() != ",") rev.add(",")
                    else -> rev.add(tok)
                }
                if (rev.size >= maxLen) break@outer
            }
            i--
        }
        val atStart = rev.size < maxLen && (boundary || i < 0)
        var n = 0
        if (atStart) s.ctx[n++] = LmPack.BOS
        for (j in rev.indices.reversed()) {
            val tok = rev[j]
            s.ctx[n++] = if (tok == ",") LmPack.COMMA else wordId(tok, atStart && j == rev.size - 1)
        }
        return n
    }

    private fun isSentenceEnd(tok: String) = tok.all { it == '.' || it == '!' || it == '?' || it == '…' }

    /**
     * n-gram ids of the current sentence: `<s>` followed by the context tokens since the last sentence end
     * (at most [maxLen] of the most recent ones). Used by sentence-level models such as [NeuralLm].
     */
    fun sentenceIds(context: List<String>, maxLen: Int = 48): IntArray {
        val rev = ArrayList<String>()
        var i = context.size - 1
        outer@ while (i >= 0 && rev.size < maxLen) {
            val tokens = splitPunct(context[i])
            for (t in tokens.indices.reversed()) {
                val tok = tokens[t]
                when {
                    tok.isEmpty() -> {}
                    isSentenceEnd(tok) -> break@outer
                    tok == "," || tok == ";" || tok == ":" -> if (rev.isEmpty() || rev.last() != ",") rev.add(",")
                    else -> rev.add(tok)
                }
                if (rev.size >= maxLen) break@outer
            }
            i--
        }
        val out = IntArray(rev.size + 1)
        out[0] = LmPack.BOS
        for (j in rev.indices) {
            val tok = rev[rev.size - 1 - j]
            out[j + 1] = if (tok == ",") LmPack.COMMA else wordId(tok, j == 0)
        }
        return out
    }

    /** Lexicon id that [logProb] uses for [word] after [context] (same truecasing rule), or -1. */
    fun lexIdIn(context: List<String>, word: String): Int {
        val s = scratch.get()
        val m = contextIds(context, s)
        return lexIdOf(word, m > 0 && s.ctx[m - 1] == LmPack.BOS)
    }

    /** log P(word | its class) in the active dialect: in-class share + overlay adjustment - normaliser. */
    fun wordTerm(lexId: Int): Float = wordLogProb(lexId, 0f)

    private fun wordId(tok: String, initial: Boolean): Int {
        if (isNumber(tok)) return LmPack.NUM
        val id = lexIdOf(tok, initial)
        return if (id < 0) LmPack.UNK else pack.ngramIdOf(id)
    }

    // ============================================================================================ scoring

    /** log P(g | ctx[0, m)) in the base model (Katz back-off over the trie). */
    fun ngramLogProb(ctx: IntArray, m: Int, g: Int): Float {
        var bow = 0f
        for (start in 0..m) {
            val len = m - start
            if (len == 0) return bow + pack.unigramLogProb(g)
            if (len >= order) continue
            val node = pack.find(ctx, start, m)
            if (node < 0) continue
            val lo = pack.childStart(len, node)
            val hi = pack.childStart(len, node + 1)
            if (hi > lo) {
                val c = pack.findChild(len + 1, lo, hi, g)
                if (c >= 0) return bow + pack.logProbAt(len + 1, c)
            }
            bow += pack.bowAt(len, node)
        }
        return bow + pack.unigramLogProb(g)
    }

    private fun wordLogProb(lexId: Int, classLogProb: Float): Float {
        val o = overlay
        return classLogProb + pack.inClassLogProb(lexId) + (if (o == null) 0f else o.adjust(lexId) - o.logZ)
    }

    /** Static (pack + dialect) log probability, without the user model. */
    fun staticLogProb(context: List<String>, word: String): Float {
        val s = scratch.get()
        val m = contextIds(context, s)
        val initial = m > 0 && s.ctx[m - 1] == LmPack.BOS
        if (isNumber(word)) return ngramLogProb(s.ctx, m, LmPack.NUM)
        val lex = lexIdOf(word, initial)
        if (lex >= 0) return wordLogProb(lex, ngramLogProb(s.ctx, m, pack.ngramIdOf(lex)))
        if (pack.isCompounding) {
            val parts = decompose(word)
            if (parts != null) {
                // first part in context, the rest as unigram continuations; a closed compound is one word
                var lp = wordLogProb(parts[0], ngramLogProb(s.ctx, m, pack.ngramIdOf(parts[0])))
                for (j in 1 until parts.size) lp += wordLogProb(parts[j], pack.unigramLogProb(pack.ngramIdOf(parts[j]))) + COMPOUND_PENALTY
                return lp
            }
        }
        return ngramLogProb(s.ctx, m, LmPack.UNK) + pack.oovPenalty
    }

    override fun logProb(context: List<String>, word: String): Float {
        val lp = staticLogProb(context, word)
        val u = user ?: return lp
        return mixWithUser(context, word, lp, u)
    }

    private fun mixWithUser(context: List<String>, word: String, staticLp: Float, u: UserNgramModel): Float {
        val canon = canonical(context)
        val w = canonicalWord(word)
        val lam = u.weight(canon)
        if (lam <= 0f) return staticLp
        val pu = u.prob(canon, w)
        val ps = exp(staticLp.toDouble())
        val p = (1 - lam) * ps + lam * pu
        return ln(max(p, 1e-300)).toFloat()
    }

    // ============================================================================================ prediction

    override fun predict(context: List<String>, prefix: String, limit: Int): List<Pair<String, Float>> {
        if (limit <= 0) return emptyList()
        val s = scratch.get()
        val m = contextIds(context, s)
        val extra = if (user != null) 8 else 0
        val cap = minOf(MAX_HEAP, limit + extra)
        heapReset(s, cap)
        s.gen++
        if (s.gen == Int.MAX_VALUE) { s.stamp.fill(0); s.gen = 1 }

        var lo = 0
        var hi = pack.nLex
        val folded = if (prefix.isEmpty()) "" else rules.fold(prefix)
        if (prefix.isNotEmpty()) {
            lo = prefixLower(folded, s)
            hi = prefixUpper(folded, s, lo)
        }
        val o = overlay
        val zAdj = (o?.logZ ?: 0f)
        val classBest = boundsFor(o).classBest
        var rareLp = Float.NaN

        fun offerClass(g: Int, classLp: Float) {
            // skip classes none of whose members can enter a full heap (BOUND_EPS absorbs float rounding, so only
            // offers the heap would reject are skipped and the result stays bit-identical)
            if (s.heapSize == s.heapCap && classLp + classBest[g] - zAdj + BOUND_EPS <= s.heapScore[0]) return
            val n = pack.classMemberCount(g)
            for (j in 0 until n) {
                val lex = pack.classMember(g, j)
                if (lex < lo || lex >= hi) continue
                if (!allowed(lex, folded, s)) continue
                heapOffer(s, wordLogProb(lex, classLp), lex)
            }
        }

        // explicit continuations, longest context first
        var bow = 0f
        if (lo < hi) {
            for (start in 0 until m) {
                val len = m - start
                if (len >= order) continue
                val node = pack.find(s.ctx, start, m)
                if (node < 0) continue
                val cLo = pack.childStart(len, node)
                val cHi = pack.childStart(len, node + 1)
                for (c in cLo until cHi) {
                    val rec = pack.record(len + 1, c)
                    val g = (rec and 0xFFFF).toInt()
                    if (s.stamp[g] == s.gen) continue
                    s.stamp[g] = s.gen
                    val lp = bow + pack.recLogProbOf(len + 1, rec)
                    if (g == LmPack.RARE) { rareLp = lp; continue }
                    if (g < LmPack.NSPECIAL) continue
                    offerClass(g, lp)
                }
                bow += pack.bowAt(len, node)
            }
            // everything else backs off to the unigram distribution
            if (prefix.isEmpty()) {
                val top = topUnigrams
                val bound = boundsFor(o).unigramBound
                for (i in top.indices) {
                    // no class from here on can enter a full heap
                    if (s.heapSize == s.heapCap && bow + bound[i] - zAdj + BOUND_EPS <= s.heapScore[0]) break
                    val g = top[i]
                    if (s.stamp[g] == s.gen) continue
                    offerClass(g, bow + pack.unigramLogProb(g))
                }
            } else {
                val rare = if (rareLp.isNaN()) bow + pack.unigramLogProb(LmPack.RARE) else rareLp
                for (lex in lo until hi) {
                    val g = pack.ngramIdOf(lex)
                    val lp = when {
                        g == LmPack.RARE -> rare
                        g < LmPack.NSPECIAL || s.stamp[g] == s.gen -> continue
                        else -> bow + pack.unigramLogProb(g)
                    }
                    if (!allowed(lex, folded, s)) continue
                    heapOffer(s, wordLogProb(lex, lp), lex)
                }
            }
        }

        // drain the heap (best first)
        val n = s.heapSize
        val ids = IntArray(n)
        val scores = FloatArray(n)
        for (j in n - 1 downTo 0) {
            ids[j] = s.heapId[0]
            scores[j] = s.heapScore[0]
            heapPop(s)
        }
        var out = (0 until n).map { pack.word(ids[it]) to scores[it] }
        val u = user
        if (u != null) out = mergeUser(context, folded, out, u, limit)
        return if (out.size > limit) out.subList(0, limit) else out
    }

    private fun allowed(lex: Int, foldedPrefix: String, s: Scratch): Boolean {
        if (!blockOffensive || !isOffensiveId(lex)) return true
        s.sb.setLength(0)
        pack.wordInto(lex, s.sb)
        if (foldedPrefix.isNotEmpty() && rules.fold(s.sb) == foldedPrefix) return true
        return user != null && user.count(s.sb.toString()) >= OFFENSIVE_LEARN_COUNT
    }

    private fun mergeUser(
        context: List<String>, foldedPrefix: String, static: List<Pair<String, Float>>, u: UserNgramModel, limit: Int,
    ): List<Pair<String, Float>> {
        val canon = canonical(context)
        val lam = u.weight(canon)
        if (lam <= 0f) return static
        val scores = LinkedHashMap<String, Float>()
        for ((w, lp) in static) scores[w] = lp
        for (w in u.candidates(canon, limit + 8)) {
            if (w in scores) continue
            if (foldedPrefix.isNotEmpty() && !rules.fold(w).startsWith(foldedPrefix)) continue
            val lex = lexIdOf(w)
            if (lex >= 0 && blockOffensive && isOffensiveId(lex) && u.count(w) < OFFENSIVE_LEARN_COUNT) continue
            scores[w] = staticLogProb(context, w)
        }
        return scores.entries.map { (w, lp) ->
            val p = (1 - lam) * exp(lp.toDouble()) + lam * u.prob(canon, canonicalWord(w))
            w to ln(max(p, 1e-300)).toFloat()
        }.sortedByDescending { it.second }
    }

    // ============================================================================================ prefix range

    private fun prefixLower(folded: String, s: Scratch): Int {
        var a = 0
        var b = pack.nLex
        while (a < b) {
            val mid = (a + b) ushr 1
            if (pack.compareFoldPrefix(mid, folded, s.sb, s.sb2) < 0) a = mid + 1 else b = mid
        }
        return a
    }

    private fun prefixUpper(folded: String, s: Scratch, from: Int): Int {
        var a = from
        var b = pack.nLex
        while (a < b) {
            val mid = (a + b) ushr 1
            if (pack.compareFoldPrefix(mid, folded, s.sb, s.sb2) <= 0) a = mid + 1 else b = mid
        }
        return a
    }

    /** Lexicon ids [first, last) whose completion key starts with [prefix] (for decoders building candidates). */
    fun prefixRange(prefix: String): IntRange {
        val s = scratch.get()
        val f = rules.fold(prefix)
        val lo = prefixLower(f, s)
        return lo until prefixUpper(f, s, lo)
    }

    // ============================================================================================ heap

    private fun heapReset(s: Scratch, cap: Int) { s.heapSize = 0; s.heapCap = cap }

    private fun heapOffer(s: Scratch, score: Float, id: Int) {
        if (s.heapSize < s.heapCap) {
            var i = s.heapSize++
            while (i > 0) {
                val p = (i - 1) ushr 1
                if (s.heapScore[p] <= score) break
                s.heapScore[i] = s.heapScore[p]; s.heapId[i] = s.heapId[p]; i = p
            }
            s.heapScore[i] = score; s.heapId[i] = id
        } else if (score > s.heapScore[0]) {
            siftDown(s, score, id)
        }
    }

    private fun heapPop(s: Scratch) {
        val n = --s.heapSize
        if (n > 0) siftDown(s, s.heapScore[n], s.heapId[n])
    }

    private fun siftDown(s: Scratch, score: Float, id: Int) {
        var i = 0
        val n = s.heapSize
        while (true) {
            var c = 2 * i + 1
            if (c >= n) break
            if (c + 1 < n && s.heapScore[c + 1] < s.heapScore[c]) c++
            if (s.heapScore[c] >= score) break
            s.heapScore[i] = s.heapScore[c]; s.heapId[i] = s.heapId[c]; i = c
        }
        s.heapScore[i] = score; s.heapId[i] = id
    }

    // ============================================================================================ learning

    /**
     * Learns that the user committed [word] after [context]. Call only for fields where learning is allowed
     * (not passwords, not IME_FLAG_NO_PERSONALIZED_LEARNING). Data stays in [user] (persist it to app-private,
     * no-backup storage with [UserNgramModel.save]).
     */
    fun observe(context: List<String>, word: String) {
        val u = user ?: return
        if (word.isBlank() || isNumber(word)) return
        u.observe(canonical(context), canonicalWord(word))
    }

    /** Context as the user model sees it: canonical words, "<s>" at sentence starts, at most 2 words. */
    private fun canonical(context: List<String>): List<String> {
        val out = ArrayList<String>(3)
        var i = context.size - 1
        outer@ while (i >= 0 && out.size < 2) {
            val tokens = splitPunct(context[i])
            for (t in tokens.indices.reversed()) {
                val tok = tokens[t]
                if (tok.isEmpty() || tok == "," || tok == ";" || tok == ":") continue
                if (tok.all { it == '.' || it == '!' || it == '?' || it == '…' }) { out.add(UserNgramModel.BOS); break@outer }
                out.add(canonicalWord(tok))
                if (out.size >= 2) break@outer
            }
            i--
        }
        if (out.size < 2 && i < 0 && (out.isEmpty() || out.last() != UserNgramModel.BOS)) out.add(UserNgramModel.BOS)
        out.reverse()
        return out
    }

    private fun canonicalWord(word: String): String {
        val id = lexIdOf(word)
        return if (id >= 0) pack.word(id) else word
    }

    // ============================================================================================ compounds

    /**
     * Greedy decomposition of an unknown closed compound into lexicon words (with the pack's linking
     * morphemes), e.g. German "Hausaufgabenheft" -> Haus + Aufgaben + Heft. Returns lexicon ids or null.
     */
    fun decompose(word: String): IntArray? {
        if (!pack.isCompounding || word.length < 6) return null
        return decomposeFrom(word, 0, 0)
    }

    private fun decomposeFrom(word: String, start: Int, depth: Int): IntArray? {
        if (depth >= 4) return null
        val rest = word.length - start
        if (depth > 0) {
            val whole = lexIdOf(word.substring(start))
            if (whole >= 0 && rest >= 3) return intArrayOf(whole)
        }
        for (end in word.length - 3 downTo start + 3) {
            val head = lexIdOf(word.substring(start, end))
            if (head < 0) continue
            for (link in pack.linkingMorphemes.ifEmpty { listOf("") }.let { listOf("") + it }.distinct()) {
                if (!word.startsWith(link, end)) continue
                val tail = decomposeFrom(word, end + link.length, depth + 1) ?: continue
                return intArrayOf(head) + tail
            }
        }
        return null
    }

    companion object {
        private const val MAX_HEAP = 64
        /** Margin on the top-k bounds for float rounding (scores are ~-30..0, so an ulp is < 1e-5). */
        private const val BOUND_EPS = 1e-3f
        private const val COMPOUND_PENALTY = -1.0f
        const val OFFENSIVE_LEARN_COUNT = 2.0

        private fun isNumber(s: String): Boolean {
            var digits = 0
            for (c in s) {
                if (c.isDigit()) digits++ else if (c !in ".,:/%$€£") return false
            }
            return digits > 0
        }

        /** Splits a context entry into a word and its surrounding punctuation tokens. */
        internal fun splitPunct(raw: String): List<String> {
            var a = 0
            var b = raw.length
            while (a < b && raw[a] in LEADING_STRIP) a++
            val trailing = ArrayList<String>(2)
            while (b > a) {
                val c = raw[b - 1]
                if (c in ".!?…") {
                    var e = b
                    while (b > a && raw[b - 1] in ".!?…") b--
                    trailing.add(0, raw.substring(b, e))
                    // a single trailing '.' after a known abbreviation is still a sentence end here; good enough
                } else if (c in ",;:") {
                    trailing.add(0, c.toString()); b--
                } else if (c in TRAILING_STRIP) {
                    b--
                } else break
            }
            val core = raw.substring(a, b)
            return if (core.isEmpty()) trailing else listOf(core) + trailing
        }

        private const val LEADING_STRIP = "\"“”‘(['«¿¡"
        private const val TRAILING_STRIP = "\"“”’)]»'"
    }
}
