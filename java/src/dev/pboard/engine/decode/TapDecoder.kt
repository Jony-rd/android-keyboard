package dev.pboard.engine.decode

import dev.pboard.engine.LanguageModel

/** One decoded word hypothesis before casing/policy: canonical spelling plus its cost breakdown (nats). */
class WordHypothesis(
    /** Index of the language slot that produced it. */
    val lang: Int,
    val wordId: Int,
    val canonical: String,
    /** Spatial + edit cost of explaining the input with this word (0 = typed exactly on centre). */
    val channelCost: Float,
    /** ln P(word | context) from that language's model (plus the language's log prior). */
    val lmLogProb: Float,
    /** Total cost used for ranking (lower is better). */
    val total: Float,
    /** True when the word is longer than the typed input (a completion rather than a correction). */
    val isCompletion: Boolean,
    val edits: Int,
    /**
     * Words before this one in a multi-word candidate, with their separators ("hello " when a missed space
     * split "hellobworld"); empty for ordinary candidates. The candidate text is prefix + canonical.
     */
    val prefix: String = "",
) {
    val text: String get() = prefix + canonical
}

/**
 * Touch input for a word being tapped, shared by all language slots. Per tap: costs over the layout's
 * keys ([KeySpace]) and the lower-case character the keyboard registered. [explicit] marks characters
 * the user produced deliberately without the touched key (long-press alternates, symbol layer, or no
 * geometry), which are matched by identity rather than spatially.
 */
class TapInput(
    val keyCosts: List<FloatArray>,
    val typed: CharArray,
    val explicit: BooleanArray,
    val xs: FloatArray,
    val ys: FloatArray,
    /** Per tap: cost that the touch was a stray hit just above the space bar (+inf when not near it). */
    val spaceCost: FloatArray? = null,
) {
    val size: Int get() = typed.size

    /** Taps [from, to) as a new input. */
    fun slice(from: Int, to: Int) = TapInput(keyCosts.subList(from, to), typed.copyOfRange(from, to), explicit.copyOfRange(from, to),
        xs.copyOfRange(from, to), ys.copyOfRange(from, to), spaceCost?.copyOfRange(from, to))
}

/**
 * Context-scoped language-model access for one decode: memoises [LanguageModel.logProb] and the context's
 * next-word predictions, which do not change while a word is being typed.
 */
class ContextLm(val lm: LanguageModel, val context: List<String>, private val predictionCount: Int) {
    private val memo = HashMap<String, Float>()
    val predictions: List<Pair<String, Float>> by lazy { if (predictionCount > 0) lm.predict(context, "", predictionCount) else emptyList() }
    fun logProb(word: String): Float = memo.getOrPut(word) { lm.logProb(context, word) }
}

/**
 * Beam search over one language's [Lexicon] trie explaining a sequence of taps (noisy-channel decoding).
 *
 * Layer `i` holds the best hypotheses (trie node, cost) that consumed exactly `i` taps. Transitions:
 *  - match: tap i -> a child character, cost = spatial cost of its key(s) (folded keys add a penalty);
 *  - insertion: tap i belongs to no character (cheaper when it repeats the previous one);
 *  - transposition: taps i, i+1 -> two characters in swapped order;
 *  - omission: the word has a character that was not tapped (cheaper for doubled letters);
 *  - silent skip: apostrophes/hyphens may be passed over for (almost) free ("dont" -> "don't").
 * Vowel-for-vowel substitutions (per [TextRules.vowels]) are capped at a cognitive-error cost.
 * Layers depend only on the taps (not on context), so they are cached and each keystroke only builds
 * one new layer. Pruning ranks hypotheses by cost minus a unigram look-ahead; contextual LM scores are
 * applied to final candidates only. Inner loops use primitive arrays (no boxing).
 */
class TapDecoder(private val lexicon: Lexicon, private val params: TapParams, private val spatialCap: Float) {

    /** A pruned layer: parallel arrays sorted by look-ahead priority (best first). */
    private class Beam(val node: IntArray, val cost: FloatArray, val edits: ByteArray, val omitted: BooleanArray) {
        val size: Int get() = node.size
    }

    /** Scratch open-addressing map node -> (cost, edits, omitted) for the layer under construction. */
    private class NodeMap {
        var keys = IntArray(1024) { -1 }
        var cost = FloatArray(1024)
        var edits = ByteArray(1024)
        var omitted = BooleanArray(1024)
        /** Slots in insertion order. */
        var order = IntArray(1024)
        var size = 0

        fun clear() {
            for (i in 0 until size) keys[order[i]] = -1
            size = 0
        }

        private fun slotOf(node: Int): Int {
            val mask = keys.size - 1
            var i = (node * -0x61c88647) ushr 7 and mask
            while (keys[i] != -1 && keys[i] != node) i = (i + 1) and mask
            return i
        }

        /** Slot holding [node], or -1. */
        fun get(node: Int): Int { val i = slotOf(node); return if (keys[i] == node) i else -1 }

        /** Returns the slot when the entry was inserted or improved, else -1. */
        fun relax(node: Int, c: Float, e: Int, om: Boolean): Int {
            if ((size + 1) * 2 > keys.size) grow()
            val i = slotOf(node)
            if (keys[i] == -1) {
                keys[i] = node; cost[i] = c; edits[i] = e.toByte(); omitted[i] = om
                order[size++] = i
                return i
            }
            if (c < cost[i] || (c == cost[i] && e < edits[i])) { cost[i] = c; edits[i] = e.toByte(); omitted[i] = om; return i }
            return -1
        }

        private fun grow() {
            val ok = keys; val oc = cost; val oe = edits; val oo = omitted; val oord = order; val n = size
            keys = IntArray(ok.size * 2) { -1 }; cost = FloatArray(ok.size * 2); edits = ByteArray(ok.size * 2)
            omitted = BooleanArray(ok.size * 2); order = IntArray(ok.size * 2); size = 0
            for (k in 0 until n) { val s = oord[k]; relax(ok[s], oc[s], oe[s].toInt(), oo[s]) }
        }
    }

    private val rules = lexicon.rules
    private val a = lexicon.alphabet.size
    private val isVowel = BooleanArray(a) { rules.isVowel(lexicon.alphabet[it]) }
    private val silentSym = BooleanArray(a) { rules.isSilent(lexicon.alphabet[it]) }
    private val foldOf = CharArray(a) { rules.fold(lexicon.alphabet[it]) ?: lexicon.alphabet[it] }
    private val hasSilent = silentSym.any { it }
    /** Nodes with at least one silent child (apostrophe/hyphen next). */
    private val silentParent: java.util.BitSet = java.util.BitSet(lexicon.nodeCount).also { bs ->
        if (hasSilent) for (n in 1 until lexicon.nodeCount) if (silentSym[lexicon.symOf(n)]) bs.set(lexicon.parentOf(n))
    }

    // Incremental cache: symbol costs and layers built for the last tap sequence.
    private var map: CharKeyMap? = null
    private var cachedXs = FloatArray(0)
    private var cachedYs = FloatArray(0)
    private var cachedTyped = CharArray(0)
    private val symCosts = ArrayList<FloatArray>()
    private val nearestSym = ArrayList<Int>()
    private val layers = ArrayList<Beam>()
    private val scratch = NodeMap()
    private var work = IntArray(256)

    fun invalidate() { layers.clear(); symCosts.clear(); nearestSym.clear(); cachedXs = FloatArray(0) }

    /** Binds the decoder to a layout; called whenever the geometry changes. */
    fun setKeyMap(map: CharKeyMap) { this.map = map; invalidate() }

    private fun maxEdits(n: Int) = when {
        n <= 2 -> 1
        n <= 6 -> 2
        n < params.longInputLength -> 3
        else -> params.maxEditsLong
    }

    private fun computeSymbolCosts(input: TapInput, i: Int): FloatArray {
        val out = FloatArray(a)
        val t = input.typed[i]
        val ts = lexicon.symbolOf(t)
        val m = map
        if (input.explicit[i] || m == null) {
            // Deliberate character: identity (or same base letter) only.
            val tf = rules.fold(t) ?: t
            for (s in 0 until a) out[s] = when {
                s == ts -> 0f
                foldOf[s] == tf -> params.foldPenalty
                else -> spatialCap
            }
        } else {
            m.symbolCosts(input.keyCosts[i], out, spatialCap, spatialCap)
        }
        // Cognitive substitutions: a vowel typed for another vowel.
        if (ts >= 0 && isVowel[ts]) {
            val v = params.vowelSubstitution
            for (s in 0 until a) if (isVowel[s] && out[s] > v) out[s] = v
        }
        return out
    }

    private var spaceCosts: FloatArray? = null

    private fun search(input: TapInput): Beam {
        val n = input.size
        spaceCosts = input.spaceCost
        var common = 0
        val lim = minOf(n, cachedXs.size, symCosts.size)
        while (common < lim && cachedXs[common] == input.xs[common] && cachedYs[common] == input.ys[common] && cachedTyped[common] == input.typed[common]) common++
        while (symCosts.size > common) { symCosts.removeAt(symCosts.size - 1); nearestSym.removeAt(nearestSym.size - 1) }
        for (i in symCosts.size until n) {
            val c = computeSymbolCosts(input, i)
            symCosts.add(c)
            var best = 0
            for (s in 1 until a) if (c[s] < c[best]) best = s
            nearestSym.add(lexicon.symbolOf(input.typed[i]).takeIf { it >= 0 } ?: best)
        }
        while (layers.size > common + 1) layers.removeAt(layers.size - 1)
        // Each layer's edit budget depends only on its own position, so a growing input never invalidates
        // cached layers (rebuilding every layer when the budget grew caused 20-40 ms keystrokes on devices).
        if (layers.isEmpty()) layers.add(initialLayer(maxEdits(1)))
        for (i in layers.size..n) layers.add(buildLayer(i, maxEdits(i)))
        cachedXs = input.xs.copyOf(); cachedYs = input.ys.copyOf(); cachedTyped = input.typed.copyOf()
        return layers[n]
    }

    private fun initialLayer(maxE: Int): Beam {
        val m = scratch
        m.clear()
        m.relax(lexicon.root, 0f, 0, false)
        closures(m, maxE)
        return prune(m)
    }

    private fun buildLayer(i: Int, maxE: Int): Beam {
        val m = scratch
        m.clear()
        val c = symCosts[i - 1]
        val nearest = nearestSym[i - 1]
        val maxMatch = params.maxMatchCost
        val prev = layers[i - 1]
        for (h in 0 until prev.size) {
            val node = prev.node[h]; val hc = prev.cost[h]; val he = prev.edits[h].toInt()
            val f = lexicon.firstChildOf(node)
            val cnt = lexicon.childCountOf(node)
            for (k in 0 until cnt) {
                val ch = f + k
                val s = c[lexicon.symOf(ch)]
                if (s > maxMatch) continue
                m.relax(ch, hc + s, he, false)
            }
            if (he < maxE) {
                var pen = when {
                    node == lexicon.root -> params.insertionFirst
                    nearest == lexicon.symOf(node) -> params.insertionSameLetter
                    else -> params.insertion
                }
                // A stray touch just above the space bar (a mistimed space, or the thumb resting) is a cheap extra tap.
                val sc = spaceCosts?.get(i - 1)
                if (sc != null && sc + params.strayTouchExtra < pen) pen = sc + params.strayTouchExtra
                m.relax(node, hc + pen, he + 1, false)
            }
        }
        if (i >= 2) {
            val c1 = symCosts[i - 2]
            val lim = params.transpositionMatchCost
            val pp = layers[i - 2]
            for (h in 0 until pp.size) {
                val he = pp.edits[h].toInt()
                if (he >= maxE) continue
                val node = pp.node[h]; val hc = pp.cost[h]
                val f = lexicon.firstChildOf(node)
                for (k in 0 until lexicon.childCountOf(node)) {
                    val x = f + k
                    val sx = lexicon.symOf(x)
                    val cx = c[sx] // the word's first character was tapped second
                    if (cx > lim) continue
                    val g = lexicon.firstChildOf(x)
                    for (kk in 0 until lexicon.childCountOf(x)) {
                        val y = g + kk
                        val sy = lexicon.symOf(y)
                        if (sy == sx) continue
                        val cy = c1[sy]
                        if (cy > lim) continue
                        m.relax(y, hc + cx + cy + params.transposition, he + 1, false)
                    }
                }
            }
        }
        closures(m, maxE)
        return prune(m)
    }

    /** Silent-character skips (free, repeatable) and single omissions (an edit, not chained). */
    private fun closures(m: NodeMap, maxE: Int) {
        if (hasSilent) silentClosure(m)
        val base = m.size
        for (k0 in 0 until base) {
            val slot = m.order[k0]
            val node = m.keys[slot]; val hc = m.cost[slot]; val he = m.edits[slot].toInt()
            if (he >= maxE || m.omitted[slot]) continue
            val f = lexicon.firstChildOf(node)
            val parentSym = if (node == lexicon.root) -1 else lexicon.symOf(node)
            for (k in 0 until lexicon.childCountOf(node)) {
                val ch = f + k
                val sym = lexicon.symOf(ch)
                if (silentSym[sym]) continue
                val pen = when {
                    node == lexicon.root -> params.omissionFirst
                    sym == parentSym -> params.omissionSameLetter
                    else -> params.omission
                }
                m.relax(ch, hc + pen, he + 1, true)
            }
        }
        if (hasSilent) silentClosure(m)
    }

    /** Propagates silent skips from every entry with a silent child (entries may have improved since the last pass). */
    private fun silentClosure(m: NodeMap) {
        var top = 0
        fun push(node: Int) { if (top == work.size) work = work.copyOf(top * 2); work[top++] = node }
        for (k in 0 until m.size) {
            val node = m.keys[m.order[k]]
            if (silentParent.get(node)) push(node)
        }
        while (top > 0) {
            val node = work[--top]
            val f = lexicon.firstChildOf(node)
            for (k in 0 until lexicon.childCountOf(node)) {
                val ch = f + k
                if (!silentSym[lexicon.symOf(ch)]) continue
                val slot = m.get(node)
                if (m.relax(ch, m.cost[slot] + params.silentSkip, m.edits[slot].toInt(), m.omitted[slot]) >= 0 && silentParent.get(ch)) push(ch)
            }
        }
    }

    private fun priority(node: Int, cost: Float): Float {
        val mp = lexicon.maxLogPrior(node)
        return cost - params.lmWeight * params.lookaheadWeight * (if (mp.isFinite()) mp else -30f)
    }

    private fun prune(m: NodeMap): Beam {
        val n = m.size
        if (n == 0) return Beam(IntArray(0), FloatArray(0), ByteArray(0), BooleanArray(0))
        val packed = LongArray(n)
        for (k in 0 until n) {
            val slot = m.order[k]
            packed[k] = (sortable(priority(m.keys[slot], m.cost[slot])).toLong() shl 32) or slot.toLong()
        }
        java.util.Arrays.sort(packed)
        val best = priority(m.keys[(packed[0] and 0xffffffffL).toInt()], m.cost[(packed[0] and 0xffffffffL).toInt()])
        var keep = 0
        while (keep < n && keep < params.beamSize) {
            val slot = (packed[keep] and 0xffffffffL).toInt()
            if (priority(m.keys[slot], m.cost[slot]) > best + params.beamWidth) break
            keep++
        }
        val node = IntArray(keep); val cost = FloatArray(keep); val edits = ByteArray(keep); val om = BooleanArray(keep)
        for (k in 0 until keep) {
            val slot = (packed[k] and 0xffffffffL).toInt()
            node[k] = m.keys[slot]; cost[k] = m.cost[slot]; edits[k] = m.edits[slot]; om[k] = m.omitted[slot]
        }
        return Beam(node, cost, edits, om)
    }

    /** Int whose signed order matches the float order. */
    private fun sortable(f: Float): Int { val b = java.lang.Float.floatToIntBits(f); return if (b < 0) b xor 0x7fffffff else b }

    /** Cost of matching the literal typed characters exactly (taps on their own keys), or null if not spellable. */
    fun literalCost(input: TapInput, upTo: Int = input.size): Float? {
        search(input)
        var c = 0f
        for (i in 0 until upTo) {
            val s = lexicon.symbolOf(input.typed[i])
            if (s < 0) return null
            c += symCosts[i][s]
        }
        return c
    }

    /**
     * Decodes [input] and returns word hypotheses (full matches and completions) scored by [ctx]'s
     * language model, best first. [langLogPrior] is added to every LM score (multi-language mixing).
     */
    fun decode(input: TapInput, ctx: ContextLm, lang: Int, langLogPrior: Float, upTo: Int = input.size, withCompletions: Boolean = true): List<WordHypothesis> {
        search(input)
        val finals = layers[upTo] // sorted by priority, best first
        if (finals.size == 0) return emptyList()
        val w = params.lmWeight

        class Pre(val id: Int, val cost: Float, val edits: Int, val completion: Boolean, val pre: Float)
        val full = ArrayList<Pre>()
        val seen = HashSet<Int>()
        // The literal spelling is always scored with its exact-path cost, however unlikely.
        val literal = String(input.typed, 0, upTo)
        val litNode = lexicon.find(literal)
        val pool = ArrayList<Pre>()
        if (litNode > 0 && lexicon.isTerminal(litNode)) {
            val lc = literalCost(input, upTo) ?: 0f
            for (id in lexicon.wordsAt(litNode)) if (seen.add(id)) pool.add(Pre(id, lc, 0, false, 0f))
        }
        for (h in 0 until finals.size) {
            val node = finals.node[h]
            if (!lexicon.isTerminal(node)) continue
            for (id in lexicon.wordsAt(node)) {
                if (seen.add(id)) full.add(Pre(id, finals.cost[h], finals.edits[h].toInt(), false, finals.cost[h] - w * lexicon.prior(id)))
            }
        }
        full.sortBy { it.pre }
        for (k in 0 until minOf(full.size, params.maxFullCandidatesRescored)) pool.add(full[k])

        // Completions from the most promising prefixes.
        val compSeen = HashSet<Int>()
        val tmp = ArrayList<Int>()
        for (h in 0 until if (withCompletions) minOf(finals.size, params.completionHyps) else 0) {
            val node = finals.node[h]
            if (node == lexicon.root) continue
            tmp.clear()
            lexicon.topCompletions(node, params.completionsPerHyp, tmp)
            for (id in tmp) if (id !in seen && compSeen.add(id)) pool.add(Pre(id, finals.cost[h], finals.edits[h].toInt(), true, 0f))
        }
        // Context-driven completions: likely next words whose spelling extends a surviving prefix.
        if (withCompletions && ctx.predictions.isNotEmpty()) {
            val finalByNode = HashMap<Int, Int>(finals.size * 2)
            for (h in finals.size - 1 downTo 0) finalByNode[finals.node[h]] = h
            for ((word, _) in ctx.predictions) {
                val id = lexicon.idOf(word)
                if (id < 0 || id in seen || id in compSeen) continue
                val path = rules.lowercase(word)
                var node = lexicon.root
                var best = -1
                for (ch in path) {
                    val s = lexicon.symbolOf(ch)
                    if (s < 0) break
                    node = lexicon.child(node, s)
                    if (node < 0) break
                    val h = finalByNode[node] ?: continue
                    if (best < 0 || finals.cost[h] < finals.cost[best]) best = h
                }
                if (best < 0) continue
                val cost = finals.cost[best]; val edits = finals.edits[best].toInt()
                if (lexicon.depthOf(finals.node[best]) == path.length) { if (seen.add(id)) pool.add(Pre(id, cost, edits, false, 0f)) }
                else if (compSeen.add(id)) pool.add(Pre(id, cost, edits, true, 0f))
            }
        }

        val out = ArrayList<WordHypothesis>(pool.size)
        for (p in pool) {
            val word = lexicon.word(p.id)
            val lp = ctx.logProb(word) + langLogPrior
            val total = p.cost - w * lp + if (p.completion) params.completionPenalty else 0f
            out.add(WordHypothesis(lang, p.id, word, p.cost, lp, total, p.completion, p.edits))
        }
        out.sortBy { it.total }
        return out
    }
}
