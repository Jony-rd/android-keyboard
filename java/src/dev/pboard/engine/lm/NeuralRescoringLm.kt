package dev.pboard.engine.lm

import dev.pboard.engine.LanguageModel
import java.util.concurrent.CompletableFuture
import java.util.concurrent.Executors
import java.util.concurrent.Future
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max

/**
 * A [LanguageModel] that interpolates the n-gram model with the neural LM:
 *
 *     P(w | h) = alpha · P_nn(w | h) + (1 - alpha) · P_ngram(w | h)
 *
 * for words in the neural vocabulary. P_nn(w) = P_nn(class(w)) · P(w | class) uses the n-gram's in-class
 * shares and the active dialect overlay, so en-GB/en-US spelling choice is unchanged. Rarer words keep their
 * n-gram probability.
 *
 * - [logProb] scores any word (the decoders' candidate lists).
 * - [predict] rescores the n-gram's top [rescoreN] candidates (prefix-constrained) and, for next-word
 *   prediction, also the neural model's own top [nnExtra] words.
 * - The LSTM runs once per new context word. States are cached by sentence prefix ([cacheSize] entries), so
 *   typing word after word costs one [NeuralLm.step] per commit.
 *
 * Learning, offensive filtering and the lexicon stay with the wrapped [ngram].
 * Thread-safe: the state cache is synchronized, and LSTM steps run outside its lock.
 */
class NeuralRescoringLm(
    val ngram: NgramLanguageModel,
    val nn: NeuralLm,
    val alpha: Float = nn.recommendedAlpha,
    val rescoreN: Int = 32,
    val nnExtra: Int = 8,
    private val cacheSize: Int = 16,
) : LanguageModel {

    init {
        require(nn.baseBuildId == ngram.pack.buildId) { "neural LM was built for a different base pack" }
        require(nn.nv <= ngram.pack.nIds)
    }

    private val pack = ngram.pack
    private val logA = ln(alpha.toDouble())
    private val log1mA = ln(1.0 - alpha)

    private class Entry(val state: NeuralLm.State, val out: NeuralLm.Output)
    private class Key(val ids: IntArray) {
        private val h = ids.contentHashCode()
        override fun hashCode() = h
        override fun equals(other: Any?) = other is Key && other.ids.contentEquals(ids)
    }

    private val cache = object : LinkedHashMap<Key, Entry>(32, 0.75f, true) {
        override fun removeEldestEntry(eldest: MutableMap.MutableEntry<Key, Entry>?) = size > cacheSize
    }

    /** Contexts whose LSTM steps are running right now (guarded by the [cache] lock). */
    private val inflight = HashMap<Key, CompletableFuture<Entry>>()

    private val stepCount = AtomicLong()

    /** Number of LSTM steps run so far (for tests and benchmarks). */
    val steps: Long get() = stepCount.get()

    /** Fast path: the decoder scores many candidate words in the same context. */
    @Volatile private var last: Pair<List<String>, NeuralLm.Output>? = null

    /** Neural next-token distribution after [context] (cached; extends the longest cached sentence prefix). */
    fun output(context: List<String>): NeuralLm.Output {
        last?.let { (c, o) -> if (c == context) return o }
        val ids = ngram.sentenceIds(context)
        return outputFor(ids).also { last = ArrayList(context) to it }
    }

    /**
     * The steps run outside the cache lock, so a query for an already-cached context never waits behind a step
     * running for another context (e.g. a [prefetch]). A second query for a context that is being computed waits
     * for that computation instead of repeating it.
     */
    private fun outputFor(ids: IntArray): NeuralLm.Output {
        val key = Key(ids)
        val pending: CompletableFuture<Entry>
        val owner: Boolean
        var state: NeuralLm.State? = null
        var n = ids.size - 1
        synchronized(cache) {
            cache[key]?.let { return it.out }
            val running = inflight[key]
            owner = running == null
            pending = running ?: CompletableFuture<Entry>().also { inflight[key] = it }
            if (owner) {
                while (n > 0) {
                    val e = cache[Key(ids.copyOf(n))]
                    if (e != null) { state = e.state; break }
                    n--
                }
            }
        }
        if (!owner) return pending.join().out
        try {
            var s = state ?: nn.initialState()
            for (j in n until ids.size) { s = nn.step(s, ids[j]); stepCount.incrementAndGet() }
            val entry = Entry(s, nn.output(s))
            synchronized(cache) {
                cache[key] = entry
                inflight.remove(key)
            }
            pending.complete(entry)
            return entry.out
        } catch (t: Throwable) {
            synchronized(cache) { inflight.remove(key) }
            pending.completeExceptionally(t)
            throw t
        }
    }

    /** Single background thread for [prefetch], created on first use. */
    private val background by lazy {
        Executors.newSingleThreadExecutor { r -> Thread(r, "pboard-nnlm-prefetch").apply { isDaemon = true } }
    }

    /**
     * Optional: runs the LSTM step for [context] on a background thread and returns at once. Call it right after a
     * word is committed, with the context the next query will use; the next [predict] / [logProb] in that context
     * then finds the state cached and only pays for the rescoring.
     *
     * A query in that same context that arrives while the step is still running waits for it instead of running
     * it a second time; queries in other, already-cached contexts are not blocked by it. Without prefetching, the
     * first query in a new context runs the step itself, which is what happens by default. [context] is copied,
     * so the caller may reuse its list.
     */
    fun prefetch(context: List<String>): Future<*> {
        val ctx = ArrayList(context)
        return background.submit { output(ctx) }
    }

    /** Forgets every cached state (benchmarks of cold contexts: the first query in a field or after a cursor jump). */
    fun clearStates() {
        synchronized(cache) { cache.clear() }
        last = null
    }

    /** Neural log-probability of lexicon word [lexId], or null if it is outside the neural vocabulary. */
    private fun nnWordLogProb(out: NeuralLm.Output, lexId: Int): Float? {
        if (lexId < 0) return null
        val g = pack.ngramIdOf(lexId)
        if (g < LmPack.NSPECIAL || g >= nn.nv) return null
        return out.logProb(g) + ngram.wordTerm(lexId)
    }

    private fun mix(nnLp: Float?, ngLp: Float): Float {
        if (nnLp == null) return ngLp
        val a = logA + nnLp
        val b = log1mA + ngLp
        val m = max(a, b)
        return (m + ln(exp(a - m) + exp(b - m))).toFloat()
    }

    override fun logProb(context: List<String>, word: String): Float {
        val ng = ngram.logProb(context, word)
        val lex = ngram.lexIdIn(context, word)
        if (lex < 0) return ng
        return mix(nnWordLogProb(output(context), lex), ng)
    }

    override fun predict(context: List<String>, prefix: String, limit: Int): List<Pair<String, Float>> {
        if (limit <= 0) return emptyList()
        return rescorePredictions(context, prefix, ngram.predict(context, prefix, maxOf(rescoreN, limit)), limit)
    }

    /**
     * [predict] over the n-gram model's own predictions for [context] and [prefix] that the caller already has
     * (best first; the first `max(rescoreN, limit)` are rescored): saves predicting twice in the same context.
     */
    fun predictFrom(context: List<String>, ngramPredictions: List<Pair<String, Float>>, limit: Int): List<Pair<String, Float>> {
        if (limit <= 0) return emptyList()
        return rescorePredictions(context, "", ngramPredictions.take(maxOf(rescoreN, limit)), limit)
    }

    private fun rescorePredictions(context: List<String>, prefix: String, ngramPredictions: List<Pair<String, Float>>, limit: Int): List<Pair<String, Float>> {
        val out = output(context)
        val cands = LinkedHashMap<String, Float>()
        for ((w, lp) in ngramPredictions) cands[w] = lp
        if (prefix.isEmpty() && nnExtra > 0) {
            // words the neural model likes but the n-gram ranks low
            val top = out.topShortlist(nnExtra * 2) { g -> g >= LmPack.NSPECIAL && pack.classMemberCount(g) > 0 }
            var added = 0
            for (g in top) {
                if (added >= nnExtra) break
                // the class member the active dialect prefers (colour in en-GB, color in en-US)
                val lex = (0 until pack.classMemberCount(g)).map { pack.classMember(g, it) }.maxBy { ngram.wordTerm(it) }
                if (ngram.isOffensiveId(lex) && ngram.blockOffensive) continue
                val w = pack.word(lex)
                if (w in cands) continue
                cands[w] = ngram.logProb(context, w)
                added++
            }
        }
        return cands.entries.map { (w, ng) -> w to mix(nnWordLogProb(out, ngram.lexIdOf(w)), ng) }
            .sortedByDescending { it.second }.take(limit)
    }

    override fun isKnown(word: String): Boolean = ngram.isKnown(word)

    /** Learning goes to the n-gram's user model (the neural model is static). */
    fun observe(context: List<String>, word: String) = ngram.observe(context, word)
}
