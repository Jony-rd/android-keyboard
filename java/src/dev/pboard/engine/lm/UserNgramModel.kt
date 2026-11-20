package dev.pboard.engine.lm

import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.InputStream
import java.io.OutputStream
import kotlin.math.pow

/**
 * Small on-device n-gram model (unigram/bigram/trigram) learned from what the user commits.
 *
 * - Counts decay exponentially with wall-clock time ([halfLifeMillis]), so old habits fade.
 * - Size is bounded ([maxEntries]); the weakest entries are evicted.
 * - It is a cache model: [prob] is an interpolated Witten-Bell estimate over the user's counts, and
 *   [weight] says how much to trust it for a context (grows with the user's evidence for that context).
 * - It never leaves the device: there is no network code; persist it with [save]/[load] into app-private,
 *   no-backup storage. Do not feed it text from password/sensitive fields or fields with
 *   IME_FLAG_NO_PERSONALIZED_LEARNING.
 *
 * Words are stored in the base pack's canonical spelling when known (so "london" learns "London").
 * Not thread-safe: call from the input thread.
 */
class UserNgramModel(
    val halfLifeMillis: Long = 60L * 24 * 3600 * 1000,
    val maxEntries: Int = 60_000,
    private val clock: () -> Long = System::currentTimeMillis,
) {
    private class Entry(var count: Double, var stamp: Long)

    private val grams = HashMap<String, Entry>()        // "w", "h1\u0001w", "h2\u0001h1\u0001w"
    private val ctxTotals = HashMap<String, Entry>()    // "" (all), "h1", "h2\u0001h1": total continuations
    private val ctxTypes = HashMap<String, Int>()       // distinct continuations seen (not decayed)
    private val blocked = HashSet<String>()             // words the user removed from suggestions
    private val successors = HashMap<String, HashSet<String>>()  // h1 -> words seen after it (index)
    private var topWords: List<String>? = null           // cached unigram ranking (invalidated on updates)

    val size: Int get() = grams.size

    /** Incremented on every change (observe, forget, clear, load), so dependent indexes can be rebuilt. */
    @Volatile var version: Int = 0
        private set

    /** Single words the user has committed, with their decayed counts (removed words excluded). */
    fun unigrams(): Map<String, Double> {
        val now = clock()
        val out = HashMap<String, Double>()
        for ((k, e) in grams) if (SEP !in k && k !in blocked) out[k] = decayed(e, now)
        return out
    }

    private fun decayed(e: Entry?, now: Long): Double {
        if (e == null) return 0.0
        val dt = (now - e.stamp).coerceAtLeast(0)
        return if (dt == 0L) e.count else e.count * 0.5.pow(dt.toDouble() / halfLifeMillis)
    }

    private fun bump(map: HashMap<String, Entry>, key: String, now: Long): Boolean {
        val e = map[key]
        return if (e == null) { map[key] = Entry(1.0, now); true } else { e.count = decayed(e, now) + 1.0; e.stamp = now; false }
    }

    /** Records [word] committed after [context] (canonical words, most recent last, "<s>" = sentence start). */
    fun observe(context: List<String>, word: String) {
        if (word.isEmpty() || word in blocked) return
        version++
        val now = clock()
        val h1 = context.lastOrNull()
        val h2 = if (context.size >= 2) context[context.size - 2] else null
        if (bump(grams, word, now)) ctxTypes[""] = (ctxTypes[""] ?: 0) + 1
        bump(ctxTotals, "", now)
        topWords = null
        if (h1 != null) {
            if (bump(grams, "$h1$SEP$word", now)) ctxTypes[h1] = (ctxTypes[h1] ?: 0) + 1
            successors.getOrPut(h1) { HashSet() }.add(word)
            bump(ctxTotals, h1, now)
            if (h2 != null) {
                val h = "$h2$SEP$h1"
                if (bump(grams, "$h$SEP$word", now)) ctxTypes[h] = (ctxTypes[h] ?: 0) + 1
                bump(ctxTotals, h, now)
            }
        }
        if (grams.size > maxEntries) evict(now)
    }

    /** Decayed number of times the user committed [word]. */
    fun count(word: String): Double = decayed(grams[word], clock())

    /** A word the user has typed often enough to stop treating it as unknown. */
    fun isKnown(word: String): Boolean = word !in blocked && count(word) >= KNOWN_COUNT

    /** Interpolated Witten-Bell probability of [word] after [context] under the user's counts. */
    fun prob(context: List<String>, word: String): Double {
        if (word in blocked) return 0.0
        val now = clock()
        val total = decayed(ctxTotals[""], now)
        if (total <= 0) return 0.0
        var p = decayed(grams[word], now) / total
        val h1 = context.lastOrNull() ?: return p
        p = wb("$h1$SEP$word", h1, p, now)
        if (context.size >= 2) {
            val h = "${context[context.size - 2]}$SEP$h1"
            p = wb("$h$SEP$word", h, p, now)
        }
        return p
    }

    private fun wb(gramKey: String, ctxKey: String, lower: Double, now: Long): Double {
        val c = decayed(ctxTotals[ctxKey], now)
        if (c <= 0) return lower
        val t = (ctxTypes[ctxKey] ?: 0).toDouble()
        return (decayed(grams[gramKey], now) + t * lower) / (c + t)
    }

    /** Mixture weight of the user model for [context]: small base weight, more after contexts the user uses. */
    fun weight(context: List<String>): Float {
        val now = clock()
        if (decayed(ctxTotals[""], now) <= 0) return 0f
        val h1 = context.lastOrNull() ?: return BASE_WEIGHT.toFloat()
        val c = decayed(ctxTotals[h1], now)
        return (BASE_WEIGHT + MAX_CONTEXT_WEIGHT * c / (c + CONTEXT_K)).toFloat()
    }

    /** The user's most frequent continuations of [context] (most recent context word), best first. */
    fun candidates(context: List<String>, limit: Int): List<String> {
        val now = clock()
        val h1 = context.lastOrNull()
        val next = if (h1 == null) null else successors[h1]
        if (!next.isNullOrEmpty()) {
            return next.asSequence().filter { it !in blocked }
                .map { it to decayed(grams["$h1$SEP$it"], now) }.filter { it.second > 0 }
                .sortedByDescending { it.second }.take(limit).map { it.first }.toList()
        }
        val top = topWords ?: grams.entries.asSequence().filter { SEP !in it.key && it.key !in blocked }
            .sortedByDescending { decayed(it.value, now) }.take(TOP_CACHE).map { it.key }.toList()
            .also { topWords = it }
        return top.take(limit)
    }

    /** Forgets [word] everywhere and stops learning it (Gboard's "remove suggestion"). */
    fun forget(word: String) {
        version++
        blocked.add(word)
        grams.keys.removeAll { k -> k == word || k.split(SEP).contains(word) }
        successors.remove(word)
        successors.values.forEach { it.remove(word) }
        topWords = null
    }

    /**
     * One-time repair (user data format 3): words learned with only a sentence-start capital ("Hello" at the start of
     * messages, learned before that capital was dropped) are folded into their lower-case form when [lowerKnown] says
     * it is a word. A form also seen mid-sentence (after a word, not after a sentence start) is the user's and is kept;
     * so are blocked words. Counts move to the lower-case n-grams; contexts that use the word are left alone (contexts
     * keep the text's spelling). Returns the number of words folded.
     */
    fun foldSentenceStartCapitals(lowerKnown: (String) -> Boolean): Int {
        val now = clock()
        val folded = HashMap<String, String>()
        for (w in grams.keys) {
            if (SEP in w || w in blocked || w.length < 2 || !Character.isUpperCase(w[0]) || w.drop(1).any { Character.isUpperCase(it) }) continue
            val lower = w.substring(0, 1).lowercase() + w.substring(1)
            if (lower == w || !lowerKnown(lower)) continue
            folded[w] = lower
        }
        if (folded.isEmpty()) return 0
        // Seen mid-sentence: a bigram whose context word is not the sentence start.
        for (k in grams.keys) {
            val parts = k.split(SEP)
            if (parts.size == 2 && parts[0] != BOS) folded.remove(parts[1])
        }
        if (folded.isEmpty()) return 0
        val moves = grams.keys.filter { k -> k.substringAfterLast(SEP) in folded }
        for (k in moves) {
            val w = k.substringAfterLast(SEP)
            val target = k.dropLast(w.length) + folded.getValue(w)
            val e = grams.remove(k) ?: continue
            val existing = grams[target]
            if (existing == null) grams[target] = e
            else {
                grams[target] = Entry(decayed(existing, now) + decayed(e, now), now)
                // Two continuations of one context became one.
                val ctx = if (SEP in k) k.substringBeforeLast(SEP) else ""
                ctxTypes[ctx]?.let { if (it > 1) ctxTypes[ctx] = it - 1 }
            }
        }
        version++
        rebuildIndex()
        return folded.size
    }

    fun clear() {
        version++
        grams.clear(); ctxTotals.clear(); ctxTypes.clear(); blocked.clear(); successors.clear(); topWords = null
    }

    private fun evict(now: Long) {
        val target = (maxEntries * 0.8).toInt()
        val sorted = grams.entries.sortedBy { decayed(it.value, now) }
        for (i in 0 until grams.size - target) grams.remove(sorted[i].key)
        rebuildIndex()
    }

    private fun rebuildIndex() {
        successors.clear()
        topWords = null
        for (k in grams.keys) {
            val parts = k.split(SEP)
            if (parts.size == 2) successors.getOrPut(parts[0]) { HashSet() }.add(parts[1])
        }
    }

    // ------------------------------------------------------------------------------------------ persistence

    fun save(out: OutputStream) {
        val d = DataOutputStream(out)
        d.writeInt(MAGIC)
        d.writeInt(FORMAT_VERSION)
        fun writeMap(m: Map<String, Entry>) {
            d.writeInt(m.size)
            for ((k, e) in m) { d.writeUTF(k); d.writeDouble(e.count); d.writeLong(e.stamp) }
        }
        writeMap(grams)
        writeMap(ctxTotals)
        d.writeInt(ctxTypes.size)
        for ((k, v) in ctxTypes) { d.writeUTF(k); d.writeInt(v) }
        d.writeInt(blocked.size)
        for (w in blocked) d.writeUTF(w)
        d.flush()
    }

    fun load(input: InputStream) {
        version++
        val d = DataInputStream(input)
        require(d.readInt() == MAGIC) { "not a user model" }
        val v = d.readInt()
        require(v in 1..FORMAT_VERSION) { "unsupported user model version $v" } // newer than this build: refuse
        clear()
        // Strict: the bytes may come from a restore file. A bad count or value throws; the caller keeps its old state.
        val now = clock()
        fun count(): Int = d.readInt().also { require(it in 0..MAX_LOAD_ENTRIES) { "bad count $it" } }
        fun readMap(m: HashMap<String, Entry>) {
            repeat(count()) {
                val k = d.readUTF()
                val c = d.readDouble()
                require(c.isFinite() && c >= 0.0 && c <= MAX_COUNT) { "bad value" }
                m[k] = Entry(c, minOf(d.readLong(), now)) // a stamp in the future would never decay
            }
        }
        readMap(grams)
        readMap(ctxTotals)
        repeat(count()) { val k = d.readUTF(); ctxTypes[k] = d.readInt().also { require(it >= 0) { "bad value" } } }
        repeat(count()) { blocked.add(d.readUTF()) }
        if (grams.size > maxEntries) evict(now) else rebuildIndex()
    }

    /**
     * Merges [other]'s state into this model (restoring a backup next to what was learned since): per n-gram and per
     * context the larger of the two decayed counts is kept, so merging the same data twice changes nothing; removed
     * words of either side stay removed.
     */
    fun mergeFrom(other: UserNgramModel) {
        version++
        val now = clock()
        blocked.addAll(other.blocked)
        fun mergeMap(mine: HashMap<String, Entry>, theirs: Map<String, Entry>) {
            for ((k, e) in theirs) {
                val c = decayed(e, now)
                if (c > decayed(mine[k], now)) mine[k] = Entry(c, now)
            }
        }
        mergeMap(grams, other.grams)
        mergeMap(ctxTotals, other.ctxTotals)
        for ((k, t) in other.ctxTypes) if (t > (ctxTypes[k] ?: 0)) ctxTypes[k] = t
        if (blocked.isNotEmpty()) grams.keys.removeAll { k -> k.split(SEP).any { it in blocked } }
        if (grams.size > maxEntries) evict(now) else rebuildIndex()
    }

    companion object {
        const val BOS = "<s>"
        private const val SEP = '\u0001'
        private const val MAGIC = 0x5042554D // "PBUM"
        /**
         * Persisted format ([save]/[load]), big-endian: magic "PBUM", version, then grams and context totals (each a
         * count of (key UTF, count double, stamp long)), context types (key UTF, int), blocked words (UTF).
         * History: 1 = initial. A new version must keep loading every older one (migrating in [load]); the frozen
         * samples in engine/src/test/resources/userdata/ enforce that (UserDataFormatTest).
         */
        const val FORMAT_VERSION = 1
        const val KNOWN_COUNT = 1.5
        private const val MAX_LOAD_ENTRIES = 4_000_000
        private const val MAX_COUNT = 1e12
        private const val TOP_CACHE = 64
        private const val BASE_WEIGHT = 0.03
        private const val MAX_CONTEXT_WEIGHT = 0.40
        private const val CONTEXT_K = 4.0
    }
}
