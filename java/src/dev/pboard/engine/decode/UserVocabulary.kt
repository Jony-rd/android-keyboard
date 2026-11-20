package dev.pboard.engine.decode

import dev.pboard.engine.LanguageModel
import kotlin.math.ln

/** Learning thresholds (as Gboard: a candidate after 1 use, "intended" after 4). */
data class LearningParams(
    /** Uses before a learned word is offered as a candidate/completion. */
    val candidateMinCount: Int = 1,
    /** Uses before a learned word counts as intended and is no longer autocorrected away. */
    val intendedMinCount: Int = 4,
    /** Log prior of a learned word seen once; grows with ln(count). */
    val baseLogPrior: Float = -11f,
    /** Cap on stored words (least used are dropped). */
    val maxWords: Int = 5000,
    /** Log prior of a contact name ([dev.pboard.engine.KeyboardEngine.setContactNames]): about a learned word used three times. */
    val contactLogPrior: Float = -10f,
    /** Cap on contact name tokens held (the rest are ignored). */
    val maxContactNames: Int = 20_000,
)

/**
 * Words the user typed that no active lexicon knows (names, slang), learned on device. Data never
 * leaves the device; the IME persists [snapshot]/[restore] in app-private storage (excluded from backup)
 * and must not report commits from no-learning fields (passwords, incognito, IME_FLAG_NO_PERSONALIZED_LEARNING).
 */
class UserVocabulary(private val params: LearningParams = LearningParams()) : LearnedWords {
    private val counts = LinkedHashMap<String, Int>()
    /** Bumped on every change so dependent tries are rebuilt. */
    override var version: Int = 0; private set

    override fun count(word: String): Double = (counts[word] ?: 0).toDouble()

    fun observe(word: String, weight: Int = 1) {
        counts[word] = (counts[word] ?: 0) + weight
        if (counts.size > params.maxWords) {
            val drop = counts.entries.minByOrNull { it.value }!!.key
            counts.remove(drop)
        }
        version++
    }

    /** CO-08 "remove suggestion" for a learned word. */
    override fun remove(word: String) { if (counts.remove(word) != null) version++ }

    override fun clear() { counts.clear(); version++ }

    fun isIntended(word: String): Boolean = count(word) >= params.intendedMinCount

    override fun logPrior(word: String): Float {
        val c = counts[word] ?: 0
        return if (c <= 0) Float.NEGATIVE_INFINITY else params.baseLogPrior + ln(c.toFloat())
    }

    /** Words eligible as candidates, with priors. */
    override fun entries(): List<LexiconEntry> = counts.filter { it.value >= params.candidateMinCount }.map { LexiconEntry(it.key, logPrior(it.key)) }

    fun snapshot(): Map<String, Int> = LinkedHashMap(counts)
    fun restore(data: Map<String, Int>) { counts.clear(); counts.putAll(data); version++ }

    /** A context-free "language model" over the learned words, so they can share the decoders. */
    override val lm: LanguageModel = object : LanguageModel {
        override fun logProb(context: List<String>, word: String): Float = logPrior(word).takeIf { it.isFinite() } ?: -30f
        override fun predict(context: List<String>, prefix: String, limit: Int): List<Pair<String, Float>> =
            counts.keys.filter { it.startsWith(prefix, ignoreCase = true) && (counts[it] ?: 0) >= params.candidateMinCount }
                .sortedByDescending { counts[it] ?: 0 }.take(limit).map { it to logPrior(it) }
        override fun isKnown(word: String): Boolean = isIntended(word)
    }
}
