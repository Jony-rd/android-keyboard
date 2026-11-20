package dev.pboard.engine.decode

import dev.pboard.engine.LanguageModel
import dev.pboard.engine.lm.NgramLanguageModel
import kotlin.math.ln

/**
 * The engine's single store of what the user taught it (CO-07): either the production language model's own
 * user n-gram model ([NgramUserWords], one model for phrases and new words) or, for language models without
 * one, the standalone [UserVocabulary]. The decoders read learned words from it as an extra lexicon.
 */
interface LearnedWords {
    /** Changes whenever the learned words change (the decoders rebuild their learned-word trie). */
    val version: Int
    fun count(word: String): Double
    /** Learned words worth offering, with log priors (words a language already knows are excluded). */
    fun entries(): List<LexiconEntry>
    fun logPrior(word: String): Float
    fun remove(word: String)
    fun clear()
    /** A context-free model over the learned words, so they can share the decoders. */
    val lm: LanguageModel
}

/** [LearnedWords] over an [NgramLanguageModel]'s user model: the LM learns phrases, the decoders reuse its words. */
class NgramUserWords(
    private val model: NgramLanguageModel,
    private val params: LearningParams,
    private val knownToLanguages: (String) -> Boolean,
) : LearnedWords {
    private val user = model.user!!

    /**
     * Changes only when the set of learned words can change (an unknown word observed, a removal, clear,
     * load), not on every commit of a known word: rebuilding the learned-word trie after every word would
     * put a scan of the whole user model on the next keystroke.
     */
    override var version: Int = 0
        private set
    override fun count(word: String): Double = user.count(word)

    override fun logPrior(word: String): Float {
        val c = user.count(word)
        return if (c <= 0.0) Float.NEGATIVE_INFINITY else params.baseLogPrior + ln(c.toFloat())
    }

    override fun entries(): List<LexiconEntry> = user.unigrams()
        .filter { (w, c) -> c >= params.candidateMinCount - 1e-6 && !knownToLanguages(w) }
        .map { (w, c) -> LexiconEntry(w, params.baseLogPrior + ln(c.toFloat())) }

    /** Learns [word] after [context] [weight] times (several for explicit signals such as undoing a correction). */
    /** See [dev.pboard.engine.lm.UserNgramModel.foldSentenceStartCapitals]. */
    fun foldSentenceStartCapitals(lowerKnown: (String) -> Boolean): Int = user.foldSentenceStartCapitals(lowerKnown).also { if (it > 0) version++ }

    fun observe(context: List<String>, word: String, weight: Int = 1) {
        repeat(weight) { model.observe(context, word) }
        if (!knownToLanguages(word)) version++
    }

    override fun remove(word: String) { user.forget(word); version++ }
    override fun clear() { user.clear(); version++ }

    /** The user model's state ([dev.pboard.engine.lm.UserNgramModel.save]). */
    fun save(): ByteArray = java.io.ByteArrayOutputStream().also { user.save(it) }.toByteArray()

    /** Restores [save]d state; corrupt or unknown input leaves the model unchanged and returns false. */
    fun load(bytes: ByteArray): Boolean {
        val backup = save()
        val ok = try { user.load(java.io.ByteArrayInputStream(bytes)); true } catch (e: Exception) {
            user.load(java.io.ByteArrayInputStream(backup)); false
        }
        version++
        return ok
    }

    override val lm: LanguageModel = object : LanguageModel {
        override fun logProb(context: List<String>, word: String): Float = logPrior(word).takeIf { it.isFinite() } ?: -30f
        override fun predict(context: List<String>, prefix: String, limit: Int): List<Pair<String, Float>> = emptyList()
        override fun isKnown(word: String): Boolean = count(word) >= params.intendedMinCount
    }
}
