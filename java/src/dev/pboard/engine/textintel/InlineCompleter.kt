package dev.pboard.engine.textintel

import dev.pboard.engine.LanguageModel
import dev.pboard.engine.lm.LmPack
import dev.pboard.engine.lm.NgramLanguageModel
import kotlin.math.exp


/**
 * An inline completion: [text] goes right after the cursor (the rest of the word being typed, then whole words,
 * space separated). [words] are the words it adds or completes; [probability] is the model's estimate that the
 * user goes on to type exactly this.
 */
data class Completion(val text: String, val words: List<String>, val probability: Float)

/**
 * Tuning of [InlineCompleter]. A completion is offered only when the model's probability for the whole phrase is at
 * least [minProbability] and the phrase has at least [minWords] words, counting the word being completed (a short
 * lone word is what the suggestion strip is for), or it is a single word of at least [longWord] letters (Gboard 18.6
 * also completes single words inline only from 9 characters). Defaults: docs/text-intelligence.md (held-out
 * exact-match rate).
 */
data class CompleterConfig(
    val minProbability: Float = 0.7f,
    val minWords: Int = 2,
    val longWord: Int = 9,
    val maxWords: Int = 4,
    /** Mid-word, at least this many letters must be typed before the word is completed inline. */
    val minPrefix: Int = 1,
)

/**
 * Smart Compose-style phrase completion from the on-device language model (the n-gram pack, or the n-gram + LSTM
 * rescorer): greedy continuation while the joint probability stays high. Runs entirely on the device.
 *
 * Probabilities: after a space the first word's probability is P(w | context); mid-word it is
 * P(w | context, w starts with the typed prefix), normalised over the model's completions of the prefix. Each
 * further word multiplies in P(w_k | context, w_1..w_k-1); the phrase stops where a sentence end is likelier than
 * the next word ([ngram] needed for that) or the joint probability would fall below the threshold.
 */
class InlineCompleter(
    private val lm: LanguageModel,
    private val ngram: NgramLanguageModel?,
    val config: CompleterConfig = CompleterConfig(),
) {
    /** One step of the greedy path: the word, its conditional probability and the running product. */
    class Step(val word: String, val p: Float, val joint: Float)

    /**
     * The completion for [context] (the sentence so far as words, most recent last, like the decoder's context;
     * empty at a sentence start) and [prefix] (the letters of the word being typed; empty after a space), or null.
     */
    fun complete(context: List<String>, prefix: String): Completion? {
        if (prefix.isNotEmpty() && (prefix.length < config.minPrefix || !prefix.all { it.isLetter() || it == '\'' })) return null
        val path = path(context, prefix, config.maxWords, config.minProbability)
        return choose(path, prefix, config.minProbability, config.minWords, config.longWord)
    }

    /**
     * The longest prefix of [path] with joint probability >= [threshold] and at least [minWords] words, or a single
     * word of at least [longWord] letters.
     */
    fun choose(path: List<Step>, prefix: String, threshold: Float, minWords: Int, longWord: Int = Int.MAX_VALUE): Completion? {
        var n = 0
        while (n < path.size && path[n].joint >= threshold) n++
        if (n == 0 || (n < minWords && !(n == 1 && path[0].word.length >= longWord))) return null
        val words = path.subList(0, n).map { it.word }
        val first = if (prefix.isEmpty()) words[0] else words[0].substring(prefix.length)
        val text = (listOf(first) + words.drop(1)).joinToString(" ")
        return Completion(text, words, path[n - 1].joint)
    }

    /**
     * The greedy continuation, up to [maxWords] words, stopping once the joint probability drops below [floor]
     * (pass 0 to get the whole path, e.g. for evaluation).
     */
    fun path(context: List<String>, prefix: String, maxWords: Int, floor: Float = 0f): List<Step> {
        val out = ArrayList<Step>(maxWords)
        val ctx = ArrayList(context)
        var joint = 1.0
        // first word: complete the prefix, or predict the next word
        val first: Pair<String, Double> = if (prefix.isNotEmpty()) {
            val cands = lm.predict(ctx, prefix, 8)
            if (cands.isEmpty()) return out
            val z = cands.sumOf { exp(it.second.toDouble()) }
            val (w, lp) = cands[0]
            if (w.length <= prefix.length || !w.startsWith(prefix, ignoreCase = true)) return out
            // keep the letters as typed ("Thank" -> "Thanks"), the model's spelling for the rest
            (prefix + w.substring(prefix.length)) to exp(lp.toDouble()) / z
        } else {
            val cands = lm.predict(ctx, "", 2)
            if (cands.isEmpty()) return out
            cands[0].first to exp(cands[0].second.toDouble())
        }
        joint *= first.second
        out += Step(first.first, first.second.toFloat(), joint.toFloat())
        ctx += first.first
        while (out.size < maxWords && joint >= floor) {
            val cands = lm.predict(ctx, "", 2)
            if (cands.isEmpty()) break
            val (w, lp) = cands[0]
            val p = exp(lp.toDouble())
            // a sentence end likelier than the next word: the phrase is complete here
            if (ngram != null && eosProb(ctx) > p) break
            joint *= p
            out += Step(w, p.toFloat(), joint.toFloat())
            ctx += w
        }
        return out
    }

    private fun eosProb(ctx: List<String>): Double {
        val g = ngram ?: return 0.0
        val ids = g.sentenceIds(ctx)
        return exp(g.ngramLogProb(ids, ids.size, LmPack.EOS).toDouble())
    }
}
