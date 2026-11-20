package dev.pboard.engine.decode

import dev.pboard.engine.LanguageModel
import dev.pboard.engine.lm.NeuralRescoringLm
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max

/**
 * One active typing language: its model, lexicon (which carries the locale [TextRules]) and a base
 * weight (how much the user types in it). Several slots may be active at once (multilingual typing
 * without switching); candidates from all of them compete in one ranked list. [lm] drives every search; an
 * optional [rescorer] (the neural LM over the same n-gram model) only re-ranks final candidates and next-word
 * predictions ([EngineConfig.neuralRerankTopN]).
 */
class LanguageSlot(
    val tag: String, val lm: LanguageModel, val lexicon: Lexicon, val weight: Float = 1f, val rescorer: NeuralRescoringLm? = null,
) {
    val rules: TextRules get() = lexicon.rules
}

/**
 * Per-language log prior that adapts to the language of recently committed words: each commit adds the
 * normalised evidence ln P_l(word) − ln Σ_l' P_l'(word), decayed geometrically, on top of the base weights.
 * A floor keeps every active language reachable. With a single language every prior is 0.
 */
class LanguageMixer(private val baseWeights: FloatArray, private val decay: Float = 0.8f, private val floor: Float = 0.05f) {
    private val evidence = FloatArray(baseWeights.size)
    private val logPriors = FloatArray(baseWeights.size)

    init { recompute() }

    val size: Int get() = baseWeights.size

    fun logPrior(lang: Int): Float = logPriors[lang]

    /** [logProbs] = per-language ln P(word | context) of a committed word. */
    fun observe(logProbs: FloatArray) {
        if (size < 2) return
        var m = Float.NEGATIVE_INFINITY
        for (v in logProbs) m = max(m, v)
        var z = 0.0
        for (v in logProbs) z += exp((v - m).toDouble())
        val lse = m + ln(z).toFloat()
        for (i in evidence.indices) evidence[i] = decay * evidence[i] + (logProbs[i] - lse).coerceAtLeast(-6f)
        recompute()
    }

    private fun recompute() {
        if (size < 2) { logPriors.fill(0f); return }
        val p = DoubleArray(size) { baseWeights[it].coerceAtLeast(1e-6f) * exp(evidence[it].toDouble()) }
        val z = p.sum()
        for (i in 0 until size) logPriors[i] = ln(max(p[i] / z, floor.toDouble())).toFloat()
    }
}
