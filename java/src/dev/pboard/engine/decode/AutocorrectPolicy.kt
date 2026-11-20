package dev.pboard.engine.decode

/**
 * Gboard-style conservative autocorrect decision for a tapped word.
 *
 * Inputs are the ranked full-word hypotheses (corrections, not completions) and what the literal typed
 * string is with respect to the lexicon:
 *  - EXACT: the literal is a valid word as typed. Replace it only when another word beats it by a large
 *    margin in context ([AutocorrectParams.marginValid]); a same-keys variant ("ill" -> "I'll",
 *    "its" -> "it's") needs a smaller margin; 1-2 letter inputs have margins of their own (single letters: never).
 *  - VARIANT: the keys spell a word only with different capitals/apostrophes ("dont", "im", "iphone", "i").
 *    Always fix it (to the best hypothesis, which is normally that variant).
 *  - UNKNOWN: compare against a fixed out-of-vocabulary prior; single letters are never replaced and
 *    two-letter inputs need an extra margin; corrections that are too far from the taps are refused.
 */
class AutocorrectPolicy(private val p: AutocorrectParams, private val lmWeight: Float) {

    enum class LiteralStatus { EXACT, VARIANT, UNKNOWN }

    class Decision(val target: WordHypothesis?, val literalTotal: Float)

    /**
     * @param ranked full-word hypotheses, best first, already filtered of user-rejected corrections.
     * @param literalExact the hypothesis whose cased form equals the typed text, if any.
     * @param literalCost spatial cost of the literal key path.
     * @param sameKeys whether a hypothesis has exactly the literal's key sequence.
     */
    fun decide(
        ranked: List<WordHypothesis>,
        status: LiteralStatus,
        literalExact: WordHypothesis?,
        literalCost: Float,
        keyCount: Int,
        sameKeys: (WordHypothesis) -> Boolean,
        literalTotalOverride: Float? = null,
        /** Whether the context moves the odds towards [ranked]'s best word compared with context-free priors. */
        contextSupports: (WordHypothesis) -> Boolean = { true },
        /**
         * Whether a same-keys variant is only weak evidence for itself: it differs from a short literal only in
         * letter case and is the kind of form a lower-case input rarely means ("AR" for "ar", "Mo" for "mo").
         */
        weakCaseVariant: (WordHypothesis) -> Boolean = { false },
    ): Decision {
        val oovLogPrior = p.oovLogPriorBase + p.oovLogPriorPerChar * keyCount
        val literalTotal = literalExact?.total ?: literalTotalOverride ?: (literalCost - lmWeight * oovLogPrior)
        if (ranked.isEmpty()) return Decision(null, literalTotal)
        val best = ranked[0]
        if (literalExact != null && best.lang == literalExact.lang && best.wordId == literalExact.wordId) return Decision(null, literalTotal)
        return when (status) {
            LiteralStatus.VARIANT -> {
                // The keys spell a word that only needs capitals/apostrophes ("tom" -> "Tom", "mr" -> "Mr"):
                // fix that, unless a different word is overwhelmingly more likely (as for a valid literal).
                val v = ranked.firstOrNull(sameKeys) ?: return Decision(best.takeIf { withinReach(it, keyCount) }, literalTotal)
                // A short input whose keys spell only a capitalised form ("ar" -> AR, "js" -> JS) is weak
                // evidence for it: a likely word ("at", "is") wins by the margin an unknown literal needs.
                val shortCase = keyCount <= 3 && weakCaseVariant(v)
                val alt = if (shortCase) ranked.firstOrNull { !sameKeys(it) } else best.takeIf { !sameKeys(it) }
                val margin = if (shortCase) p.marginOov + (if (keyCount == 2) p.shortExtraMargin else 0f) else p.marginValid
                val other = alt != null && (keyCount > 2 || shortCase) && v.total - alt.total > margin && withinReach(alt, keyCount)
                Decision(if (other) alt else v, literalTotal)
            }
            LiteralStatus.EXACT -> {
                val variant = sameKeys(best)
                // Same-keys variants ("its" -> "it's"): follow the LM when the context itself favours the variant;
                // otherwise (no informative context) require a clear margin, i.e. less eager than Gboard. Short valid
                // literals have margins of their own ("yo" -> "to" after "talk").
                val margin = when {
                    variant -> if (contextSupports(best)) p.marginVariant else p.marginVariantContextFree
                    keyCount <= 1 -> p.marginValidLetter
                    keyCount == 2 -> p.marginValidShort
                    else -> p.marginValid
                }
                if (margin.isInfinite()) return Decision(null, literalTotal)
                val ok = literalTotal - best.total > margin && withinReach(best, keyCount)
                Decision(best.takeIf { ok }, literalTotal)
            }
            LiteralStatus.UNKNOWN -> {
                if (keyCount <= 1) return Decision(null, literalTotal)
                val margin = p.marginOov + if (keyCount == 2) p.shortExtraMargin else 0f
                val ok = literalTotal - best.total > margin && withinReach(best, keyCount)
                Decision(best.takeIf { ok }, literalTotal)
            }
        }
    }

    private fun withinReach(h: WordHypothesis, keyCount: Int): Boolean = h.channelCost <= p.maxCostPerKey * keyCount
}
