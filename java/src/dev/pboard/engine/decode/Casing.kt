package dev.pboard.engine.decode

/** How the user shaped the letters they typed (shift / caps lock), which candidates must follow. */
enum class TypedShape { LOWER, FIRST_UPPER, ALL_UPPER;
    companion object {
        fun of(typed: String): TypedShape {
            var letters = 0; var upper = 0
            var first: Char? = null
            for (c in typed) if (Character.isLetter(c)) {
                if (first == null) first = c
                letters++
                if (Character.isUpperCase(c) || Character.isTitleCase(c)) upper++
            }
            if (first == null) return LOWER
            return when {
                letters >= 2 && upper == letters -> ALL_UPPER
                Character.isUpperCase(first) || Character.isTitleCase(first) -> FIRST_UPPER
                else -> LOWER
            }
        }
    }
}

object Casing {
    /**
     * Applies the user's shift state and sentence-start capitalisation to a canonical lexicon spelling,
     * using the word's locale rules (Turkish i -> İ). Canonical capitals are never removed ("I'm", "Tom"),
     * and words with internal capitals keep their shape at a sentence start ("iPhone", "eBay") unless caps
     * lock was on.
     */
    fun apply(canonical: String, shape: TypedShape, sentenceStart: Boolean, rules: TextRules): String {
        if (shape == TypedShape.ALL_UPPER) return rules.uppercase(canonical)
        val capitalise = shape == TypedShape.FIRST_UPPER || sentenceStart
        if (!capitalise || canonical.isEmpty()) return canonical
        if (hasInternalCapital(canonical)) return canonical
        return if (Character.isLowerCase(canonical[0])) rules.capitalize(canonical) else canonical
    }

    private fun hasInternalCapital(w: String): Boolean {
        for (i in 1 until w.length) if (Character.isUpperCase(w[i])) return true
        return false
    }
}
