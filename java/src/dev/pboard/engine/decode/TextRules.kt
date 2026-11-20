package dev.pboard.engine.decode

import java.text.Normalizer
import java.util.Locale

/**
 * Per-locale text rules. Everything language-specific the decoders need goes through this interface,
 * so no algorithm assumes English: case mapping (Turkish dotted/dotless i), which in-word marks users
 * omit (apostrophes, hyphens), diacritic folding from lexicon characters to key labels, elision and
 * compounding hints, and punctuation spacing for the IME.
 */
interface TextRules {
    val locale: Locale

    fun lowercase(s: String): String = s.lowercase(locale)
    fun uppercase(s: String): String = s.uppercase(locale)

    /** Upper-cases the first character only ("hello" -> "Hello"; Turkish "istanbul" -> "İstanbul"). */
    fun capitalize(word: String): String {
        if (word.isEmpty()) return word
        val cp = word.codePointAt(0)
        val n = Character.charCount(cp)
        return word.substring(0, n).uppercase(locale) + word.substring(n)
    }

    /** In-word characters users routinely leave out ("dont" for "don't", "email" for "e-mail"). */
    fun isSilent(c: Char): Boolean

    /**
     * The base character typed for [c] when there is no dedicated key (diacritic folding: 'é' -> 'e',
     * 'ü' -> 'u'), or null when [c] has no simpler form. Applied to lower-case lexicon characters.
     */
    fun fold(c: Char): Char?

    /** Characters that can be part of a word being composed. */
    fun isWordChar(c: Char): Boolean = Character.isLetter(c) || isSilent(c) || Character.getType(c) == Character.NON_SPACING_MARK.toInt()

    /** Elided prefixes that attach to the following word without a space (French "l'", "qu'"). */
    val elisionPrefixes: Set<String> get() = emptySet()

    /** The language freely forms closed compounds (German, Dutch, Finnish…); decoders may join lexicon words. */
    val compounding: Boolean get() = false

    /** Whether the IME should put a (narrow) space before [punct] (French "!", "?", ":", ";"). */
    fun spaceBefore(punct: Char): Boolean = false

    fun isSentenceEnd(c: Char): Boolean = c == '.' || c == '!' || c == '?'

    /**
     * Vowels of the script (lower case, unaccented). Spelling mistakes confuse vowels far more than their
     * key distance suggests ("definately", "seperate"), so decoders make vowel-for-vowel substitutions cheap.
     */
    val vowels: Set<Char> get() = LATIN_VOWELS

    fun isVowel(c: Char): Boolean = c in vowels || (fold(c)?.let { it in vowels } ?: false)

    /**
     * Comparison form: lower-case, silent characters removed, diacritics folded. Two spellings with the
     * same skeleton are typed with the same keys ("don't"/"dont", "déjà"/"deja", "iPhone"/"iphone").
     */
    fun skeleton(word: String): String {
        val lower = lowercase(word)
        val sb = StringBuilder(lower.length)
        for (c in lower) {
            if (isSilent(c)) continue
            sb.append(fold(c) ?: c)
        }
        return sb.toString()
    }

    /**
     * Key shared by the spellings of one word across dialects of this language ("colour"/"color" in
     * English). Decoders never autocorrect a valid spelling into another with the same key, and resolve
     * misspellings to the form the active language model prefers. Default: the lower-case word.
     */
    fun variantKey(word: String): String = lowercase(word)

    /**
     * Real-word confusions of this language (homophones and near-homophones: "their"/"there"/"they're").
     * Offered as alternatives for a valid word — notably when a committed word is re-opened for
     * correction without touch data — and scored by the language model in context. Default: none.
     */
    fun confusables(word: String): List<String> = emptyList()

    /**
     * Whether [variant] is the possessive of the valid word [typed] with the same keys ("neighbor's" for
     * "neighbors"). A plural is never autocorrected into a possessive; contractions that happen to look
     * the same ("it's" for "its", "let's" for "lets") are not possessives. Default: never.
     */
    fun isPossessiveOf(variant: String, typed: String): Boolean = false

    companion object {
        val LATIN_VOWELS: Set<Char> = "aeiouy".toSet()
        val CYRILLIC_VOWELS: Set<Char> = "аеёиоуыэюяіїєў".toSet()
        val GREEK_VOWELS: Set<Char> = "αεηιουω".toSet()

        /** Rules for a BCP-47 tag; unknown languages get [DefaultTextRules]. */
        fun forLanguageTag(tag: String): TextRules {
            val locale = Locale.forLanguageTag(tag)
            return when (locale.language) {
                "en" -> EnglishTextRules(locale)
                "fr" -> DefaultTextRules(locale, elisions = setOf("l'", "d'", "j'", "m'", "n'", "s'", "t'", "c'", "qu'", "jusqu'", "lorsqu'", "puisqu'"), spaceBeforePunct = "!?:;")
                "it" -> DefaultTextRules(locale, elisions = setOf("l'", "un'", "dell'", "all'", "dall'", "nell'", "sull'", "c'", "d'"))
                "de", "nl", "sv", "da", "nb", "nn", "no", "fi", "et", "hu", "is" -> DefaultTextRules(locale, compounds = true)
                "ru", "uk", "be", "bg", "sr", "mk", "kk" -> DefaultTextRules(locale, vowelSet = CYRILLIC_VOWELS)
                "el" -> DefaultTextRules(locale, vowelSet = GREEK_VOWELS)
                else -> DefaultTextRules(locale)
            }
        }
    }
}

/**
 * Unicode-driven defaults: folding strips combining marks (NFD), apostrophe-like marks and hyphens are
 * silent. A locale can override folds where the stripped form is not what its users type
 * (e.g. 'ø' -> 'o', 'ł' -> 'l' have no decomposition).
 */
open class DefaultTextRules(
    override val locale: Locale,
    private val elisions: Set<String> = emptySet(),
    private val compounds: Boolean = false,
    private val spaceBeforePunct: String = "",
    private val vowelSet: Set<Char> = TextRules.LATIN_VOWELS,
    private val extraFolds: Map<Char, Char> = mapOf('ø' to 'o', 'ł' to 'l', 'đ' to 'd', 'ħ' to 'h', 'ı' to 'i', 'œ' to 'o', 'æ' to 'a', 'ß' to 's'),
) : TextRules {
    private val foldCache = HashMap<Char, Char?>()

    // '.' is deliberately not silent: a token with an interior period is a URL/abbreviation and passes through as typed.
    override fun isSilent(c: Char): Boolean = c == '\'' || c == '’' || c == '-' || c == '‐' || c == '·'

    override fun fold(c: Char): Char? = synchronized(foldCache) {
        foldCache.getOrPut(c) {
            extraFolds[c] ?: run {
                val d = Normalizer.normalize(c.toString(), Normalizer.Form.NFD)
                val base = d.filter { Character.getType(it) != Character.NON_SPACING_MARK.toInt() }
                if (base.length == 1 && base[0] != c) base[0] else null
            }
        }
    }

    override val vowels: Set<Char> get() = vowelSet
    override val elisionPrefixes: Set<String> get() = elisions
    override val compounding: Boolean get() = compounds
    override fun spaceBefore(punct: Char): Boolean = punct in spaceBeforePunct
}
