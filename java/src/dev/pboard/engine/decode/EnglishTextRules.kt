package dev.pboard.engine.decode

import java.util.Locale

/**
 * English rules: defaults plus a conservative British/American spelling-variant key, so the decoder can
 * recognise "colour"/"color", "organise"/"organize", "travelling"/"traveling", "centre"/"center",
 * "catalogue"/"catalog", "defence"/"defense", "analyse"/"analyze", "paediatric"/"pediatric" as one word in
 * two dialects. The key only matters when both spellings exist in a lexicon, and the rewrites are
 * restricted (stem lengths, multi-syllable stems for -ll-) to avoid pairing distinct words such as
 * "four"/"for" or "filled"/"filed". Lexicons may also supply explicit variant keys ([LexiconEntry.variantKey]).
 */
class EnglishTextRules(locale: Locale) : DefaultTextRules(locale) {

    override fun variantKey(word: String): String {
        var w = lowercase(word).replace('’', '\'')
        // Possessives and contractions share their stem's variants ("neighbour's" ~ "neighbor's").
        for (clitic in CLITICS) if (w.endsWith(clitic) && w.length > clitic.length + 2) return variantKey(w.dropLast(clitic.length)) + clitic
        LEXICAL[w]?.let { return it }
        if (w in NOT_VARIANTS) return w
        w = suffixRules(w)
        w = infixRules(w)
        return w
    }

    override fun isPossessiveOf(variant: String, typed: String): Boolean {
        val v = lowercase(variant).replace('’', '\'')
        val t = lowercase(typed)
        if (!v.endsWith("'s") || !t.endsWith("s")) return false
        val stem = v.dropLast(2)
        return stem.length >= 2 && t == stem + "s" && stem !in CONTRACTION_STEMS
    }

    override fun confusables(word: String): List<String> {
        val l = lowercase(word)
        return CONFUSABLE_INDEX[l]?.filter { it != l } ?: emptyList()
    }

    private fun suffixRules(w0: String): String {
        var w = w0
        // -isation/-ise/-ised/-ises/-ising/-iser -> z ; -yse… -> yze…
        for ((gb, us) in ISE) if (w.endsWith(gb) && w.length - gb.length >= 2) { w = w.dropLast(gb.length) + us; break }
        // -our(s/ed/ing/ite/able/…) -> -or…, stem of at least two letters ("four", "hour", "your" excluded).
        for (suf in OUR_SUFFIXES) {
            val gb = "our$suf"
            if (w.endsWith(gb) && w.length - gb.length >= 2) { w = w.dropLast(gb.length) + "or" + suf; break }
        }
        // -tre/-bre/-tres/-tred -> -ter/-ber… (centre, metre, theatre, fibre, calibre).
        for ((gb, us) in RE) if (w.endsWith(gb) && w.length - gb.length >= 2) { w = w.dropLast(gb.length) + us; break }
        // -ogue(s) -> -og(s) (catalogue, dialogue, analogue).
        for ((gb, us) in OGUE) if (w.endsWith(gb) && w.length - gb.length >= 3) { w = w.dropLast(gb.length) + us; break }
        // -ence -> -ense for the defence/licence/offence/pretence family.
        for ((gb, us) in ENCE) if (w.endsWith(gb)) { w = w.dropLast(gb.length) + us; break }
        // Doubled l before -ed/-ing/-er/-or/-ery after an unstressed multi-syllable stem (travelled, cancelling, jeweller).
        for (suf in LL_SUFFIXES) {
            val gb = "ll$suf"
            if (!w.endsWith(gb)) continue
            val stem = w.dropLast(gb.length)
            if (stem.length >= 3 && isVowel(stem.last()) && vowelGroups(stem) >= 2) { w = stem + "l" + suf; break }
        }
        return w
    }

    private fun infixRules(w: String): String {
        var r = w
        for ((gb, us) in INFIX) if (r.contains(gb)) r = r.replace(gb, us)
        return r
    }

    private fun vowelGroups(s: String): Int {
        var n = 0; var prev = false
        for (c in s) { val v = c in "aeiouy"; if (v && !prev) n++; prev = v }
        return n
    }

    companion object {
        private val ISE = listOf(
            "isations" to "izations", "isation" to "ization", "ising" to "izing", "ised" to "ized", "ises" to "izes",
            "isers" to "izers", "iser" to "izer", "ise" to "ize",
            "ysing" to "yzing", "ysed" to "yzed", "yses" to "yzes", "yse" to "yze",
        )
        private val OUR_SUFFIXES = listOf("ites", "ite", "ably", "able", "ings", "ing", "ers", "er", "ed", "ful", "less", "ist", "ists", "ism", "s", "")
        private val RE = listOf("tres" to "ters", "tred" to "tered", "tre" to "ter", "bres" to "bers", "bre" to "ber")
        private val OGUE = listOf("ogues" to "ogs", "ogue" to "og")
        private val ENCE = listOf("defence" to "defense", "defences" to "defenses", "offence" to "offense", "offences" to "offenses",
            "licence" to "license", "licences" to "licenses", "pretence" to "pretense", "pretences" to "pretenses")
        private val LL_SUFFIXES = listOf("ings", "ing", "ers", "er", "ors", "or", "ed", "ery", "ous", "ously")
        private val INFIX = listOf("paed" to "ped", "haem" to "hem", "anaem" to "anem", "oestr" to "estr", "foet" to "fet",
            "manoeuvr" to "maneuver", "oesoph" to "esoph", "diarrhoe" to "diarrhe")
        /** Whole-word pairs that no suffix rule covers (lower case, British -> American). */
        private val LEXICAL = mapOf(
            "grey" to "gray", "greys" to "grays", "greyish" to "grayish", "aluminium" to "aluminum", "plough" to "plow", "ploughs" to "plows",
            "mould" to "mold", "moulds" to "molds", "mouldy" to "moldy", "sceptic" to "skeptic", "sceptical" to "skeptical",
            "scepticism" to "skepticism", "jewellery" to "jewelry", "pyjamas" to "pajamas", "cosy" to "cozy", "artefact" to "artifact",
            "artefacts" to "artifacts", "ageing" to "aging", "judgement" to "judgment", "judgements" to "judgments",
            "acknowledgement" to "acknowledgment", "enrol" to "enroll", "enrols" to "enrolls", "enrolment" to "enrollment",
            "fulfil" to "fulfill", "fulfils" to "fulfills", "fulfilment" to "fulfillment", "skilful" to "skillful",
            "wilful" to "willful", "instalment" to "installment", "practise" to "practice", "practised" to "practiced",
            "practising" to "practicing", "manoeuvre" to "maneuver", "omelette" to "omelet", "doughnut" to "donut",
            "doughnuts" to "donuts",
            // Deliberately absent: pairs where the "American" form is also a different British word
            // (cheque/check, storey/story, tyre/tire, programme/program, draught/draft, mum/mom, maths/math).
        )
        private val CLITICS = listOf("s'", "'s", "'ll", "'d")

        /** Common real-word confusions (homophones and frequent mix-ups), lower case. */
        private val CONFUSABLE_GROUPS = listOf(
            "their there they're", "your you're", "its it's", "to too two", "then than", "were where we're wear",
            "affect effect", "lose loose", "whose who's", "accept except", "advice advise", "quite quiet",
            "lead led", "passed past", "principal principle", "stationary stationery", "weather whether",
            "hear here", "know no", "knew new", "brake break", "piece peace", "right write", "buy by bye",
            "our are", "of off", "witch which", "weak week", "sight site cite", "allowed aloud",
            "complement compliment", "desert dessert", "breath breathe", "cloth clothe", "chose choose",
        ).map { g -> g.split(' ') }
        private val CONFUSABLE_INDEX: Map<String, List<String>> = HashMap<String, List<String>>().also { m ->
            for (g in CONFUSABLE_GROUPS) for (w in g) m[w] = (m[w].orEmpty() + g).distinct()
        }
        /** Distinct words that the suffix rules would otherwise pair. */
        /** Stems whose "'s" is a contraction ("it is", "let us", "he is"), not a possessive. */
        private val CONTRACTION_STEMS = setOf("it", "let", "he", "she", "that", "what", "who", "where", "there", "here", "how", "when", "why")
        private val NOT_VARIANTS = setOf("prise", "prised", "prises", "prising")
    }
}
