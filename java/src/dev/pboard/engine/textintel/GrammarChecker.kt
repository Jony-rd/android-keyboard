package dev.pboard.engine.textintel

import dev.pboard.engine.LanguageModel
import dev.pboard.engine.lm.LmPack
import dev.pboard.engine.lm.NgramLanguageModel

/**
 * A grammar issue: replacing text [start, end) with [replacement] fixes it. [markStart]/[markEnd] is the range to
 * underline (the word(s) at fault; for a deletion, the word to remove). [rule] names the rule family; [score] is
 * the language model's preference for the fix in nats (higher = surer).
 */
data class GrammarIssue(
    val start: Int,
    val end: Int,
    val replacement: String,
    val rule: String,
    val score: Float,
    val markStart: Int = start,
    val markEnd: Int = end,
)

/**
 * Per-family acceptance thresholds: a candidate fix is reported when the language model prefers it by at least
 * this many nats over what was written (log P(window | fix) - log P(window | original)). A family missing from
 * the map is off. Defaults were chosen on dev data for precision first (docs/text-intelligence.md).
 */
data class GrammarConfig(
    val thresholds: Map<String, Float> = DEFAULT_THRESHOLDS,
    /** Words after the change that are scored (an order-4 n-gram sees every n-gram touching the change with 3). */
    val window: Int = 3,
    /** Extra nats required to flag the last word of a sentence that has not ended yet (it may be continued). */
    val tailMargin: Float = 3f,
    /**
     * A candidate the n-gram alone scores this many nats below its threshold is rejected without the LSTM (which costs
     * a few ms per position on a phone). +inf always runs the LSTM.
     */
    val screenMargin: Float = 4f,
) {
    companion object {
        private const val OFF = Float.POSITIVE_INFINITY
        private const val ALWAYS = Float.NEGATIVE_INFINITY

        /**
         * Tuned on dev data only (synthetic errors in held-out-dev chat/web text, and the dev splits' own errors and
         * near-misses inspected by hand); see docs/text-intelligence.md. Families first, then per-pair overrides
         * for confusions whose members are both common in correct text (too/to at a sentence end, off/of in
         * phrasal verbs, an/and, site/sight).
         */
        val DEFAULT_THRESHOLDS: Map<String, Float> = mapOf(
            GrammarChecker.SPACING to ALWAYS,
            GrammarChecker.I_LOWER to ALWAYS,
            GrammarChecker.MODAL_OF to -1f,
            GrammarChecker.DOUBLE to 0f,
            GrammarChecker.A_AN to 0f,
            GrammarChecker.SVA to 2f,
            GrammarChecker.MODAL_FORM to 2.5f,
            GrammarChecker.PERFECT to 0f,
            GrammarChecker.PRON_PP to 2f,
            GrammarChecker.COMPARATIVE to 0f,
            GrammarChecker.SUPPOSED to 1.5f,
            GrammarChecker.CONFUSION to 3f,
            "conf:too>to" to 6f, "conf:off>of" to OFF, "conf:of>off" to 4f,
            "conf:an>and" to 5f, "conf:and>an" to 5f,
            "conf:bye>buy" to 5f, "conf:bye>by" to 5f, "conf:buy>bye" to 5f, "conf:by>bye" to 5f, "conf:by>buy" to 4f,
            "conf:buy>by" to 4f, "conf:site>sight" to OFF, "conf:sight>site" to OFF, "conf:wear>where" to OFF,
            "conf:where>wear" to OFF, "conf:desert>dessert" to 4f, "conf:dessert>desert" to 4f, "conf:role>roll" to 5f,
            "conf:roll>role" to 5f, "conf:board>bored" to 5f, "conf:bored>board" to 5f, "conf:lead>led" to 4f,
            "conf:led>lead" to 4f, "conf:quite>quiet" to 4f, "conf:quiet>quite" to 4f, "conf:weight>wait" to 4f,
            "conf:wait>weight" to 4f, "conf:our>are" to 4f, "conf:are>our" to 4f, "conf:thought>though" to 5f,
            "conf:which>witch" to 4f, "conf:witch>which" to 4f, "conf:principal>principle" to 4f,
            "conf:principle>principal" to 4f, "conf:break>brake" to 4f, "conf:brake>break" to 4f,
            "conf:breath>breathe" to 4f, "conf:breathe>breath" to 4f, "conf:write>right" to 4f, "conf:right>write" to 4f,
            "conf:allowed>aloud" to 4f, "conf:aloud>allowed" to 4f,
        )
    }
}

/**
 * On-device English grammar checker: hand-written candidate rules (pboard's own) propose a fix, and the language
 * model ([lm]: the n-gram pack, or the n-gram + LSTM rescorer) verifies it by comparing the probability of the
 * text around the change with and without the fix. Nothing leaves the device.
 *
 * Scope (precision first; a wrong mark is worse than a missed one):
 *  - real-word confusions (their/there/they're, your/you're, its/it's, then/than, lose/loose, ...);
 *  - subject-verb agreement with a pronoun subject ("he go", "they was", "I has", "she don't");
 *  - verb form after a modal or do ("can goes", "didn't went"), after have ("have went"), "I seen";
 *  - "could of", a/an, doubled function words ("the the"), "more taller", "suppose to", "alot", lowercase "i".
 * Spelling is not checked (that is the spell checker's and autocorrect's job), and dialect spellings are never
 * touched: no rule maps one spelling of a word to another (colour/color, organise/organize).
 *
 * Thread-safety: [check] may run on any thread; the [lm] must be thread-safe (both pboard models are).
 */
class GrammarChecker(
    private val lm: LanguageModel,
    private val ngram: NgramLanguageModel?,
    private val morph: EnglishMorphology,
    val config: GrammarConfig = GrammarConfig(),
) {
    /** A proposed fix before verification: tokens [a, b) of the sentence become [words]. */
    class Candidate(val a: Int, val b: Int, val words: List<String>, val rule: String, val family: String,
                    /** Evidence from the rule itself, in nats, added to the LM's preference. */
                    val prior: Float = 0f) {
        /** Preference for the fix in nats: the LM's, plus [prior] (set by [score]). */
        var delta = Float.NaN
    }

    /** Issues in [text], in text order. Only sentences that intersect [from, to) are checked. */
    fun check(text: String, from: Int = 0, to: Int = text.length): List<GrammarIssue> {
        val out = ArrayList<GrammarIssue>()
        forEachSentence(text, from, to) { toks, sent ->
            var view: LmView? = null
            // the last word of a sentence still being typed may be continued ("I went their" + " way")
            val tail = if (sent.terminated) -1 else (sent.to - 1 downTo sent.from).firstOrNull { toks[it].isWord } ?: -1
            for (c in candidates(toks, sent)) {
                var th = threshold(c)
                if (th == Float.POSITIVE_INFINITY) continue
                if (tail in c.a until c.b && th != Float.NEGATIVE_INFINITY) th += config.tailMargin
                if (th != Float.NEGATIVE_INFINITY) scoreScreened(view ?: lmView(toks, sent).also { view = it }, sent, c, th)
                if (th == Float.NEGATIVE_INFINITY || c.delta >= th) out += issue(text, toks, c)
            }
        }
        return resolveOverlaps(out)
    }

    /** Every candidate with its LM score, accepted or not (evaluation and threshold tuning). */
    fun scoredCandidates(text: String): List<Pair<GrammarIssue, Candidate>> {
        val out = ArrayList<Pair<GrammarIssue, Candidate>>()
        forEachSentence(text, 0, text.length) { toks, sent ->
            val view = lmView(toks, sent)
            for (c in candidates(toks, sent)) {
                score(view, sent, c)
                out += issue(text, toks, c) to c
            }
        }
        return out
    }

    /** Acceptance threshold of a candidate: per rule ("conf:then>than") if set, else per family; +inf = off. */
    fun threshold(c: Candidate): Float = config.thresholds[c.rule] ?: config.thresholds[c.family] ?: Float.POSITIVE_INFINITY

    private inline fun forEachSentence(text: String, from: Int, to: Int, body: (List<TextToken>, TextSentence) -> Unit) {
        val toks = TextTokens.tokenize(text)
        for (s in TextTokens.sentences(text, toks)) {
            if (s.to <= s.from) continue
            if (toks[s.to - 1].end < from || toks[s.from].start > to) continue
            if (!checkable(toks, s)) continue
            body(toks, s)
        }
    }

    /** Skips sentences that are not prose: URLs, e-mail addresses, code, very long runs. */
    private fun checkable(toks: List<TextToken>, s: TextSentence): Boolean {
        if (s.to - s.from > MAX_SENTENCE_TOKENS) return false
        var words = 0
        for (i in s.from until s.to) {
            val t = toks[i]
            if (t.isWord) words++
            if (!t.isWord && t.text[0] in "@#/\\{}<>=_|~`$^*") return false
        }
        return words >= 2
    }

    // ------------------------------------------------------------------------------------------ candidates

    /** All candidate fixes in sentence [s] (token indices are absolute). */
    fun candidates(toks: List<TextToken>, s: TextSentence): List<Candidate> {
        val out = ArrayList<Candidate>()
        for (i in s.from until s.to) {
            val t = toks[i]
            if (!t.isWord) continue
            val w = t.lower
            val prev = prevWord(toks, s, i)
            val next = nextWord(toks, s, i)
            // lowercase "i" ("i think", "i'm"): not "i.e.", not a list marker "i)"
            if (t.text == "i" || t.text.startsWith("i'") && t.text.length in 3..4 && w in I_CONTRACTIONS) {
                val after = toks.getOrNull(i + 1)
                if (!(after != null && after.start == t.end && after.text in listOf(".", ")"))) {
                    out += Candidate(i, i + 1, listOf("I" + t.text.substring(1)), "i_lower", I_LOWER)
                }
            }
            SPACING_FIXES[w]?.let { out += Candidate(i, i + 1, caseLike(t.text, it).split(' '), "spacing", SPACING) }
            // could of / should of / would of / must of / might of
            if (w == "of" && prev != null && toks[prev].lower in MODALS_OF && adjacent(toks, prev, i) &&
                (next == null || toks[next].lower != "course")) {
                out += Candidate(i, i + 1, listOf("have"), "modal_of", MODAL_OF)
            }
            // doubled function word: "the the", "to to"
            // the second copy written the same way, or in lower case ("The the"); "a A C cobra" is a name
            if (next != null && toks[next].lower == w && w in DOUBLE_OK && adjacent(toks, i, next) &&
                (toks[next].text == t.text || toks[next].text == w)) {
                out += Candidate(i, next + 1, listOf(t.text), "double", DOUBLE)
            }
            // a / an
            if ((w == "a" || w == "an") && (t.text[0].isLowerCase() || isSentenceStart(toks, s, i)) && next != null &&
                adjacent(toks, i, next)) {
                val vowel = vowelSound(acronymCase(toks[next].text))
                if (vowel == true && w == "a") out += Candidate(i, i + 1, listOf("an"), "a_an", A_AN)
                if (vowel == false && w == "an") out += Candidate(i, i + 1, listOf("a"), "a_an", A_AN)
            }
            // more taller / most biggest
            if ((w == "more" || w == "most") && next != null && adjacent(toks, i, next) &&
                morph.isComparative(toks[next].lower) && toks[next].lower !in COMPARATIVE_OK) {
                out += Candidate(i, i + 1, emptyList(), "comparative", COMPARATIVE)
            }
            // I'm suppose to / he use to
            if ((w == "suppose" || w == "use") && next != null && toks[next].lower == "to" && prev != null) {
                val p = toks[prev].lower
                // "didn't you use to": after do-support "use" is right
                val doSupport = prevWord(toks, s, prev)?.let { toks[it].lower in DO_FORMS || toks[it].lower == "do" } == true
                val beforeOk = if (w == "suppose") p in BE_FORMS || p == "not" else p in SUBJECT_PRONOUNS && !doSupport
                if (beforeOk) out += Candidate(i, i + 1, listOf(w + "d"), "${w}d_to", SUPPOSED)
            }
            agreement(toks, s, i, prev, out)
            verbForm(toks, s, i, prev, out)
            CONFUSIONS[w]?.let { set ->
                // a sentence-initial capital is ordinary; any other capital may be a name ("Peace River")
                if (t.text[0].isUpperCase() && !isSentenceStart(toks, s, i) && t.text != "I") return@let
                for (alt in set) if (alt != w) {
                    val rule = "conf:$w>$alt"
                    out += Candidate(i, i + 1, listOf(caseLike(t.text, alt)), rule, CONFUSION, contextPrior(rule, toks, i, prev, next))
                }
            }
        }
        return out
    }

    /** Pronoun subject + verb: "he go", "she don't", "they was", "I has", "we is". */
    private fun agreement(toks: List<TextToken>, s: TextSentence, i: Int, prev: Int?, out: MutableList<Candidate>) {
        val subj = toks[i].lower
        if (subj !in SUBJECT_PRONOUNS) return
        if (subj == "i" && toks[i].text != "I" && toks[i].text != "i") return
        if (prev != null && toks[prev].lower in CONJUNCTIONS && coordinatedSubject(toks, s, prev)) return
        // inverted question or object position: "does he go", "let it go", "make it work"
        val third = subj == "he" || subj == "she" || subj == "it"
        if (prev != null) {
            val p = toks[prev].lower
            if (p in AUX_INVERSION || p in BE_FORMS || p in HAVE_FORMS) return
            // "insist that he go", "lest she fall": a base form after these can be a subjunctive
            if (third && p in SUBJUNCTIVE_INTRO) return
            if (subj == "it" && p !in CLAUSE_STARTERS && !(toks.subList(prev + 1, i).any { !it.isWord && it.text == "," })) return
            if (subj == "you" && p !in CLAUSE_STARTERS && p !in SUBJUNCTIVE_INTRO &&
                !(toks.subList(prev + 1, i).any { !it.isWord })) return
        }
        var v = nextWord(toks, s, i) ?: return
        if (!adjacent(toks, i, v)) return
        if (toks[v].lower in AGREEMENT_ADVERBS) {
            v = nextWord(toks, s, v) ?: return
            if (!adjacent(toks, v - 1, v) && toks[v - 1].isWord) return
        }
        if (titleCase(toks, s, v)) return
        val verb = toks[v].lower
        val fix: String? = when {
            verb == "don't" && third -> "doesn't"
            verb == "doesn't" && !third -> "don't"
            verb == "does" && !third && subj != "i" -> "do"
            verb == "have" && third -> "has"
            verb == "has" && !third -> "have"
            verb == "are" && third -> "is"
            verb == "is" && subj == "i" -> "am"
            verb == "are" && subj == "i" -> "am"
            verb == "is" && (subj == "you" || subj == "we" || subj == "they") -> "are"
            verb == "was" && (subj == "you" || subj == "we" || subj == "they") -> "were"
            verb == "am" && subj != "i" -> if (third) "is" else "are"
            else -> null
        }
        if (fix != null) {
            out += Candidate(v, v + 1, listOf(caseLike(toks[v].text, fix)), "sva", SVA)
            return
        }
        if (verb in NOT_LEXICAL_VERBS) return
        val all = morph.verbsWithForm(verb)
        if (third && all.any { it.past == verb || it.participle == verb || it.s3 == verb || it.ing == verb }) return
        if (!third && all.any { it.past == verb || it.base == verb }) return
        for (vb in all) {
            if (vb.base in AMBIGUOUS_PAST_BASE || vb.past == vb.base) continue
            if (third && verb == vb.base && verb != vb.s3 && verb != vb.past && verb != vb.participle) {
                out += Candidate(v, v + 1, listOf(caseLike(toks[v].text, vb.s3)), "sva", SVA)
                return
            }
            if (!third && verb == vb.s3 && verb != vb.base && verb != vb.past) {
                out += Candidate(v, v + 1, listOf(caseLike(toks[v].text, vb.base)), "sva", SVA)
                return
            }
        }
    }

    /**
     * "Paul and I are", "My wife and I have": the conjunction at [conj] joins two subjects (a name or pronoun before it,
     * or it comes within the first words of the sentence), rather than two clauses ("...yesterday and he go").
     */
    private fun coordinatedSubject(toks: List<TextToken>, s: TextSentence, conj: Int): Boolean {
        val before = prevWord(toks, s, conj) ?: return true
        val b = toks[before]
        if (b.lower in OBJECT_OR_SUBJECT_PRONOUNS) return true
        if (b.text[0].isUpperCase() && !isSentenceStart(toks, s, before)) return true
        var words = 0
        for (j in s.from until conj) if (toks[j].isWord) words++ else if (toks[j].text == ",") return false
        return words <= 3
    }

    /** Verb form after a modal / do ("can goes", "didn't went", "did you saw"), after have ("have went"), "I seen". */
    private fun verbForm(toks: List<TextToken>, s: TextSentence, i: Int, prev: Int?, out: MutableList<Candidate>) {
        val w = toks[i].lower
        // "do" counts only in a question ("Do you has"); "I do works" is too rare to be worth the risk
        if (w in MODALS || w in DO_FORMS || (w == "do" && nextWord(toks, s, i)?.let { toks[it].lower in SUBJECT_PRONOUNS } == true)) {
            if (prev != null && toks[prev].lower in DETERMINERS) return // "a can", "free will", "the must"
            // "Will went home": a capitalised Will/May/Can is a name unless a question follows ("Will you ...")
            if (toks[i].text[0].isUpperCase() && w in NAME_MODALS &&
                nextWord(toks, s, i)?.let { toks[it].lower in SUBJECT_PRONOUNS } != true) return
            var v = nextWord(toks, s, i) ?: return
            if (!adjacent(toks, i, v)) return
            if (toks[v].lower == "not" || toks[v].lower in AGREEMENT_ADVERBS) v = nextWord(toks, s, v) ?: return
            // "did you saw": a subject pronoun between do/modal and the verb (questions)
            if (toks[v].lower in SUBJECT_PRONOUNS && toks[v].lower != "it") v = nextWord(toks, s, v) ?: return
            if (!adjacent(toks, v - 1, v) || !toks[v - 1].isWord) return
            if (titleCase(toks, s, v)) return
            val verb = toks[v].lower
            if (verb in NOT_LEXICAL_VERBS) return
            val vb = morph.verbsWithForm(verb).firstOrNull { it.base != verb } ?: return
            if (morph.verbsWithForm(verb).any { it.base == verb }) return // also a base form ("saw", "found")
            val nonBase = (verb == vb.s3 || verb == vb.past) && verb != vb.base
            if (!nonBase) return
            // (a plural noun cannot follow a modal either: "can helps" is an error even though "helps" is a noun)
            out += Candidate(v, v + 1, listOf(caseLike(toks[v].text, vb.base)), if (w in MODALS) "modal_form" else "do_form", MODAL_FORM)
            return
        }
        if (w in HAVE_PERFECT) {
            var v = nextWord(toks, s, i) ?: return
            if (!adjacent(toks, i, v)) return
            if (toks[v].lower in PERFECT_ADVERBS) v = nextWord(toks, s, v) ?: return
            if (!adjacent(toks, v - 1, v) || !toks[v - 1].isWord) return
            if (titleCase(toks, s, v)) return
            val verb = toks[v].lower
            val forms = morph.verbsWithForm(verb)
            if (forms.isEmpty() || forms.any { it.participle == verb }) return
            val vb = forms.firstOrNull { it.past == verb && it.participle != verb } ?: return
            out += Candidate(v, v + 1, listOf(caseLike(toks[v].text, vb.participle)), "perfect", PERFECT)
            return
        }
        if (w in SUBJECT_PRONOUNS && w != "it" && w != "you") {
            if (prev != null && (toks[prev].lower in AUX_INVERSION || toks[prev].lower in HAVE_FORMS)) return
            val v = nextWord(toks, s, i) ?: return
            if (!adjacent(toks, i, v)) return
            if (titleCase(toks, s, v)) return
            val verb = toks[v].lower
            val past = PARTICIPLE_TO_PAST[verb] ?: return
            out += Candidate(v, v + 1, listOf(caseLike(toks[v].text, past)), "pron_pp", PRON_PP)
        }
    }

    /**
     * Word-class evidence the LM underrates because the error is common in its training text: a possessive cannot be
     * followed by an article, an adverb or a verb ("its a", "your welcome", "your going"), "their" cannot start a
     * clause before a verb ("their is"), and "then" never follows a comparative ("taller then").
     */
    private fun contextPrior(rule: String, toks: List<TextToken>, i: Int, prev: Int?, next: Int?): Float {
        val n = next?.let { toks[it].lower }
        val p = prev?.let { toks[it].lower }
        val hit = when (rule) {
            "conf:its>it's", "conf:your>you're" -> n != null && (n in NOT_AFTER_POSSESSIVE || isVerbIng(n))
            "conf:their>there" -> n != null && n in COPULA_AND_MODALS
            "conf:their>they're" -> n != null && (n in NOT_AFTER_POSSESSIVE || isVerbIng(n)) && n !in COPULA_AND_MODALS
            "conf:then>than" -> p != null && (morph.isComparative(p) || p in COMPARISON_WORDS)
            "conf:to>too" -> (n == null && toks.getOrNull(i + 1)?.let { !it.isWord && it.text[0] in ".!?" } == true) || n in TOO_BEFORE
            else -> false
        }
        return if (hit) CONTEXT_PRIOR else 0f
    }

    /** An -ing form of a verb that is not also a common noun ("going", "doing"; not "wedding", "building"). */
    private fun isVerbIng(w: String): Boolean =
        w.endsWith("ing") && morph.verbsWithForm(w).any { it.ing == w } && w !in NOUN_INGS

    // ------------------------------------------------------------------------------------------ verification

    /** The LM's view of sentence [s]: words and commas, with their token indices. */
    class LmView internal constructor(val words: List<String>, val tokIndex: IntArray)

    fun lmView(toks: List<TextToken>, s: TextSentence): LmView {
        val words = ArrayList<String>()
        val idx = ArrayList<Int>()
        for (i in s.from until s.to) {
            val t = toks[i]
            if (t.isWord) { words += t.text; idx += i }
            else if (t.text == "," || t.text == ";" || t.text == ":") { words += ","; idx += i }
        }
        return LmView(words, idx.toIntArray())
    }

    /** Sets [Candidate.delta]: log P(text around the fix) - log P(text around the original), with [model]. */
    fun score(view: LmView, s: TextSentence, c: Candidate, model: LanguageModel = lm) {
        val w = view.words
        val la = view.tokIndex.indexOfFirst { it >= c.a }.let { if (it < 0) w.size else it }
        val lb = view.tokIndex.indexOfFirst { it >= c.b }.let { if (it < 0) w.size else it }
        val alt = ArrayList<String>(w.size + c.words.size)
        alt.addAll(w.subList(0, la)); alt.addAll(c.words); alt.addAll(w.subList(lb, w.size))
        val k = config.window
        val origEnd = minOf(lb + k, w.size)
        val altEnd = minOf(la + c.words.size + k, alt.size)
        val eos = s.terminated && origEnd == w.size
        c.delta = windowLogProb(model, alt, la, altEnd, eos) - windowLogProb(model, w, la, origEnd, eos) + c.prior
    }

    /**
     * Scores [c] for a decision against [threshold]: first with the n-gram alone (microseconds), and with [lm] (the
     * LSTM mixture, milliseconds per position) only when the n-gram does not already put it more than
     * [GrammarConfig.screenMargin] nats below the threshold.
     */
    private fun scoreScreened(view: LmView, s: TextSentence, c: Candidate, threshold: Float) {
        val g = ngram
        if (g != null && lm !== g && config.screenMargin.isFinite()) {
            score(view, s, c, g)
            if (c.delta < threshold - config.screenMargin) return
        }
        score(view, s, c)
    }

    private fun windowLogProb(model: LanguageModel, words: List<String>, from: Int, to: Int, eos: Boolean): Float {
        var lp = 0f
        for (j in from until to) {
            if (words[j] == ",") continue
            lp += model.logProb(words.subList(0, j), words[j])
        }
        if (eos && ngram != null) {
            val ids = ngram.sentenceIds(words)
            lp += ngram.ngramLogProb(ids, ids.size, LmPack.EOS)
        }
        return lp
    }

    // ------------------------------------------------------------------------------------------ output

    /**
     * How [issue] is shown: a non-empty range of [text] to mark and the text that replaces it, Gboard style. A
     * deletion is shown on the word pair ("The the" -> "The", "more taller" -> "taller"); a/an and "could of" take
     * the next word along ("a apple" -> "an apple", "of done" -> "have done"); everything else is the word itself.
     */
    fun displayEdit(text: String, issue: GrammarIssue): DisplayEdit {
        fun wordAfter(pos: Int): IntRange? {
            var i = pos
            while (i < text.length && text[i] == ' ') i++
            if (i == pos || i >= text.length || !text[i].isLetterOrDigit()) return null
            var j = i
            while (j < text.length && (text[j].isLetterOrDigit() || text[j] == '\'' || text[j] == '’')) j++
            return i until j
        }
        if (issue.replacement.isEmpty()) {
            // the doubled word: keep the pair; "more taller": the comparative after it
            if (issue.rule == "double" && issue.start < issue.markStart) {
                // the deletion runs from the end of the first copy: show both copies, keep the first
                var first = issue.start
                while (first > 0 && (text[first - 1].isLetterOrDigit() || text[first - 1] == '\'')) first--
                return DisplayEdit(first, issue.markEnd, text.substring(first, issue.start))
            }
            val next = wordAfter(issue.markEnd) ?: return DisplayEdit(issue.markStart, issue.markEnd, "")
            return DisplayEdit(issue.markStart, next.last + 1, text.substring(next))
        }
        if (issue.rule == "a_an" || issue.rule == "modal_of") {
            val next = wordAfter(issue.end)
            if (next != null) return DisplayEdit(issue.start, next.last + 1, issue.replacement + text.substring(issue.end, next.last + 1))
        }
        return DisplayEdit(issue.start, issue.end, issue.replacement)
    }

    /** A marked range [start, end) and its replacement (see [displayEdit]). */
    data class DisplayEdit(val start: Int, val end: Int, val replacement: String)

    private fun issue(text: String, toks: List<TextToken>, c: Candidate): GrammarIssue {
        // trim tokens that the fix keeps ("the the" -> "the": only the second one goes)
        var a = c.a
        var b = c.b
        var words = c.words
        while (words.isNotEmpty() && a < b && toks[a].lower == words.first().lowercase() && toks[a].text == words.first()) {
            a++; words = words.drop(1)
        }
        while (words.isNotEmpty() && a < b && toks[b - 1].text == words.last()) {
            b--; words = words.dropLast(1)
        }
        val rep = words.joinToString(" ")
        return if (rep.isEmpty() && a < b) {
            // deletion: take the space before the word with it (or after, at the start of the text)
            val start = if (a > 0 && toks[a - 1].end <= toks[a].start) toks[a - 1].end else toks[a].start
            val end = if (start == toks[a].start && b < toks.size) toks[b].start else toks[b - 1].end
            GrammarIssue(start, end, "", c.rule, c.delta, toks[a].start, toks[b - 1].end)
        } else {
            GrammarIssue(toks[a].start, toks[b - 1].end, rep, c.rule, c.delta)
        }
    }

    /** Keeps the best-scoring issue where several overlap (e.g. their -> there and their -> they're). */
    private fun resolveOverlaps(issues: List<GrammarIssue>): List<GrammarIssue> {
        val sorted = issues.sortedByDescending { it.score }
        val kept = ArrayList<GrammarIssue>()
        for (x in sorted) if (kept.none { it.markStart < x.markEnd && x.markStart < it.markEnd }) kept += x
        return kept.sortedBy { it.start }
    }

    // ------------------------------------------------------------------------------------------ helpers

    private fun prevWord(toks: List<TextToken>, s: TextSentence, i: Int): Int? {
        var j = i - 1
        while (j >= s.from) { if (toks[j].isWord) return j; if (toks[j].text != ",") return null; j-- }
        return null
    }

    private fun nextWord(toks: List<TextToken>, s: TextSentence, i: Int): Int? {
        val j = i + 1
        return if (j < s.to && toks[j].isWord) j else null
    }

    /** Tokens [i] and [j] are consecutive words with only whitespace between them. */
    private fun adjacent(toks: List<TextToken>, i: Int, j: Int) = j == i + 1 && toks[i].isWord && toks[j].isWord

    /** A capitalised word inside a sentence ("have Beat the Retreat"): a title or a name, not a verb to fix. */
    private fun titleCase(toks: List<TextToken>, s: TextSentence, i: Int): Boolean {
        val t = toks[i].text
        return t[0].isUpperCase() && !t.all { !it.isLetter() || it.isUpperCase() } && !isSentenceStart(toks, s, i)
    }

    private fun isSentenceStart(toks: List<TextToken>, s: TextSentence, i: Int): Boolean {
        for (j in s.from until i) if (toks[j].isWord) return false
        return true
    }

    private fun caseLike(original: String, word: String): String = when {
        original.length > 1 && original.all { !it.isLetter() || it.isUpperCase() } -> word.uppercase()
        original[0].isUpperCase() -> word.replaceFirstChar { it.uppercase() }
        else -> word
    }

    /** Whether [word] starts with a vowel sound (a/an), or null when unsure. */
    /**
     * A short word typed in lower case that the lexicon only knows in capitals is an acronym ("nfl" -> "NFL"), read
     * letter by letter; "us" after an article is the country. Other words are returned unchanged.
     */
    private fun acronymCase(word: String): String {
        if (word == "us") return "US"
        if (word.length > 4 || word.any { !it.isLowerCase() } || ngram == null) return word
        val id = ngram.lexIdOf(word)
        if (id < 0) return word
        val canon = ngram.word(id)
        return if (canon.length == word.length && canon.all { it.isUpperCase() }) canon else word
    }

    internal fun vowelSound(word: String): Boolean? {
        if (word.isEmpty()) return null
        val c0 = word[0]
        if (c0.isDigit()) {
            return when {
                c0 == '8' -> true
                word == "11" || word == "18" || word.startsWith("11-") || word.startsWith("18-") -> true
                word.length >= 2 && (word.startsWith("11") || word.startsWith("18")) && word.length % 3 == 2 -> true
                else -> false
            }
        }
        if (!c0.isLetter()) return null
        val letters = word.takeWhile { it.isLetter() }
        if (letters.length in 1..4 && letters.all { it.isUpperCase() } && word.length == letters.length) {
            // letter names: an FBI, an MP, a URL, a CV; skip longer acronyms read as words (NASA, NATO)
            if (letters.length == 1 && word.length == 1 && c0 !in "AEFHILMNORSX") return false
            if (letters.length >= 4) return null
            return c0 in "AEFHILMNORSX"
        }
        val w = word.lowercase()
        if (w.length == 1) return null
        if (SILENT_H.any { w.startsWith(it) }) return true
        if (H_DIALECT.any { w.startsWith(it) }) return null
        if (w[0] in "aeio") {
            if (w == "one" || w.startsWith("one-") || w == "once" || w == "oneself" || w.startsWith("eu") || w.startsWith("ewe")) return false
            if (w.startsWith("ouija")) return false
            return true
        }
        if (w[0] == 'u') {
            if (w.startsWith("uni")) {
                if (w.length > 3 && w[3] in "nmdlb") return !w.startsWith("unil") // un-informed, un-important; uni-lateral
                return false // university, unique, unit
            }
            if (U_YOO.any { w.startsWith(it) }) return false
            return true
        }
        if (w[0] == 'x' || w[0] == 'y' && w.startsWith("ytt")) return null
        return false
    }

    companion object {
        const val SPACING = "spacing"
        const val I_LOWER = "i_lower"
        const val MODAL_OF = "modal_of"
        const val DOUBLE = "double"
        const val A_AN = "a_an"
        const val SVA = "sva"
        const val MODAL_FORM = "modal_form"
        const val PERFECT = "perfect"
        const val PRON_PP = "pron_pp"
        const val COMPARATIVE = "comparative"
        const val SUPPOSED = "supposed"
        const val CONFUSION = "confusion"

        const val MAX_SENTENCE_TOKENS = 80
        /** Nats added by [contextPrior]. */
        const val CONTEXT_PRIOR = 3f

        private val NOT_AFTER_POSSESSIVE = setOf("a", "an", "the", "not", "so", "very", "too", "really", "just",
            "gonna", "always", "never", "still", "probably", "already", "also", "actually", "totally", "absolutely",
            "welcome", "ok", "okay", "fine", "sure", "been", "all", "about", "only", "getting", "going",
            "being", "doing", "kidding", "joking", "crazy", "amazing", "awesome", "funny", "late", "early", "done",
            "ready", "wrong", "true", "raining", "snowing", "cold", "hot", "hard", "easy", "fun", "nice",
            "great", "good", "bad", "weird", "sad", "important", "possible", "impossible", "interesting",
            "over", "my", "your", "his", "her", "our", "their", "this", "that", "these", "those", "what",
            "how", "who", "where", "when", "why", "in", "on", "at", "from", "with", "here", "there")
        private val COPULA_AND_MODALS = setOf("is", "are", "was", "were", "isn't", "aren't", "wasn't", "weren't", "will",
            "won't", "would", "could", "should", "must", "might", "may", "has", "have", "had", "seems", "seem", "goes")
        private val COMPARISON_WORDS = setOf("more", "less", "rather", "other", "else", "fewer", "better", "worse")
        private val TOO_BEFORE = setOf("much", "many", "late", "early", "bad", "big", "small", "far", "long", "little",
            "hard", "busy", "tired", "hot", "cold", "expensive", "soon", "fast", "slow", "young", "old", "short")
        private val NOUN_INGS = setOf("wedding", "building", "morning", "evening", "feeling", "meeting", "painting",
            "ceiling", "clothing", "thing", "nothing", "something", "anything", "everything", "king", "ring", "sing",
            "string", "spring", "wing", "bring", "earring", "darling", "pudding", "stuffing", "frosting", "setting",
            "beginning", "ending", "rating", "training", "parking", "booking", "shopping", "housing", "funding",
            "reading", "writing", "drawing", "hearing", "listing", "posting", "recording", "warning", "opening")

        private val I_CONTRACTIONS = setOf("i'm", "i've", "i'll", "i'd")

        /** Run-together words people type often; each fix is a space (none is a dialect spelling). */
        val SPACING_FIXES = mapOf(
            "alot" to "a lot", "aswell" to "as well", "atleast" to "at least", "infront" to "in front",
            "eachother" to "each other", "abit" to "a bit", "alittle" to "a little", "ofcourse" to "of course",
            "inspite" to "in spite", "thankyou" to "thank you", "everytime" to "every time", "noone" to "no one",
        )

        private val MODALS_OF = setOf("could", "should", "would", "must", "might", "may", "couldn't", "shouldn't",
            "wouldn't", "mustn't", "mightn't")

        /** Function words never meant twice in a row ("that that", "had had", "is is" can be; those are absent). */
        private val DOUBLE_OK = setOf("the", "a", "an", "to", "of", "in", "on", "at", "for", "with", "and", "or",
            "but", "are", "was", "were", "be", "been", "i", "he", "she", "we", "they", "my", "your", "his", "our",
            "their", "this", "these", "those", "from", "by", "if", "will", "would", "can", "could", "should",
            "have", "has", "it's", "i'm", "you're", "they're", "we're", "is", "am", "an", "as", "into", "than",
            "about", "when", "what", "which", "who", "it", "me", "him", "them", "us", "not", "do", "does", "did")
            .minus(setOf("is")) // "what it is is ..."

        private val COMPARATIVE_OK = setOf("more", "most", "less", "least", "fewer", "lesser", "later", "latter",
            "further", "farther", "sooner", "earlier", "best", "better", "worse", "worst")
            .minus(setOf("better", "worse", "best", "worst")) // "more better" is a real error

        private val SUBJECT_PRONOUNS = setOf("i", "you", "he", "she", "it", "we", "they")

        private val BE_FORMS = setOf("am", "is", "are", "was", "were", "be", "been", "being", "i'm", "you're",
            "he's", "she's", "it's", "we're", "they're", "isn't", "aren't", "wasn't", "weren't", "'m", "'re")

        private val HAVE_FORMS = setOf("have", "has", "had", "having")

        /** Words before a pronoun that make it an inverted subject or an object ("does he go", "let it go"). */
        private val AUX_INVERSION = setOf("do", "does", "did", "don't", "doesn't", "didn't", "will", "would", "can",
            "could", "should", "shall", "may", "might", "must", "won't", "wouldn't", "can't", "couldn't",
            "shouldn't", "mustn't", "mightn't", "shan't", "let", "make", "makes", "made", "making", "help", "helps",
            "helped", "helping", "see", "saw", "seen", "watch", "watched", "hear", "heard", "have", "has", "had",
            "than", "as", "nor", "is", "are", "was", "were", "am", "isn't", "aren't", "wasn't", "weren't")

        /** Before a third-person pronoun, a base form may be a subjunctive ("I insist that he go", "lest she fall"). */
        private val SUBJUNCTIVE_INTRO = setOf("that", "lest", "if", "whether", "unless")

        /** Words after which "it"/"you" starts a clause (otherwise they may be objects: "make it work"). */
        private val CLAUSE_STARTERS = setOf("and", "but", "so", "because", "cause", "cos", "if", "when", "then",
            "though", "although", "since", "while", "until", "unless", "or", "once", "whether", "yet")

        /** "Paul and I are", "he or she goes": a coordinated subject agrees as a whole. */
        private val CONJUNCTIONS = setOf("and", "or", "nor", "&")
        private val OBJECT_OR_SUBJECT_PRONOUNS = setOf("i", "me", "you", "he", "him", "she", "her", "we", "us", "they",
            "them", "it")

        private val AGREEMENT_ADVERBS = setOf("always", "never", "often", "usually", "sometimes", "also", "really",
            "just", "only", "still", "even", "actually", "probably", "rarely", "seldom", "already", "certainly",
            "definitely", "totally", "generally", "normally", "obviously", "literally", "basically", "honestly")

        private val PERFECT_ADVERBS = setOf("not", "never", "already", "just", "ever", "also", "always", "recently",
            "finally", "since", "all", "both", "really", "probably", "obviously", "certainly", "definitely")

        private val MODALS = setOf("can", "could", "will", "would", "shall", "should", "may", "might", "must",
            "can't", "cannot", "couldn't", "won't", "wouldn't", "shouldn't", "mustn't", "mightn't", "shan't")
        private val DO_FORMS = setOf("did", "didn't", "does", "doesn't", "don't")
        private val NAME_MODALS = setOf("will", "may", "can", "must", "might")

        /** have/has/had + past simple; 'd and 's are left out (would/had, is/has). */
        private val HAVE_PERFECT = setOf("have", "has", "had", "i've", "you've", "we've", "they've", "haven't",
            "hasn't", "hadn't", "should've", "could've", "would've", "might've", "must've")

        private val DETERMINERS = setOf("a", "an", "the", "this", "that", "my", "your", "his", "her", "our",
            "their", "its", "free", "good", "ill", "last", "of", "every", "any", "some", "no")

        /** Common words the verb table also lists, never verbs after a pronoun or modal in practice. */
        private val NOT_LEXICAL_VERBS = setOf("be", "do", "have", "need", "dare", "used", "ought", "better",
            "rather", "kind", "sort", "lot", "bit", "too", "very", "so", "not", "also", "even", "all", "both",
            "each", "only", "just", "as", "like", "well", "back", "home", "there", "here", "then", "now")

        /** Bases whose past is the same in one dialect ("she fit", "it quit", "he wet"): never agreement errors. */
        private val AMBIGUOUS_PAST_BASE = setOf("fit", "quit", "wet", "knit", "sweat", "spit", "bet", "wed", "rid",
            "slit", "forecast", "broadcast", "input", "output", "upset", "beat", "light", "plead", "dive", "sneak",
            "learn", "spell", "dream", "burn", "smell", "spill", "spoil", "kneel", "lean", "leap", "dwell")

        /** Participles used as a past tense after a bare pronoun ("I seen it"); the fix is the past simple. */
        private val PARTICIPLE_TO_PAST = mapOf("seen" to "saw", "taken" to "took", "eaten" to "ate",
            "written" to "wrote", "given" to "gave", "spoken" to "spoke", "chosen" to "chose", "begun" to "began",
            "drunk" to "drank", "sung" to "sang", "swum" to "swam", "rung" to "rang", "forgotten" to "forgot",
            "stolen" to "stole", "driven" to "drove", "ridden" to "rode", "fallen" to "fell", "thrown" to "threw",
            "flown" to "flew", "grown" to "grew", "drawn" to "drew", "shown" to "showed", "woken" to "woke")

        private val SILENT_H = listOf("hour", "honest", "honor", "honour", "heir")
        private val H_DIALECT = listOf("herb", "historic", "hotel", "homage", "humble")
        private val U_YOO = listOf("use", "usa", "usu", "uten", "util", "utop", "ura", "ure", "uri", "uro", "ubiq",
            "ukr", "ufo", "uvu", "ukulele", "unanim", "uganda", "ugand")

        /**
         * Real-word confusion sets: each member may be a mistake for another. None mixes dialect spellings. Which
         * sets are on, and how sure the LM must be, is in [GrammarConfig].
         */
        val CONFUSION_SETS: List<List<String>> = listOf(
            listOf("their", "there", "they're"), listOf("your", "you're"), listOf("its", "it's"),
            listOf("whose", "who's"), listOf("then", "than"), listOf("lose", "loose"), listOf("affect", "effect"),
            listOf("affects", "effects"), listOf("affected", "effected"), listOf("to", "too"),
            listOf("were", "where", "we're"), listOf("weather", "whether"), listOf("accept", "except"),
            listOf("advice", "advise"), listOf("quiet", "quite"), listOf("passed", "past"), listOf("peace", "piece"),
            listOf("hear", "here"), listOf("of", "off"), listOf("breath", "breathe"), listOf("brake", "break"),
            listOf("principal", "principle"), listOf("desert", "dessert"), listOf("complement", "compliment"),
            listOf("know", "no"), listOf("knew", "new"), listOf("buy", "by", "bye"), listOf("wear", "where"),
            listOf("which", "witch"), listOf("allowed", "aloud"), listOf("bored", "board"), listOf("threw", "through"),
            listOf("lead", "led"), listOf("sight", "site"), listOf("role", "roll"), listOf("week", "weak"),
            listOf("whole", "hole"), listOf("write", "right"), listOf("loose", "lose"), listOf("thought", "though"),
            listOf("are", "our"), listOf("an", "and"), listOf("weight", "wait"), listOf("meat", "meet"),
        ).distinctBy { it.sorted() }

        private val CONFUSIONS: Map<String, List<String>> = HashMap<String, List<String>>().apply {
            for (set in CONFUSION_SETS) for (w in set) put(w, (get(w).orEmpty() + set).distinct())
        }
    }
}
