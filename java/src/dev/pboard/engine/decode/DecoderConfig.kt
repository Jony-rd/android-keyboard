package dev.pboard.engine.decode

/**
 * Tunables of the tap decoder. Costs are negative log-probabilities in nats so they combine directly
 * with language-model log probabilities: total = spatial + edits − [lmWeight]·ln P(word | context).
 * The relative ordering of edit costs (same-letter omission < omission ≈ transposition < insertion) follows
 * what typing studies and the AOSP LatinIME scoring parameters suggest.
 */
data class TapParams(
    val lmWeight: Float = 1.0f,
    /** Typed word misses a letter ("helo" for "hello" is the same-letter case below). */
    val omission: Float = 4.0f,
    /** Missed one of a doubled letter ("leter" -> "letter"). */
    val omissionSameLetter: Float = 2.6f,
    /** Missing first letter: rare. */
    val omissionFirst: Float = 8.0f,
    /** Extra tap that belongs to no letter of the word. */
    val insertion: Float = 5.5f,
    /** Stray tap before the first letter. */
    val insertionFirst: Float = 7.0f,
    /** Extra tap repeating the previous letter ("helllo"). */
    val insertionSameLetter: Float = 3.2f,
    /** Two adjacent letters swapped ("teh", "recieve"). */
    val transposition: Float = 4.6f,
    /** Transpositions are only tried when both taps are this close (spatial cost) to the swapped letters. */
    val transpositionMatchCost: Float = 4.5f,
    /** Skipping a silent character (apostrophe, hyphen) the user did not type. */
    val silentSkip: Float = 0.05f,
    /** Typing the unaccented key for an accented letter ('e' for 'é'); keeps exact spellings ahead on ties. */
    val foldPenalty: Float = 0.4f,
    /** Prior cost that a touch on the bottom letter row was meant for the space bar below it (missed space). */
    val missedSpacePenalty: Float = 3.0f,
    /** Touches further above the space bar than this (key widths) are never a missed space. */
    val missedSpaceMaxDistance: Float = 1.0f,
    /**
     * Extra cost, on top of the missed-space cost, of skipping a touch just above the space bar as a stray
     * tap inside a word (+inf disables it).
     */
    val strayTouchExtra: Float = Float.POSITIVE_INFINITY,
    /** Inputs of at least this many taps may use [maxEditsLong] edits (several extra/missing letters). */
    val longInputLength: Int = 8,
    val maxEditsLong: Int = 4,
    /** Interior word splits are only tried when the best single word costs at least this per 4 taps (it decodes poorly). */
    val missedSpaceSplitMinCost: Float = 3.0f,
    /** Interior word splits are only tried at touches whose missed-space cost is below this. */
    val missedSpaceMaxSplitCost: Float = 4.5f,
    /** Offering a real-word confusion of a valid literal ("their" for "there"), see [TextRules.confusables]. */
    val confusionCost: Float = 3.5f,
    /** Cap on substituting one vowel for another (spelling errors: "definately", "seperate"). */
    val vowelSubstitution: Float = 6.0f,
    /** Added to completions (words longer than what was typed) when ranking them against full matches. */
    val completionPenalty: Float = 0.6f,
    /** Matches whose spatial cost exceeds this are not expanded (saves time; the cap already bounds them). */
    val maxMatchCost: Float = 9f,
    val beamSize: Int = 160,
    /** Hypotheses whose look-ahead priority is this much worse than the layer's best are dropped. */
    val beamWidth: Float = 16f,
    /** Weight of the unigram look-ahead (best prior below a node) used only for beam pruning. */
    val lookaheadWeight: Float = 0.8f,
    val maxFullCandidatesRescored: Int = 48,
    val completionHyps: Int = 10,
    val completionsPerHyp: Int = 6,
    val contextPredictions: Int = 40,
    val maxCandidates: Int = 8,
    /**
     * Split-word repair: a word the space bar cut in two ("hel lo", "som ething") is joined again once its second part
     * completes a word, when one word over both parts' taps beats the two parts by [joinMargin] nats after the cost
     * that the space was accidental: [joinBase] plus [joinPerKey] per key width the space touch lay inside the bar.
     * Measured on Gboard 18.6 (emulator farm, >= 400 dp, a clean tap in the bar's middle): "hel lo" -> "hello", "beau tiful" ->
     * "beautiful", "som ething" -> "something", "may e" -> "maybe" on the space after the second part (the strip shows
     * "hel lo | hello*" before it); "the re", "in to", "hello world", "th is", "wor ld" stay. Human TUNE (en-GB,
     * neural, resolved taps): VTAP words split by a tap-time space 22, joined back 8 (all right); TSI WER and false
     * corrections unchanged (looser costs joined 9 but cost a TSI word).
     */
    val joinSplitWords: Boolean = true,
    val joinBase: Float = 2.0f,
    val joinPerKey: Float = 6.0f,
    val joinMargin: Float = 0f,
    /**
     * Two words stay two words (Gboard keeps "the re", "in to"): a join needs a first part that is no common word
     * (lexicon prior below this: "hel", "som") or a second part that no lexicon has ("ething", the "e" of "may e").
     */
    val joinCommonPrior: Float = -11f,
    /** The joined word explains both parts' taps with at most this many edits. */
    val joinMaxEdits: Int = 1,
)

/** When to replace the literal typed word on space (see [AutocorrectPolicy]). */
data class AutocorrectParams(
    /** When false the would-be correction is still the first candidate, but space commits the literal. */
    val enabled: Boolean = true,
    /**
     * Never autocorrect an unknown word the user capitalised mid-sentence into another word (likely a name). Gboard
     * 18.6 does correct it ("I met Jhon" -> "John", conformance farm-440): off; human TUNE unchanged either way.
     */
    val protectCapitalisedUnknown: Boolean = false,
    /**
     * A word the user capitalised mid-sentence is most likely a name: it may be corrected into
     * another name or a capitalised form ("I met Jhon" -> "John", as Gboard 18.6; "Iphone" -> "iPhone"), never into an
     * ordinary lower-case word shown with a capital ("I met Tolu" gave "You", "Soyeon" "Someone", "Xin" "In"). Gboard
     * treats capitalised out-of-vocabulary words separately, and a
     * daily user of both reports Gboard 18.6 leaving such names alone (CO-09). At a sentence start
     * capitals carry no signal ([protectNamesAtSentenceStart]); in CAP_WORDS fields every word is capitalised, so there
     * no word is corrected into an ordinary word either. Human TUNE (TSI / VTAP): WER and false corrections unchanged.
     */
    val capitalisedOnlyToNames: Boolean = true,
    /**
     * Log prior for "the user meant exactly this unknown string" (names, slang…): a new-word base plus a
     * per-character cost, as a character-level model would give — long non-words are far more likely
     * typos than inventions, short ones may well be intended.
     */
    val oovLogPriorBase: Float = -5.5f,
    val oovLogPriorPerChar: Float = -2.4f,
    /** Required advantage (nats) of the best word over an unknown literal. */
    val marginOov: Float = 1.0f,
    /**
     * Required advantage of another word over a literal that is itself a valid word. On the human VTAP set Gboard 18.6
     * fixes rare words typed for a neighbour-key word the context wants ("we ate" -> "we are", "hid" -> "his"); paired
     * device run (emulator farm, TEST): 111 words Gboard got and pboard did not, 40 of them short words with one wrong key, mostly
     * valid literals kept. Human TUNE (neural, resolved taps; WER TSI / VTAP, false corrections TSI / VTAP), with
     * [marginValidShort] 8: 7.0 2.08 / 4.71 %, 0 / 5; 6.0 2.02 / 4.46, 0 / 5; 5.5 1.99 / 4.37, 0 / 5; 5.0 1.96 / 4.35,
     * 0 / 7 ("neatest" -> "nearest"); 4.0 1.84 / 4.19, 0 / 8. 5.5: the most with no new false correction.
     */
    val marginValid: Float = 5.5f,
    /**
     * Same, for a valid literal of two letters ("yo" -> "to" after "want", "ne" -> "me" after "call"), which used to be
     * never replaced. Human TUNE with [marginValid] 5.5 (WER TSI / VTAP, false corrections 0 / 5 throughout): never
     * 2.08 / 4.80 % (at 7.0), 8 1.99 / 4.37, 6 1.93 / 4.33, 5 1.90 / 4.33, 4 1.87 / 4.33, 3 1.87 / 4.33 %.
     */
    val marginValidShort: Float = 4.0f,
    /** Same, for a valid one-letter literal ("o" -> "I" at a sentence start); infinite: never replaced (TUNE: 10 or 14 change nothing). */
    val marginValidLetter: Float = Float.POSITIVE_INFINITY,
    /** Margin for a variant when the context does not favour it beyond its context-free prior. */
    val marginVariantContextFree: Float = 3.0f,
    /** Same, when the other word has the same keys (apostrophe/case variant: "ill" -> "I'll") and the context favours it. */
    val marginVariant: Float = 0.0f,
    /** Extra margin for 2-letter inputs; single letters are only ever case-corrected ("i" -> "I"). */
    val shortExtraMargin: Float = 2.5f,
    /** A correction may not cost more than this per typed key (spatial + edits) — too far from what was typed. */
    val maxCostPerKey: Float = 4.5f,
    /**
     * CO-09 at a sentence start, where capitals carry no signal: a capitalised unknown word that reads like a
     * word of the language and was typed precisely is not autocorrected into a proper noun ("Mikey" stays
     * "Mikey", not "Mike"): replacing one name by another is rarely right, while misspellings of common
     * words ("Teh", "Recieve") are still fixed. Mid-sentence, see [protectCapitalisedUnknown].
     */
    val protectNamesAtSentenceStart: Boolean = true,
    /** Longest input (keys) it applies to: given names are short, long capitalised misspellings ("Feburary") are fixed. */
    val nameMaxLength: Int = 5,
    /** "Reads like a word": its least likely letter transition (letter-trigram model of the lexicon) is at least this, in nats. */
    val nameMinLetterLogProb: Float = -6.0f,
    /** "Typed precisely": the literal key path costs at most this per key (spatial nats). */
    val nameMaxCostPerKey: Float = 1.0f,
    /**
     * Autocorrect a word re-opened from its text alone (no touches: backspaced or moved into). Gboard does not
     * (SU-04: "i like teh " + backspace shows `teh | the | they`, nothing bold); its alternatives are only offered.
     * Typing more letters turns it into an ordinary word again. The text-level eval sets switch it on to measure
     * correction quality from text.
     */
    val reopenedWords: Boolean = false,
    /**
     * Never correct to a strict prefix of the typed word ("hel" -> "he") unless every dropped letter looks like a
     * stray tap: a repeat of the letter before it ("andd", "iss"), a key next to it ("andf"), a touch just above the
     * space bar ("hellob") or a sloppy touch (see [keepTypedTailCleanRadius]). Otherwise the extra letters were
     * meant; such prefixes are also ranked after the completions of what was typed ("hel | hello"). Human TUNE
     * (VTAP): without the clean-touch exception this cost 0.2 pt WER ("visiti", "bef", "thand" typed for
     * visit/be/than), with it about one word in 4,400.
     */
    val keepTypedTail: Boolean = true,
    /** A dropped letter's touch is clean (so meant) within this distance (key widths) of its key's centre. */
    val keepTypedTailCleanRadius: Float = 0.3f,
    /**
     * A correction that only adds letters to what was typed ("hel" -> "hell", "he'll") is a completion in disguise:
     * it replaces the typed word only when the language model prefers it by at least [extendMargin] nats to every other
     * word starting with those letters, its own longer forms aside ("wan" -> "want" despite "wanted", "wha" -> "what";
     * not "hel" -> "hell" while "help" is about as likely). Human TUNE, margin 0.5 / 1 / 2: TSI WER +0 / +1 / +1
     * words of 3,300, VTAP +1 word each; the first version (any likelier completion, inflections included) lost
     * "wan" -> "want", "tak" -> "take", "revie" -> "review".
     */
    val extendOnlyWhenLikeliest: Boolean = true,
    /**
     * Always capitalise a word typed in lower case to its capitalised form. Off, as Gboard 18.6: only when the lower-case
     * form is no word and the field asks for capitals ([dev.pboard.engine.Context.capitalsAllowed]: "brian" -> "Brian",
     * "friday" -> "Friday"; "dome", "beau" stay, "london" stays in a no-caps field; conformance farm passes 18:23, 21:24).
     */
    val capitaliseLowercase: Boolean = false,
    /**
     * Completion on space: a word typed only in part ("ti", "goi", "exc") is completed by space when the context makes
     * one completion overwhelmingly likely (Gboard 18.6 bolds "time" for "The first ti", "going" for "goi"). Only when
     * the typed letters are no common word (lexicon prior below [completeCommonPrior]) and the completion beats every
     * other reading, the literal included, by [completeMargin] nats.
     *
     * Human TUNE (en-GB, neural, resolved taps). Fully typed words, TSI / VTAP WER and false corrections: off 2.14 /
     * 5.03 %, 0 / 0.13 %; margin 2-4: TSI 2.35-2.38 %, false corrections 0.23-0.26 % (completions of short words that
     * were typed whole); margin 5 and 6: TSI unchanged, VTAP 5.00 %. Partial-word set (prefixes of 2-5 letters,
     * PartialEval), margin 5 against off: space completes into the word being typed for +3.4 (TSI) / +4.1 (VTAP) % of
     * the prefixes at 94 / 97 % precision of the added completions. Conformance mid-word marks: agrees with Gboard on
     * 35 of 66 cases (off: 30).
     */
    val completeOnSpace: Boolean = true,
    val completeMinLength: Int = 2,
    val completeCommonPrior: Float = -11f,
    val completeMargin: Float = 5f,
    val extendMargin: Float = 0.5f,
)

/** TF-01 tap-time key resolution ([TapResolver]). */
data class TapResolveParams(
    val enabled: Boolean = true,
    /**
     * Taps within this distance (key widths) of the hit key's centre always type that key. Human TUNE (keys only,
     * resolved per-tap accuracy TSI / VTAP): 0.15 97.28 / 95.47, 0.2 97.28 / 95.45, 0.25 97.28 / 95.39, 0.3 97.20 /
     * 95.24 %, fewer overrides of right keys as it grows. 0.2: Gboard 18.6 types the key under the finger where we
     * resolved more eagerly (conformance farm-440-04: "Be ho" + a tap 0.18 key widths from the centre of `e`).
     * 0.3: the farm-440 pass on dbe4302 still has Gboard typing the key under the finger and matching the intended letter
     * more often (11 scenarios vs 6; "I do donn" + a tap 0.21 key widths left of the centre of `o` types "o", not "i").
     * Human TUNE words (IME touch model, WER TSI / VTAP): 0.2 2.11 / 4.80 %, 0.3 2.08 / 4.80 %; priorWeight 0.2 or
     * 0.15 instead is worse (2.14 / 4.82, 2.17 / 4.93).
     */
    val centreZone: Float = 0.3f,
    /**
     * Weight of the log letter prior against the spatial log-likelihood. Human TUNE taps (per-tap key accuracy,
     * rectangle -> resolved; overridden = the rectangle was right): 0.15 TSI 94.8 -> 97.2 % (0.15 %), VTAP
     * 90.9 -> 94.4 % (0.26 %); 0.25 97.5 % (0.31 %), 95.2 % (0.52 %); 0.35 97.5 % (0.63 %), 95.6 % (0.80 %);
     * 0.5 96.8 % (1.4 %). 0.25: nearly all the gain with few overrides of deliberate letters.
     */
    val priorWeight: Float = 0.25f,
    /** Candidates: the hit key and the letter keys whose centres lie within this distance (key widths) of its centre. */
    val neighbourRadius: Float = 1.6f,
    /** Share of the language model's next-word predictions in the letter prior, when some start with the prefix. */
    val contextWeight: Float = 0.5f,
    /**
     * Floor on a letter's (candidate-normalised) prior: impossible continuations are penalised, not excluded. After an
     * earlier slip the word so far is often no word's beginning, and a hard floor turned taps into letters the lexicon
     * allows (conformance farm-440, tsi/user13: "I do donn" + a tap 0.2 key widths from the centre of `o` gave "i"; Gboard
     * types "o", as meant). Human TUNE (IME touch model, centreZone 0.3; WER TSI / VTAP, right keys overridden TSI):
     * 1e-4 2.08 / 4.78 % (0.28 %), 1e-3 2.08 / 4.78 (0.27), 1e-2 2.02 / 4.80 (0.22), 3e-2 2.08 / 4.84 (0.22).
     */
    val minPrior: Float = 1e-2f,
    /**
     * Also resolve across the space bar and the delete key, with the same spatial x prior model (Gboard does): a
     * bottom-row letter tap near the bar may end a complete word with a space, a tap on the bar's top edge may
     * type a letter that continues the word ("ba" + c -> "bac"), a tap on the delete key's edge a letter ("l").
     * Only inside a word (never at its start).
     */
    val spaceAndDelete: Boolean = true,
    /** Taps deeper than this (key widths) inside the space bar or the delete key, from their edges facing letter keys, always do that. */
    val functionEdge: Float = 0.35f,
    /** Prior share of "delete" among the candidates of a tap at the delete key's edge (the letters share the rest). */
    val deletePrior: Float = 0.1f,
    /**
     * Moves the space bar's / the delete key's spatial centre line down by this many key widths: their boundary
     * with the letter keys shifts into them (letters win more of the bar's top edge), towards Gboard's boundaries.
     *
     * Calibrated with [wideKeyPriorWeight] against Gboard 18.6 probes on the phone layout (emulator farm, density 440):
     * "ba" + a tap 30 % of a row gap above the bar's centre types "c", "hel" + a tap on the delete key's visual
     * top-left corner types "l" (both as Gboard). Gboard also keeps "hello" + a tap on b's bottom edge as "b"; that
     * needs spaceShift >= 0.5, which drops tap-time recovery of missed spaces on human TSI from 71.5 % to 3 %, so
     * it stays unmatched. Human intent (net right minus wrong space/delete resolutions, keys only): TUNE 127 for
     * the old (0.25, 0, 0) vs 113 for this, TEST 222 vs 243: within noise, pooled 349 vs 356; Gboard parity breaks
     * the tie.
     */
    val spaceShift: Float = 0.2f,
    val deleteShift: Float = 0.3f,
    /** [priorWeight] for decisions that involve the space bar or the delete key. */
    val wideKeyPriorWeight: Float = 0.5f,
    /**
     * Inside a word, a tap on the edge of another key next to letters ([OTHER_KEYS]: "." below `m`, "," below `z`/`x`,
     * shift beside `z` and below `a`) may type the letter it borders, like the delete key's edge ("ho" + a tap on the
     * top edge of "." types "m"; Gboard 18.6 made "home", "deal", "maybe" of such taps on the human VTAP set).
     */
    val otherKeys: Boolean = true,
    /**
     * At a word's start (after a space or at the start of the field), a tap on the edge of the space bar, "." or shift may
     * type the letter it borders too (Gboard 18.6 on the human VTAP set: ".aybe" -> "maybe", shift + "long" -> "along",
     * space + "ext" -> "next"). Only within [wordStartEdge] of the edge facing letters, with [wordStartKeyPrior] for the
     * key itself: a deliberate space or period after a word must stay one.
     *
     * Human TUNE (IME touch model): VTAP WER 4.80 -> 4.66 % (7 words gained, none lost), TSI 2.02 % unchanged, false
     * corrections unchanged, and none of TSI's 2546 deliberate space-bar taps after a word, replayed as a second space,
     * becomes a letter. Edge 0.15 and 0.25 key widths give the same; a key prior of 0.5 gains 2 more words but turns 3
     * of those deliberate spaces into letters.
     */
    val wordStart: Boolean = true,
    val wordStartEdge: Float = 0.15f,
    val wordStartKeyPrior: Float = 0.8f,
    /**
     * Prior share of shift among the candidates of such a tap (as [deletePrior]); "." and "," have the space bar's prior
     * (the word ends here). Human TUNE (IME touch model, otherKeys off -> on): VTAP WER 5.84 -> 5.52 % with 0.1 (0.03
     * 5.61, 0.3 5.63), TSI unchanged.
     */
    val otherPrior: Float = 0.1f,
    /**
     * Inside a word of letters, a tap on "." (or ",") next to the space bar within [punctuationSpaceEdge] key widths of
     * the edge it shares with the bar may be a space that missed the bar (after a letter it borders, if any, lost). The
     * bar against the key, with the wide keys' spatial model and [wideKeyPriorWeight]; priors: the word ends here (1 for
     * "little", ~0 for "litt") and the sentence goes on (bar) or ends there (key: the n-gram's P(sentence end) after the
     * word, P(clause end) for ",", at least [punctuationSpaceKeyPrior]).
     *
     * Gboard 15.1 (pboard emulator, >= 400 dp set): after a whole word, taps up to 0.36
     * key widths into "." type a space, 0.42 sometimes, 0.47 a period, also where the sentence may end ("See you
     * tomorrow"); after "litt", after a space and in an empty field the period stays. Its boundary is what a key prior of
     * 0.01 gives without the sentence-end term. That term is ours: on human TSI (production LM, EvalMain "period key")
     * Gboard's rule (no term, edge 0.45) types 10 of 17 spaces that landed on "." as spaces but turns 62 of 573
     * deliberate sentence-final periods into spaces; with the term, 14 of 17 and 4 of 573 (the 3 kept are typos such as
     * "gome": no whole word). 18.6 adds a punctuation/space-bar triggering model that this approximates.
     */
    val punctuationSpace: Boolean = true,
    /** Taps deeper than this (key widths) into "." / "," from the edge they share with the space bar are always the key. */
    val punctuationSpaceEdge: Float = 0.45f,
    val punctuationSpaceKeyPrior: Float = 0.01f,
) {
    companion object {
        /** Labels of the other keys that take part ([otherKeys]), as in the engine geometry. */
        val OTHER_KEYS = listOf(".", ",", "shift")
    }
}

data class EngineConfig(
    val spatial: SpatialParams = SpatialParams(),
    val tap: TapParams = TapParams(),
    val autocorrect: AutocorrectParams = AutocorrectParams(),
    val swipe: dev.pboard.engine.swipe.SwipeParams = dev.pboard.engine.swipe.SwipeParams(),
    /** Learning of words the user types (CO-07). */
    val learning: LearningParams = LearningParams(),
    /** Never offer potentially offensive words unless typed literally (SU-11). */
    val blockOffensive: Boolean = true,
    /** Learn per-key touch offsets from committed words. */
    val adaptTouchOffsets: Boolean = true,
    val predictions: Int = 3,
    /**
     * A candidate that differs from a better one only in case ("Tour" after "tour") keeps its place on the
     * strip only within this many nats of it: "US" next to "us" stays, a rare proper-noun form does not
     * push a different word off the strip.
     */
    val caseVariantMargin: Float = 2.3f,
    val tapResolve: TapResolveParams = TapResolveParams(),
    /**
     * With a neural LM ([dev.pboard.engine.lm.NeuralRescoringLm]): the tap and gesture searches, completions and
     * live gesture previews use its n-gram model; the neural model re-ranks only the final [neuralRerankTopN]
     * candidates of a word (strip and autocorrect decision) and next-word predictions, its state prefetched after
     * each commit. 0 disables re-ranking (n-gram only).
     */
    val neuralRerankTopN: Int = 16,
    /**
     * Each commit computes the neural LM's state for the next word's context, so predictions and the next word's
     * keystrokes find it cached: on the calling (engine) thread by default, or on a background thread when true.
     * Measured on a big.LITTLE phone, the background step lands on a slow core: commit + predictNext p50 22-25 ms
     * against 10 ms on the engine thread.
     */
    val neuralPrefetchInBackground: Boolean = false,
    /** Use the neural LM everywhere instead (every candidate the searches score); for comparisons. */
    val neuralFullSearch: Boolean = false,
)
