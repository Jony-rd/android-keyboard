package dev.pboard.engine

/*
 * Engine API contract shared by the IME (:app) and the decoding engine.
 * Pure Kotlin/JVM: no Android dependencies, so everything here is unit-testable on the host.
 * Coordinates are in keyboard-local pixels; geometry is re-sent whenever the layout changes.
 */

/**
 * One key's hit box. [label] is the lowercase character(s) it produces (e.g. "a", "'"); non-character keys
 * use a name ("space", "shift", "delete"). [alternates] are the characters reachable by long-pressing this
 * key ("é", "è", "ê" on `e`): the decoder treats them as typed on this key.
 */
data class KeyGeom(
    val label: String,
    val cx: Float,
    val cy: Float,
    val width: Float,
    val height: Float,
    val alternates: List<String> = emptyList(),
)

data class KeyboardGeometry(val keys: List<KeyGeom>, val width: Float, val height: Float) {
    /** Typical key width; used to normalise distances so models are resolution independent. */
    val keyWidth: Float get() = keys.map { it.width }.sorted().let { it[it.size / 2] }
}

/** A raw touch sample. [t] in milliseconds (monotonic). */
data class TouchPoint(val x: Float, val y: Float, val t: Long = 0L)

enum class CandidateKind { TYPED, CORRECTION, COMPLETION, PREDICTION, EMOJI }

/**
 * A suggestion. [text] is committed exactly as given and may contain spaces: a multi-word candidate
 * ("hello world") is offered when a missed space bar ran two words together. [score] is a posterior
 * probability over the list.
 *
 * [replacesBefore], when not empty, is the text just before the current word that committing this candidate also
 * replaces: a word split by an accidental space ("th" + space + "is") is offered whole ("this", replacesBefore =
 * "th "). The host deletes [replacesBefore] (after checking the text before the word ends with it) and commits
 * [text] in its place; if the text does not match, it commits the candidate as an ordinary one.
 */
data class Candidate(val text: String, val score: Float, val kind: CandidateKind, val replacesBefore: String = "")

/**
 * Ranked suggestions for the strip.
 *  - candidates[0] is the best candidate: the strip's centre slot.
 *  - [autoCorrectIndex] is 0 when pressing space/punctuation should commit candidates[0] instead of the
 *    literal typed text (Gboard-style autocorrect), and -1 when space commits the literal. With autocorrect
 *    off (or [Context.autoCorrectAllowed] false) a would-be correction still leads the list at index 0.
 *  - For taps, the literal typed text is always in the list (kind [CandidateKind.TYPED]), at index 0 or 1.
 *  - For swipes, candidates[0] is committed by default and the rest are alternatives for the strip.
 */
data class Suggestions(val candidates: List<Candidate>, val autoCorrectIndex: Int = -1) {
    companion object { val EMPTY = Suggestions(emptyList()) }
}

/**
 * The text before the word as tokens, most recent last: words, plus sentence-ending punctuation ("." "!" "?") as
 * tokens of their own so the language model sees sentence boundaries (commas may be omitted). Pass at least the
 * whole current sentence: a context shorter than the LM's order is read as the start of the field. Empty at the
 * start of the field. [sentenceStart] drives capitalisation and means "the LM context starts here".
 * [learningAllowed] is false in fields where nothing may be learned (passwords, incognito,
 * IME_FLAG_NO_PERSONALIZED_LEARNING, …); [autoCorrectAllowed] is false where corrections must not be applied
 * on space (URL, email or password fields, autocorrect switched off). Suggestions are still returned.
 * [exactKeys] is true in fields where every tap must type exactly the key it hit (URL, e-mail, password,
 * code/number-like fields): no tap-time key resolution ([KeyboardEngine.resolveTap]).
 */
data class Context(
    val words: List<String>,
    val sentenceStart: Boolean = false,
    val learningAllowed: Boolean = true,
    val autoCorrectAllowed: Boolean = true,
    val exactKeys: Boolean = false,
    /**
     * The field asks for capitals (an InputType CAP_* flag): autocorrection may capitalise a word typed in lower case
     * whose lower-case form is no word ("brian" -> "Brian", "friday" -> "Friday"). False in no-caps fields, where
     * Gboard keeps "london".
     */
    val capitalsAllowed: Boolean = true,
)

interface LanguageModel {
    /**
     * Natural-log probability of [word] following [context]. An empty [context] means the start of a
     * sentence. Unknown words get a finite penalty.
     */
    fun logProb(context: List<String>, word: String): Float
    /** Most likely next words (optionally constrained to [prefix]), best first, with log probs. Empty context = sentence start. */
    fun predict(context: List<String>, prefix: String = "", limit: Int = 8): List<Pair<String, Float>>
    fun isKnown(word: String): Boolean
}

/**
 * One vocabulary word for decoding: its canonical spelling (capitals, apostrophes, accents: "I'm",
 * "iPhone", "déjà"), a context-free natural-log prior, an optional dialect-variant key (spellings of one
 * word in different dialects share it: "colour"/"color" -> "color"; the engine never autocorrects one into
 * the other) and whether it is potentially offensive (never suggested unless typed in full).
 */
data class LexiconEntry(
    val word: String,
    val logPrior: Float,
    val variantKey: String? = null,
    val offensive: Boolean = false,
)

/** A language's vocabulary, as the decoders consume it. */
fun interface LexiconSource {
    fun entries(): Sequence<LexiconEntry>
}

/**
 * One active typing language: BCP-47 [tag] ("en-GB"), its language model and vocabulary, and a base
 * [weight] (how much the user types in it). Several may be active at once (multilingual typing).
 */
data class LanguageSpec(val tag: String, val lm: LanguageModel, val lexicon: LexiconSource, val weight: Float = 1f)

/** A personal-dictionary entry: [word] (may be a phrase) and an optional [shortcut] that offers it. */
data class PersonalWord(val word: String, val shortcut: String = "")

interface KeyboardEngine {
    fun setGeometry(geometry: KeyboardGeometry)

    /** Replaces the active languages; candidates from all of them compete in one list. */
    fun setLanguages(languages: List<LanguageSpec>) {}

    /**
     * Suggestions while composing a word from individual taps. [typed] is the literal key sequence (with the
     * shift state applied). When [taps] is empty (or does not match [typed]) the word is re-decoded from its
     * text alone: use this when the cursor moves into, or backspaces into, a committed word (re-correction).
     */
    fun suggestForTaps(context: Context, taps: List<TouchPoint>, typed: String): Suggestions

    /**
     * TF-01: the key a single tap types, decided at touch-up. [rectKey] is the label of the key whose hit
     * rectangle contains [touch]; [typedSoFar] is the current word before this tap (as displayed). Returns
     * [rectKey] or the label of a neighbouring letter key of the current layout that is more likely given the
     * touch position and the letters and words before it ("wh" + a tap on the e/r border types "e"). Taps
     * near a key's centre always stay literal; nothing is resolved when [Context.autoCorrectAllowed] is false
     * or [Context.exactKeys] is true. Synchronous and allocation-free: safe to call on the UI thread while
     * other engine calls run on a worker. Word decoding keeps using the raw touch points.
     *
     * Inside a word ([typedSoFar] not empty) the space bar and the delete key take part too: pass [rectKey]
     * "space" or "delete" for taps on them. The result is then a letter key's label (lower case: apply the shift
     * state), "space" or "delete": "ba" + a tap on the bar's top edge types "c"; "hello" + a tap on the bottom
     * edge of `b` above the bar ends the word ("space"); a tap on the delete key's edge after "hel" types "l".
     */
    fun resolveTap(context: Context, typedSoFar: String, touch: TouchPoint, rectKey: String): String = rectKey

    /** Suggestions for a gesture (swipe) path. A path too short to be a gesture is decoded as a tap. */
    fun suggestForSwipe(context: Context, path: List<TouchPoint>): Suggestions

    /**
     * The live preview while the finger is still moving: may skip work that only the final decode needs (e.g.
     * neural re-ranking). Default: the same as [suggestForSwipe].
     */
    fun suggestForSwipePreview(context: Context, path: List<TouchPoint>): Suggestions = suggestForSwipe(context, path)

    /**
     * Optional: the context changed outside a commit (a field was opened, the cursor moved, a word was re-opened).
     * Prepares what the next taps, glides and predictions in [context] need (the neural LM's state for it, the
     * next word's completion list), so the first of them does not pay for it. Call it on the engine's worker
     * thread right after the change, off the input path; cheap when [context] is already prepared.
     */
    fun prepareContext(context: Context) {}

    /** Next-word predictions after a word was committed (sentence starters when [Context.sentenceStart]). */
    fun predictNext(context: Context): Suggestions

    /** On-device learning hooks. Data never leaves the device; nothing is learned when [Context.learningAllowed] is false. */
    fun onWordCommitted(context: Context, word: String, wasTypedLiterally: Boolean) {}
    fun onAutoCorrectReverted(context: Context, original: String, corrected: String) {}

    /**
     * Optional: a word typed by taps was left in the field as typed, without space or punctuation, when the input
     * session ended or the cursor left it. Gboard 18.6 learns it (offers it from its first letters, keeps it on space
     * the next time). Ignored when [Context.learningAllowed] is false.
     */
    fun onWordAbandoned(context: Context, word: String) {}

    /** The user tapped a strip candidate while composing [typed] (picking the literal keeps and learns it). */
    fun onSuggestionPicked(context: Context, candidate: Candidate, typed: String) {}

    /** The IME removed, as a unit, the word it had just committed from a gesture (one backspace after a glide). */
    fun onGestureWordDeleted(context: Context, word: String) {}

    /**
     * A new gesture began: every [suggestForSwipe] until the next call belongs to it (live previews while the
     * finger moves, then the final decode). Hosts that preview glides should call it so per-gesture state
     * (e.g. demoting a just-deleted glided word) spans the whole gesture rather than its first decode.
     */
    fun onGestureStarted() {}

    /** "Remove suggestion": forget a learned word, or stop offering a dictionary word. */
    fun removeSuggestion(word: String) {}

    /**
     * Optional: name tokens from the user's contacts ("Tolu", "Adebayo"; opt-in in the host, which reads them with the
     * user's permission), replacing any earlier set; empty turns the feature off. They count as known names: kept when
     * typed, offered as completions and corrections with their capitals, and a capitalised word may be corrected into
     * one. A name whose lower-case form is a common word of an active language adds nothing (that language decides).
     * Held in memory only: never part of [exportUserData], never logged, and unaffected by [clearUserData].
     */
    fun setContactNames(names: Collection<String>) {}

    /**
     * The user's personal dictionary (Settings > Dictionary), replacing the previous one. Its words are valid words:
     * never autocorrected away, offered as completions and glide targets. Typing an entry's [PersonalWord.shortcut]
     * offers its word on the strip (never as an autocorrection). The host owns and stores the dictionary; the engine
     * keeps no copy in its user data.
     */
    fun setPersonalDictionary(words: List<PersonalWord>) {}

    /**
     * Personal data learned on device (learned words, touch-offset model, removed suggestions, undone
     * corrections) for the IME to keep in app-private, no-backup storage. [importUserData] accepts what
     * [exportUserData] produced by this or any OLDER version (the format is versioned, see
     * dev.pboard.engine.decode.UserDataCodec); [clearUserData] forgets everything.
     *
     * [importUserData] returns false when it did not understand the data (corrupt, or written by a NEWER version
     * after a downgrade); the engine is then left unchanged and the caller must keep the bytes rather than
     * overwrite them with the next export.
     */
    fun exportUserData(): ByteArray = ByteArray(0)
    fun importUserData(data: ByteArray): Boolean = true
    fun clearUserData() {}
}
