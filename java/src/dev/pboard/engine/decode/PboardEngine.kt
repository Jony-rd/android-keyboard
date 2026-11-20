package dev.pboard.engine.decode

import dev.pboard.engine.Candidate
import dev.pboard.engine.CandidateKind
import dev.pboard.engine.Context
import dev.pboard.engine.KeyboardEngine
import dev.pboard.engine.KeyboardGeometry
import dev.pboard.engine.LanguageModel
import dev.pboard.engine.LanguageSpec
import dev.pboard.engine.PersonalWord
import dev.pboard.engine.Suggestions
import dev.pboard.engine.TouchPoint
import dev.pboard.engine.lm.LmPack
import dev.pboard.engine.lm.NeuralRescoringLm
import dev.pboard.engine.lm.NgramLanguageModel
import dev.pboard.engine.swipe.GestureDecoder
import dev.pboard.engine.swipe.NeuralSwipeDecoder
import dev.pboard.engine.swipe.NeuralSwipeEncoder
import dev.pboard.engine.swipe.SwipeModel
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.ln

/**
 * The typing-intelligence engine: tap decoding with Gboard-style autocorrect, completions, gesture
 * decoding and next-word prediction over one or more [LanguageSlot]s (any [LanguageModel] + [Lexicon]),
 * plus the user's learned words. Keys, proximity and gesture templates all derive from the
 * [KeyboardGeometry]; language specifics come from each lexicon's [TextRules].
 *
 * Candidate list contract (taps and swipes):
 *  - candidates[0] is the best candidate (the strip's centre slot). [Suggestions.autoCorrectIndex] is 0
 *    when space/punctuation should commit it instead of the literal typed text; with autocorrect
 *    disabled the would-be correction still leads the list but autoCorrectIndex is -1;
 *  - for taps the literal typed text is always present (kind TYPED), at index 0 or 1;
 *  - the rest follow by decreasing [Candidate.score], a posterior probability over the list.
 *
 * Re-correction (cursor moved into a committed word, or backspacing into one) is the same call with
 * no touches: `suggestForTaps(context, emptyList(), word)` decodes from the text alone and adds the
 * language's real-word confusions ([TextRules.confusables]).
 *
 * Not thread-safe: call from one thread (the IME's decoding thread).
 */
class PboardEngine(languages: List<LanguageSlot>, val config: EngineConfig = EngineConfig()) : KeyboardEngine {

    constructor(lm: LanguageModel, lexicon: Lexicon, config: EngineConfig = EngineConfig()) :
        this(listOf(LanguageSlot(lexicon.rules.locale.toLanguageTag(), lm, lexicon)), config)

    private inner class Runtime(val slot: LanguageSlot) {
        val tap = TapDecoder(slot.lexicon, config.tap, config.spatial.maxCost)
        /** Decodes the word after a missed space (its own incremental cache). */
        val splitTap = TapDecoder(slot.lexicon, config.tap, config.spatial.maxCost)
        /** Decodes a word split by an accidental space, both parts joined (its own incremental cache). */
        val joinTap = TapDecoder(slot.lexicon, config.tap, config.spatial.maxCost)
        var map: CharKeyMap? = null
        private var gestureDecoder: GestureDecoder? = null
        private var neuralDecoder: NeuralSwipeDecoder? = null
        fun bind(keys: KeySpace) {
            map = CharKeyMap(slot.lexicon, keys, config.tap.foldPenalty).also { tap.setKeyMap(it); splitTap.setKeyMap(it); joinTap.setKeyMap(it) }
            gestureDecoder = null
            neuralDecoder = null
        }
        /** Letter-trigram model of the lexicon (built on first use, or by [warmUp]). */
        var lettersModel: LetterModel? = null
            private set
        val letters: LetterModel get() = lettersModel ?: LetterModel(slot.lexicon).also { lettersModel = it }
        /** Gesture templates are built lazily: typing-only users never pay for them. */
        fun gesture(): GestureDecoder? = gestureDecoder ?: map?.let { GestureDecoder(slot.lexicon, it, config.swipe) }?.also { gestureDecoder = it }
        /** The neural gesture decoder, when enabled, a model is loaded and the layout is QWERTY (else null: use [gesture]). */
        fun neural(): NeuralSwipeDecoder? {
            neuralDecoder?.let { return it }
            val enc = neuralEncoder() ?: return null
            return map?.let { NeuralSwipeDecoder(slot.lexicon, it, enc, config.swipe) { gesture() } }?.also { neuralDecoder = it }
        }
        fun resetNeural() { neuralDecoder = null }
    }

    // ---------------------------------------------------------------- neural gesture model

    private var modelLoaded = false
    private var model: SwipeModel? = null
    private var encoder: NeuralSwipeEncoder? = null

    /**
     * The neural gesture model used when [SwipeParams.neural][dev.pboard.engine.swipe.SwipeParams.neural] is on: by
     * default the one bundled with the engine ([SwipeModel.bundled], loaded on first use or by [warmUp]).
     */
    var swipeModel: SwipeModel?
        get() { if (!modelLoaded) { modelLoaded = true; model = if (config.swipe.neural) SwipeModel.bundled() else null }; return model }
        set(value) { modelLoaded = true; model = value; encoder = null; for (rt in langs) rt.resetNeural(); userRuntime?.resetNeural() }

    private fun neuralEncoder(): NeuralSwipeEncoder? {
        if (!config.swipe.neural) return null
        encoder?.let { return it }
        val m = swipeModel ?: return null
        val ks = keySpace ?: return null
        return NeuralSwipeEncoder.create(m, ks)?.also { encoder = it }
    }

    private var langs: List<Runtime> = emptyList()
    private var mixer = LanguageMixer(floatArrayOf(1f))
    private var keySpace: KeySpace? = null
    private var spatial: SpatialModel? = null
    /** Space bar hit box (left, top, right, bottom), if the layout has one: for missed-space decoding. */
    private var spaceRect: FloatArray? = null
    private val touchOffsets = AdaptiveTouchOffsets()
    private val policy = AutocorrectPolicy(config.autocorrect, config.tap.lmWeight)

    /** Standalone learned-word store, used when no language model brings its own user model. */
    val userVocabulary = UserVocabulary(config.learning)
    private var ngramUser: NgramUserWords? = null
    private var ngramUserOf: NgramLanguageModel? = null

    /**
     * The single store of what the user taught the engine: the production LM's user n-gram model when the
     * active language has one (it learns phrases and new words; the decoders reuse its words), else
     * [userVocabulary].
     */
    val learned: LearnedWords get() {
        val m = langs.firstNotNullOfOrNull { ngramOf(it.slot.lm)?.takeIf { lm -> lm.user != null } } ?: return userVocabulary
        if (ngramUserOf !== m) {
            ngramUserOf = m
            ngramUser = NgramUserWords(m, config.learning) { w -> langs.any { it.slot.lexicon.contains(w) } }
        }
        return ngramUser!!
    }

    private fun ngramOf(lm: LanguageModel): NgramLanguageModel? = lm as? NgramLanguageModel ?: (lm as? NeuralRescoringLm)?.ngram

    private fun isIntended(word: String) = word in personalWords || word in contactActive || learned.count(word) >= config.learning.intendedMinCount

    /** Personal-dictionary words (as written) and its shortcuts (lower case -> words), see [setPersonalDictionary]. */
    private var personalWords: Set<String> = emptySet()
    private var personalShortcuts: Map<String, List<String>> = emptyMap()

    override fun setPersonalDictionary(words: List<PersonalWord>) {
        personalWords = words.mapNotNullTo(HashSet()) { it.word.trim().ifEmpty { null } }
        val shortcuts = HashMap<String, MutableList<String>>()
        for (w in words) {
            val s = w.shortcut.trim()
            val word = w.word.trim()
            if (s.isEmpty() || word.isEmpty() || s == word) continue
            shortcuts.getOrPut(primaryRules.lowercase(s)) { ArrayList() }.let { if (word !in it) it.add(word) }
        }
        personalShortcuts = shortcuts
        userVersion = -1
        memoContext = null
    }

    /** Log prior of a personal-dictionary word: that of a learned word that has just become intended. */
    private val personalLogPrior: Float get() = config.learning.baseLogPrior + ln(config.learning.intendedMinCount.toFloat())

    /** Personal-dictionary words the decoders can reach (single words no language has), as extra learned-word entries. */
    private fun personalEntries(have: Set<String>): List<LexiconEntry> = personalWords
        .filter { w -> w.length in 2..Lexicon.MAX_DEPTH && w !in have && w.all(::isWordChar) && langs.none { it.slot.lexicon.contains(w) } }
        .map { LexiconEntry(it, personalLogPrior) }

    /** The learned-words model with the personal dictionary's words on top (they need no uses to count). */
    private fun withPersonal(lm: LanguageModel, store: LearnedWords): LanguageModel = if (personalWords.isEmpty()) lm else object : LanguageModel {
        override fun logProb(context: List<String>, word: String): Float {
            val lp = store.logPrior(word)
            return if (word in personalWords) (if (lp.isFinite()) maxOf(lp, personalLogPrior) else personalLogPrior) else lm.logProb(context, word)
        }
        override fun predict(context: List<String>, prefix: String, limit: Int) = lm.predict(context, prefix, limit)
        override fun isKnown(word: String): Boolean = word in personalWords || lm.isKnown(word)
    }

    /** [s] with the words whose personal-dictionary shortcut is [typed] added right after the leading candidates. */
    private fun withShortcuts(typed: String, s: Suggestions): Suggestions {
        if (personalShortcuts.isEmpty()) return s
        val words = personalShortcuts[primaryRules.lowercase(typed)] ?: return s
        val add = words.filter { w -> s.candidates.take(SHORTCUT_SLOT + 1).none { it.text == w } }.map { Candidate(it, 0.5f, CandidateKind.COMPLETION) }
        if (add.isEmpty()) return s
        val rest = s.candidates.filter { c -> add.none { it.text == c.text } }
        // After the candidate space commits and the literal: visible on the strip, never what space commits.
        val at = minOf(if (s.autoCorrectIndex >= 0) 2 else 1, rest.size)
        return Suggestions(rest.subList(0, at) + add + rest.subList(at, rest.size), s.autoCorrectIndex)
    }

    private var userRuntime: Runtime? = null
    private var userVersion = -1
    private var userStore: LearnedWords? = null

    /** Contact name tokens as the host gave them ([setContactNames]). In memory only: never exported or logged. */
    private var contactNames: Set<String> = emptySet()
    /** The contact names in use: [contactNames] minus what the active languages already know (see [active]). */
    private var contactActive: Set<String> = emptySet()
    private var contactVersion = 0
    private var contactBuilt = 0

    override fun setContactNames(names: Collection<String>) {
        val set = LinkedHashSet<String>()
        for (n in names) {
            if (set.size >= config.learning.maxContactNames) break
            if (n.length in 2..Lexicon.MAX_DEPTH && n.all(::isWordChar) && n.any { Character.isLetter(it) }) set.add(n)
        }
        if (set == contactNames) return
        contactNames = set
        contactVersion++
    }

    /**
     * The contact names the decoders use: not the ones a language's lexicon has as written ("John"), nor those whose
     * lower-case form is a common word there ("Rose", "Will": the language decides their capital).
     */
    private fun activeContactNames(): Set<String> {
        if (contactNames.isEmpty()) return emptySet()
        val common = config.autocorrect.completeCommonPrior
        return contactNames.filterTo(HashSet()) { w ->
            langs.none { rt ->
                val lex = rt.slot.lexicon
                lex.contains(w) || lex.idOf(rt.slot.rules.lowercase(w)).let { it >= 0 && lex.prior(it) >= common }
            }
        }
    }

    /** The learned-words slot's model with the contact names added (context-free, like the learned words). */
    private inner class UserSlotLm(private val base: LanguageModel) : LanguageModel {
        override fun logProb(context: List<String>, word: String): Float =
            if (word in contactActive) maxOf(config.learning.contactLogPrior, base.logProb(context, word)) else base.logProb(context, word)
        override fun predict(context: List<String>, prefix: String, limit: Int) = base.predict(context, prefix, limit)
        override fun isKnown(word: String): Boolean = word in contactActive || base.isKnown(word)
    }

    /** Corrections the user undid: lower-case original -> lower-case corrections never to auto-apply again. */
    private val rejected = HashMap<String, MutableSet<String>>()

    /** Touches of the last tapped word, so a commit can teach the touch model. */
    private var lastTaps: List<TouchPoint> = emptyList()

    /** Lower-case words the user removed from the strip (CO-08): never offered again unless typed in full. */
    private val suppressed = HashSet<String>()

    init { setLanguageSlots(languages) }

    private val primaryRules: TextRules get() = langs.first().slot.rules

    override fun setLanguages(languages: List<LanguageSpec>) {
        setLanguageSlots(languages.map { spec -> slotFor(spec, config) })
    }

    /** Replaces the active languages with prebuilt lexicons. */
    fun setLanguageSlots(slots: List<LanguageSlot>) {
        require(slots.isNotEmpty()) { "at least one language" }
        langs = slots.map { Runtime(it) }
        mixer = LanguageMixer(FloatArray(slots.size) { slots[it].weight })
        userVersion = -1
        // Rebind to the current layout (setGeometry alone is a no-op for an unchanged geometry).
        keySpace?.let { ks -> keySpace = null; setGeometry(ks.geometry) }
    }

    override fun setGeometry(geometry: KeyboardGeometry) {
        // The IME re-sends geometry whenever the keyboard is laid out; rebuilding would throw away the
        // gesture templates and decoder caches for nothing.
        if (geometry == keySpace?.geometry) return
        val ks = KeySpace(geometry, primaryRules)
        keySpace = ks
        encoder = null
        spaceRect = geometry.keys.firstOrNull { it.label == " " || it.label.equals("space", ignoreCase = true) }
            ?.let { floatArrayOf(it.cx - it.width / 2, it.cy - it.height / 2, it.cx + it.width / 2, it.cy + it.height / 2) }
        spatial = SpatialModel(ks, config.spatial, if (config.adaptTouchOffsets) touchOffsets else null)
        for (rt in langs) rt.bind(ks)
        userRuntime?.bind(ks)
        rebuildTapResolver()
    }

    // ---------------------------------------------------------------- TF-01 tap-time key resolution

    /** Published for [resolveTap] (UI thread); rebuilt here on the worker when the layout, languages or touch offsets change. */
    @Volatile private var tapResolver: TapResolver? = null
    private var resolverOffsetsVersion = -1

    override fun resolveTap(context: Context, typedSoFar: String, touch: TouchPoint, rectKey: String): String {
        val r = tapResolver?.resolve(context, typedSoFar, touch.x, touch.y, rectKey) ?: rectKey
        // Split-word repair: remember where the space that ends this word was touched (the UI thread writes, the
        // engine thread reads it at the commit; plain fields, no allocation).
        if ((r == TapResolver.SPACE || r == " ") && typedSoFar.isNotEmpty()) {
            spaceTouchY = touch.y
            spaceTouchWord = typedSoFar
            spaceTouchSeq++
        }
        return r
    }

    @Volatile private var spaceTouchY = 0f
    @Volatile private var spaceTouchWord: String? = null
    @Volatile private var spaceTouchSeq = 0

    private fun rebuildTapResolver() {
        val ks = keySpace
        val sm = spatial
        if (ks == null || sm == null) { tapResolver = null; return }
        val n = ks.size
        val kw = ks.keyWidth
        val isLetter = BooleanArray(n) { Character.isLetter(ks.labels[it]) }
        val r2 = config.tapResolve.neighbourRadius * kw * config.tapResolve.neighbourRadius * kw
        val rectOf = { label: String -> ks.geometry.keys.firstOrNull { it.label.equals(label, ignoreCase = true) }
            ?.let { floatArrayOf(it.cx - it.width / 2, it.cy - it.height / 2, it.cx + it.width / 2, it.cy + it.height / 2) } }
        val space = spaceRect
        val delete = rectOf(TapResolver.DELETE)
        // A letter key is next to a wide key when its centre is within the radius of the wide key's centre line.
        val nearWide = { k: Int, r: FloatArray? ->
            r != null && run {
                val left = minOf(r[0] + kw / 2, (r[0] + r[2]) / 2); val right = maxOf(r[2] - kw / 2, (r[0] + r[2]) / 2)
                val dx = if (ks.cx[k] < left) left - ks.cx[k] else if (ks.cx[k] > right) ks.cx[k] - right else 0f
                val dy = ks.cy[k] - (r[1] + r[3]) / 2
                dx * dx + dy * dy <= r2
            }
        }
        // Other keys next to letters (".", ",", shift), with the edges that face letter keys and their centre line's
        // shift (the space bar's for keys below the letter rows, the delete key's for keys in them).
        val letterGeoms = ks.geometry.keys.filter { it.label.length == 1 && Character.isLetter(it.label[0]) }
        val letterBottom = letterGeoms.maxOfOrNull { it.cy + it.height / 2 } ?: 0f
        val others = TapResolveParams.OTHER_KEYS.mapNotNull { l -> rectOf(l)?.let { l to it } }
        val otherEdges = others.map { (_, r) ->
            val tol = kw * 0.25f
            booleanArrayOf(
                letterGeoms.any { abs(it.cy + it.height / 2 - r[1]) < tol && it.cx > r[0] && it.cx < r[2] },
                letterGeoms.any { abs(it.cx + it.width / 2 - r[0]) < tol && it.cy > r[1] && it.cy < r[3] },
                letterGeoms.any { abs(it.cx - it.width / 2 - r[2]) < tol && it.cy > r[1] && it.cy < r[3] },
            )
        }
        // Which side of the space bar an other key borders in the bar's row (1: the bar on its left, 2: on its right).
        val otherSpaceSide = IntArray(others.size) { i ->
            val r = others[i].second; val tol = kw * 0.25f
            when {
                space == null || abs((r[1] + r[3]) / 2 - (space[1] + space[3]) / 2) > tol -> 0
                abs(r[0] - space[2]) < tol -> 1
                abs(r[2] - space[0]) < tol -> 2
                else -> 0
            }
        }
        val otherShift = FloatArray(others.size) { i ->
            if (others[i].second[1] >= letterBottom - 1f) config.tapResolve.spaceShift else config.tapResolve.deleteShift
        }
        val neighbours = Array(n + 2 + others.size) { k ->
            val near = ArrayList<Int>()
            near.add(k)
            when {
                k == n -> for (j in 0 until n) { if (isLetter[j] && nearWide(j, space)) near.add(j) }
                k == n + 1 -> for (j in 0 until n) { if (isLetter[j] && nearWide(j, delete)) near.add(j) }
                k >= n + 2 -> for (j in 0 until n) { if (isLetter[j] && nearWide(j, others[k - n - 2].second)) near.add(j) }
                isLetter[k] -> {
                    for (j in 0 until n) {
                        if (j == k || !isLetter[j]) continue
                        val dx = ks.cx[j] - ks.cx[k]; val dy = ks.cy[j] - ks.cy[k]
                        if (dx * dx + dy * dy <= r2) near.add(j)
                    }
                    if (nearWide(k, space)) near.add(n)
                    if (nearWide(k, delete)) near.add(n + 1)
                }
            }
            near.toIntArray()
        }
        val letterKeys = (0 until n).filter { isLetter[it] }
        val offX = if (letterKeys.isEmpty()) 0f else letterKeys.map { sm.idealX(it) - ks.cx[it] }.average().toFloat()
        val offY = if (letterKeys.isEmpty()) 0f else letterKeys.map { sm.idealY(it) - ks.cy[it] }.average().toFloat()
        val rlangs = Array(langs.size) { i ->
            val lex = langs[i].slot.lexicon
            TapResolver.Lang(lex, langs[i].lettersModel, IntArray(n + 2) { if (it < n) lex.symbolOf(ks.labels[it]) else -1 }, langPrior(i))
        }
        val sx = config.spatial.sigmaX * kw; val sy = config.spatial.sigmaY * kw
        val r = TapResolver(
            ks.labels.copyOf(), Array(n) { ks.labels[it].toString() }, Array(n) { primaryRules.uppercase(ks.labels[it].toString()) },
            ks.cx.copyOf(), ks.cy.copyOf(), FloatArray(n) { sm.idealX(it) }, FloatArray(n) { sm.idealY(it) },
            1f / (2f * sx * sx), 1f / (2f * sy * sy), kw, neighbours, rlangs, primaryRules, config.tapResolve,
            space, delete, offX, offY,
            others.map { it.first }.toTypedArray(), others.map { it.second }.toTypedArray(), otherShift, otherEdges.toTypedArray(),
            otherSpaceSide, punctuationProb(),
        )
        r.contextPrior = tapResolver?.contextPrior
        resolverOffsetsVersion = touchOffsets.version
        tapResolver = r
    }

    /**
     * For [TapResolver]: the chance that "." (the sentence ends) or "," (a clause ends) follows the word typed so far, from
     * the first language's n-gram model (thread-safe reads; called on the UI thread). Null without an n-gram model.
     */
    private fun punctuationProb(): ((Context, String, String) -> Float)? {
        val lm = langs.firstNotNullOfOrNull { ngramOf(it.slot.lm) } ?: return null
        return { context, word, label ->
            val ids = lm.sentenceIds(if (context.sentenceStart) listOf(word) else context.words + word)
            val token = if (label == ",") LmPack.COMMA else LmPack.EOS
            exp(lm.ngramLogProb(ids, ids.size, token).toDouble()).toFloat()
        }
    }

    /** Hands the next-word predictions for [context] to the tap resolver (letter priors at the next taps). */
    private fun publishContextPrior(context: Context, ranked: List<Pair<String, Float>>) {
        val r = tapResolver ?: return
        val top = ranked.take(config.tap.contextPredictions)
        r.contextPrior = TapResolver.ContextPrior(
            context.sentenceStart, ArrayList(context.words),
            Array(top.size) { primaryRules.lowercase(top[it].first) }, FloatArray(top.size) { exp(top[it].second.toDouble()).toFloat() },
        )
    }

    /** Language runtimes plus the learned-words runtime (last, when the user has learned words). */
    private fun active(): List<Runtime> {
        val store = learned
        if (userVersion != store.version || userStore !== store || contactBuilt != contactVersion) {
            userVersion = store.version
            userStore = store
            contactBuilt = contactVersion
            contactActive = activeContactNames()
            val learnt = store.entries()
            val entries = (if (personalWords.isEmpty()) learnt else learnt + personalEntries(learnt.mapTo(HashSet()) { it.word })) +
                contactActive.map { LexiconEntry(it, config.learning.contactLogPrior) }
            userRuntime = if (entries.isEmpty()) null else Runtime(LanguageSlot("user", UserSlotLm(withPersonal(store.lm, store)), Lexicon.build(entries, primaryRules)))
                .also { rt -> keySpace?.let { rt.bind(it) } }
            memoContext = null
        }
        val u = userRuntime
        return if (u == null) langs else langs + u
    }

    private fun isUserSlot(lang: Int) = lang >= langs.size
    private fun langPrior(i: Int): Float = if (isUserSlot(i)) 0f else mixer.logPrior(i)

    private fun lmContext(context: Context): List<String> = if (context.sentenceStart) emptyList() else context.words

    private fun isWordChar(c: Char) = langs.any { it.slot.rules.isWordChar(c) }

    /** Per-language LM memo for the current context (reused across the keystrokes of one word). */
    private var memoContext: List<String>? = null
    private var memo: List<ContextLm> = emptyList()

    private fun contextLms(lmCtx: List<String>, rts: List<Runtime>): List<ContextLm> {
        if (memoContext != lmCtx || memo.size != rts.size) {
            memoContext = lmCtx
            memo = rts.map { ContextLm(it.slot.lm, lmCtx, config.tap.contextPredictions) }
        }
        return memo
    }

    // ---------------------------------------------------------------- taps

    /** Optional per-phase timing for profiling (the IME's debug latency log); null in normal use. */
    @Volatile var phaseTimer: ((phase: String, nanos: Long) -> Unit)? = null

    /**
     * Set by a host that decodes on a worker thread: a gesture decode polls it and gives up (no suggestions)
     * once it returns true, i.e. once a newer request has made the result unwanted. A glide's live previews
     * then never hold up the decode of the finished glide. Null: never cancelled.
     */
    @Volatile var cancelled: (() -> Boolean)? = null

    private inline fun <T> phase(name: String, block: () -> T): T {
        val pt = phaseTimer ?: return block()
        val t0 = System.nanoTime()
        return block().also { pt(name, System.nanoTime() - t0) }
    }

    override fun suggestForTaps(context: Context, taps: List<TouchPoint>, typed: String): Suggestions =
        withShortcuts(typed, decodeTaps(context, taps, typed))

    private fun decodeTaps(context: Context, taps: List<TouchPoint>, typed: String): Suggestions {
        if (typed.isEmpty()) return Suggestions.EMPTY
        val literal = Candidate(typed, 1f, CandidateKind.TYPED)
        if (typed.length > Lexicon.MAX_DEPTH || !typed.all(::isWordChar)) return Suggestions(listOf(literal))
        val input = phase("input") { tapInput(taps, typed) }
        val lmCtx = lmContext(context)
        val rts = phase("learned-words") { active() }
        val ctxLms = phase("context") { contextLms(lmCtx, rts).onEach { it.predictions } }
        val hyps = ArrayList<WordHypothesis>()
        var literalCost = Float.POSITIVE_INFINITY
        phase("decode") {
            for ((i, rt) in rts.withIndex()) {
                hyps.addAll(rt.tap.decode(input, ctxLms[i], i, langPrior(i)))
                rt.tap.literalCost(input)?.let { if (it < literalCost) literalCost = it }
            }
        }
        if (literalCost.isInfinite()) literalCost = 0f
        addConfusables(typed, literalCost, ctxLms, hyps)
        if (taps.size == typed.length) addMissedSpaces(taps, typed, input, lmCtx, rts, ctxLms, hyps)
        hyps.sortBy { it.total }
        phase("rerank") { rerank(hyps, lmCtx, config.tap.lmWeight, typed) }
        // Re-opened from text alone (backspaced or moved into): no touches although a keyboard is there.
        val reopened = taps.isEmpty() && keySpace != null
        val literalTotal = literalCost - config.tap.lmWeight * (config.autocorrect.oovLogPriorBase + config.autocorrect.oovLogPriorPerChar * typed.length)
        lastDecode = DecodeSummary(context, typed, taps, hyps, literalTotal)
        val s = phase("assemble") { assembleTapSuggestions(context, lmCtx, typed, literalCost, input, taps, hyps, reopened) }
        val join = phase("join") { joinWithPrevious(context, taps, typed, s) } ?: return s
        val cand = Candidate(join.text, 1f, CandidateKind.CORRECTION, join.replacesBefore)
        val rest = s.candidates.filter { it.text != join.text }
        // Confidently better than the two parts: what space commits; otherwise offered next to the parts.
        return if (join.confident) {
            val ac = config.autocorrect.enabled && context.autoCorrectAllowed && !reopened
            Suggestions(listOf(cand) + rest, if (ac) 0 else -1)
        } else {
            val at = minOf(if (s.autoCorrectIndex >= 0) 2 else 1, rest.size)
            Suggestions(rest.subList(0, at) + cand + rest.subList(at, rest.size), s.autoCorrectIndex)
        }
    }

    // ---------------------------------------------------------------- split-word repair

    /** The latest tap decode (its context, input and hypotheses): what a commit right after it committed. */
    private class DecodeSummary(val context: Context, val typed: String, val taps: List<TouchPoint>, val hyps: List<WordHypothesis>,
                                val literalTotal: Float) {
        /** Total of the best interpretation of the typed letters as [word] (or as the literal, when it is no hypothesis). */
        fun totalOf(word: String, rules: TextRules): Float {
            val w = rules.lowercase(word)
            return hyps.firstOrNull { it.prefix.isEmpty() && !it.isCompletion && rules.lowercase(it.canonical) == w }?.total
                ?: if (rules.lowercase(typed) == w) literalTotal else Float.NaN
        }
    }

    /** A word committed by a space tap: the first part of a possible split word. */
    private class SplitPart(val context: Context, val word: String, val typed: String, val taps: List<TouchPoint>, val total: Float,
                            val spaceDepth: Float)

    private class Join(val text: String, val replacesBefore: String, val confident: Boolean)

    /** The best lexicon prior of [w] (as typed or in lower case) over the languages; NaN when no lexicon has it. */
    private fun lexiconPrior(w: String): Float {
        var best = Float.NaN
        for (rt in langs) {
            val lex = rt.slot.lexicon
            for (form in arrayOf(w, rt.slot.rules.lowercase(w))) {
                val id = lex.idOf(form)
                if (id >= 0 && (best.isNaN() || lex.prior(id) > best)) best = lex.prior(id)
            }
        }
        return best
    }

    private var lastDecode: DecodeSummary? = null
    private var splitPart: SplitPart? = null
    private var spaceSeqSeen = 0

    /** At a commit: remember it as a possible first part when the latest decode was that word and a space ended it. */
    private fun noteCommit(context: Context, word: String) {
        val d = lastDecode
        val seq = spaceTouchSeq
        val spaced = seq != spaceSeqSeen && spaceTouchWord == d?.typed
        spaceSeqSeen = seq
        splitPart = null
        val r = spaceRect ?: return
        val kw = keySpace?.keyWidth ?: return
        if (!config.tap.joinSplitWords || d == null || !spaced || d.context != context || d.taps.size != d.typed.length) return
        val total = d.totalOf(word, primaryRules)
        if (total.isNaN()) return
        splitPart = SplitPart(context, word, d.typed, ArrayList(d.taps), total, (spaceTouchY - r[1]) / kw)
    }

    /**
     * The current word after a space-committed [SplitPart] in the same text: one word over both parts' taps ("th" +
     * "is" -> "this"), when it beats the two parts after the cost that the space touch was accidental.
     */
    private fun joinWithPrevious(context: Context, taps: List<TouchPoint>, typed: String, s: Suggestions): Join? {
        val part = splitPart ?: return null
        // Only a first part committed as typed: a corrected one ("prioritie" -> "priorities") was a finished word.
        if (!part.word.equals(part.typed, ignoreCase = true)) return null
        if (taps.size != typed.length || context.words.lastOrNull() != part.word || context.sentenceStart ||
            context.words.size != part.context.words.size + 1 || context.words.subList(0, context.words.size - 1) != part.context.words) return null
        val input = tapInput(part.taps + taps, part.typed + typed, record = false)
        val partLm = lmContext(part.context)
        // A join uses the second part (not "announce" for "announce" + "a") and edits little.
        val usesBoth = { h: WordHypothesis ->
            h.prefix.isEmpty() && h.edits <= config.tap.joinMaxEdits && h.canonical.length > part.typed.length &&
                h.canonical.length >= part.typed.length + typed.length - config.tap.joinMaxEdits
        }
        var best: WordHypothesis? = null
        for ((i, rt) in langs.withIndex()) {
            val ctx = ContextLm(rt.slot.lm, partLm, 0)
            val h = rt.joinTap.decode(input, ctx, i, langPrior(i), withCompletions = false).firstOrNull(usesBoth) ?: continue
            if (best == null || h.total < best.total) best = h
        }
        val j = best ?: return null
        // As Gboard: two words stay two words ("the re", "in to", "earth quakes"); a part that is no word joins
        // ("hel lo", "som ething", "may e").
        if (lexiconPrior(part.word) >= config.tap.joinCommonPrior && !lexiconPrior(typed).isNaN()) return null
        // The alternative: the two parts as space would commit them now (the second part's correction or literal).
        val now = if (s.autoCorrectIndex >= 0) s.candidates[s.autoCorrectIndex].text else typed
        val second = lastDecode?.totalOf(now, primaryRules)?.takeIf { !it.isNaN() } ?: return null
        val split = part.total + second
        val joined = j.total + config.tap.joinBase + config.tap.joinPerKey * maxOf(part.spaceDepth, 0f)
        if (joined >= split) return null
        val text = display(j, TypedShape.of(part.typed + typed), part.context.sentenceStart)
        return Join(text, part.word + " ", joined + config.tap.joinMargin < split)
    }

    /**
     * [h] is a strict prefix of [typed] ("he" for "hel") and some dropped letter was meant: not a repeat of the
     * letter before it ("andd"), on a key next to it ("andf") or a touch just above the space bar ("hellob").
     * See [AutocorrectParams.keepTypedTail].
     */
    private fun dropsMeantLetters(h: WordHypothesis, typed: String, input: TapInput, taps: List<TouchPoint>): Boolean {
        if (!config.autocorrect.keepTypedTail || h.prefix.isNotEmpty()) return false
        val keep = h.canonical.length
        if (keep == 0 || keep >= typed.length || keep > input.size) return false
        val rules = rulesOf(h)
        if (!rules.lowercase(typed).startsWith(rules.lowercase(h.canonical))) return false
        val ks = keySpace
        val touched = taps.size == input.size
        val bar = spaceRect
        for (j in keep until input.size) {
            val prev = input.typed[j - 1]; val c = input.typed[j]
            if (c == prev) continue
            if (ks != null) {
                val a = ks.keyOf(prev); val b = ks.keyOf(c)
                if (touched) {
                    // A touch on the lower edge of a bottom-row key, just above the bar, may have been meant for it.
                    if (bar != null && missedSpaceCost(taps[j]).isFinite() && bar[1] - taps[j].y <= NEAR_BAR * ks.keyWidth) continue
                    // A sloppy touch, away from its key's centre, may be a stray one; a clean one was meant, even on a
                    // key next to the last one (Gboard keeps "wer", "ab": no "we", "a").
                    if (b >= 0 && hypot(taps[j].x - ks.cx[b], taps[j].y - ks.cy[b]) > config.autocorrect.keepTypedTailCleanRadius * ks.keyWidth) continue
                } else if (a >= 0 && b >= 0 && hypot(ks.cx[a] - ks.cx[b], ks.cy[a] - ks.cy[b]) <= ADJACENT_KEYS * ks.keyWidth) {
                    continue // from text alone: a key next to the last one ("andf")
                }
            }
            return true
        }
        return false
    }

    /**
     * Builds what would otherwise be built on first use (gesture templates for every language, the decoders'
     * hot paths, the LM's lookup tables) so the first keystrokes and the first swipe are not slow. Call once
     * on the engine thread after [setGeometry].
     */
    private var warmedFor: Pair<KeyboardGeometry, List<LanguageSlot>>? = null

    fun warmUp() {
        val ks = keySpace ?: return
        val key = ks.geometry to langs.map { it.slot }
        if (warmedFor == key) return
        warmedFor = key
        for (rt in langs) { rt.neural() ?: rt.gesture(); rt.letters }
        rebuildTapResolver()
        val ctx = Context(listOf("I", "think"), false)
        val keys = "the".mapNotNull { c -> ks.keyOf(c).takeIf { it >= 0 } }
        if (keys.size == 3) {
            val taps = keys.map { TouchPoint(ks.cx[it], ks.cy[it]) }
            repeat(3) { for (k in 1..3) suggestForTaps(ctx, taps.subList(0, k), "the".substring(0, k)) }
            suggestForSwipe(ctx, taps)
        }
        predictNext(ctx)
        lastTaps = emptyList()
    }

    /**
     * Cost (nats) that a touch which hit a letter was really meant for the space bar just below it: a
     * prior plus the vertical distance to the bar's top edge (6.9 % of real word-ending taps land on the
     * bottom letter row). Infinite when the touch is not just above the bar.
     */
    private fun missedSpaceCost(p: TouchPoint): Float {
        val r = spaceRect ?: return Float.POSITIVE_INFINITY
        val kw = keySpace?.keyWidth ?: return Float.POSITIVE_INFINITY
        if (p.x < r[0] - 0.3f * kw || p.x > r[2] + 0.3f * kw) return Float.POSITIVE_INFINITY
        val dy = (r[1] - p.y) / kw
        if (dy > config.tap.missedSpaceMaxDistance) return Float.POSITIVE_INFINITY
        val sy = config.spatial.sigmaY
        return config.tap.missedSpacePenalty + if (dy > 0f) dy * dy / (2f * sy * sy) else 0f
    }

    /**
     * Missed space bar (TF/space-miss): the last tap may have been meant as the space ending the word
     * ("hellob" -> "hello"), and an earlier tap may have split two words that ran together
     * ("hellobworld" -> "hello world"). Adds such candidates, scored by both words' channel and LM costs.
     */
    private fun addMissedSpaces(taps: List<TouchPoint>, typed: String, input: TapInput, lmCtx: List<String>,
                                rts: List<Runtime>, ctxLms: List<ContextLm>, hyps: MutableList<WordHypothesis>) {
        val n = taps.size
        if (n < 2 || spaceRect == null) return
        val costs = FloatArray(n) { i -> if (i == 0 || !Character.isLetter(typed[i])) Float.POSITIVE_INFINITY else missedSpaceCost(taps[i]) }
        // Trailing: the last tap ended the word.
        val ct = costs[n - 1]
        if (ct.isFinite()) phase("missed-space-end") { for ((i, rt) in rts.withIndex()) {
            if (isUserSlot(i)) continue
            for (h in rt.tap.decode(input, ctxLms[i], i, langPrior(i), upTo = n - 1, withCompletions = false).take(4)) {
                hyps.add(WordHypothesis(h.lang, h.wordId, h.canonical, h.channelCost + ct, h.lmLogProb, h.total + ct, false, h.edits + 1))
            }
        } }
        // Interior: two words run together; try the most space-like taps — only when the input does not
        // already decode well as one word (the split decodes are the costliest part of a keystroke).
        val bestSingle = hyps.filter { !it.isCompletion && it.prefix.isEmpty() }.minOfOrNull { it.channelCost } ?: Float.POSITIVE_INFINITY
        if (bestSingle < config.tap.missedSpaceSplitMinCost * n / 4f) return
        val splits = (1..n - 2).filter { costs[it] < config.tap.missedSpaceMaxSplitCost }.sortedBy { costs[it] }.take(1)
        if (splits.isNotEmpty()) phase("missed-space-split") { for (j in splits) for ((i, rt) in rts.withIndex()) {
            if (isUserSlot(i)) continue
            val w1 = rt.tap.decode(input, ctxLms[i], i, langPrior(i), upTo = j, withCompletions = false).firstOrNull() ?: continue
            // No context-driven completions for the second word: a fresh prediction list per keystroke is too costly.
            val ctx2 = ContextLm(rt.slot.lm, lmCtx + w1.canonical, 0)
            for (w2 in rt.splitTap.decode(input.slice(j + 1, n), ctx2, i, langPrior(i)).take(3)) {
                hyps.add(WordHypothesis(i, w2.wordId, w2.canonical, w1.channelCost + w2.channelCost + costs[j], w1.lmLogProb + w2.lmLogProb,
                    w1.total + w2.total + costs[j], w2.isCompletion, w1.edits + w2.edits + 1, prefix = w1.canonical + " "))
            }
        } }
    }

    /** Real-word confusions of a valid literal ("their" -> "there", "they're"), as alternatives. */
    private fun addConfusables(typed: String, literalCost: Float, ctxLms: List<ContextLm>, hyps: MutableList<WordHypothesis>) {
        for ((i, rt) in langs.withIndex()) {
            val conf = rt.slot.rules.confusables(typed)
            if (conf.isEmpty()) continue
            val lex = rt.slot.lexicon
            for (w in conf) {
                val id = lex.idOf(w)
                if (id < 0 || hyps.any { it.lang == i && it.wordId == id }) continue
                val lp = ctxLms[i].logProb(w) + langPrior(i)
                val cost = literalCost + config.tap.confusionCost
                hyps.add(WordHypothesis(i, id, w, cost, lp, cost - config.tap.lmWeight * lp, false, 1))
            }
        }
    }

    private fun tapInput(taps: List<TouchPoint>, typed: String, record: Boolean = true): TapInput {
        val n = typed.length
        val sm = spatial
        val ks = keySpace
        val keyCosts = ArrayList<FloatArray>(n)
        val lower = CharArray(n)
        val explicit = BooleanArray(n)
        val xs = FloatArray(n); val ys = FloatArray(n)
        val points = ArrayList<TouchPoint>(n)
        for (i in 0 until n) {
            val c = primaryRules.lowercase(typed[i].toString())[0]
            lower[i] = c
            val arr = FloatArray(ks?.size ?: 0)
            val key = ks?.keyOf(c) ?: -1
            val p: TouchPoint? = if (taps.size == n) taps[i] else null
            when {
                sm != null && p != null -> {
                    sm.costs(p.x, p.y, arr)
                    xs[i] = p.x; ys[i] = p.y
                    // A character whose key was not (nearly) touched was produced deliberately: long-press
                    // alternate, symbol layer, or a different keyboard page.
                    explicit[i] = key < 0 || arr[key] > EXPLICIT_COST
                    points.add(p)
                }
                sm != null && key >= 0 -> {
                    // No usable touch data (a word re-opened for editing): a perfect tap on the key.
                    val x = sm.idealX(key); val y = sm.idealY(key)
                    sm.costs(x, y, arr)
                    xs[i] = x; ys[i] = y
                    points.add(TouchPoint(x, y))
                }
                else -> {
                    explicit[i] = true
                    xs[i] = -1f - c.code; ys[i] = -1f - i
                }
            }
            keyCosts.add(arr)
        }
        if (record) lastTaps = points
        val space = if (config.tap.strayTouchExtra.isFinite() && taps.size == n && spaceRect != null)
            FloatArray(n) { i -> if (Character.isLetter(typed[i])) missedSpaceCost(taps[i]) else Float.POSITIVE_INFINITY } else null
        return TapInput(keyCosts, lower, explicit, xs, ys, space)
    }

    private fun runtimeOf(h: WordHypothesis): Runtime = if (isUserSlot(h.lang)) userRuntime!! else langs[h.lang]
    private fun rulesOf(h: WordHypothesis): TextRules = runtimeOf(h).slot.rules
    private fun lexiconOf(h: WordHypothesis): Lexicon = runtimeOf(h).slot.lexicon

    /** Offensive words are offered only when they are exactly what was typed (SU-11). */
    private fun blocked(h: WordHypothesis, typed: String?): Boolean {
        if (typed != null && h.prefix.isEmpty() && h.canonical.equals(typed, ignoreCase = true)) return false
        if (suppressed.isNotEmpty() && primaryRules.lowercase(h.canonical) in suppressed) return true
        return config.blockOffensive && lexiconOf(h).isOffensive(h.wordId)
    }

    private class Entry(val text: String, val total: Float, val kind: CandidateKind, val demoted: Boolean = false)

    /** Best cost per case-folded text on the strip so far, to drop case-only variants that are much less likely. */
    private inner class CaseFold {
        private val best = HashMap<String, Float>()
        fun note(e: Entry) { best.merge(primaryRules.lowercase(e.text), e.total, ::minOf) }
        fun weakVariant(text: String, total: Float): Boolean {
            val b = best[primaryRules.lowercase(text)] ?: return false
            return total > b + config.caseVariantMargin
        }
    }

    /** Candidate text with the user's shift state / sentence start applied (to the first word only). */
    private fun display(h: WordHypothesis, shape: TypedShape, sentenceStart: Boolean): String {
        val r = rulesOf(h)
        if (h.prefix.isEmpty()) return Casing.apply(h.canonical, shape, sentenceStart, r)
        val second = if (shape == TypedShape.ALL_UPPER) r.uppercase(h.canonical) else h.canonical
        return Casing.apply(h.prefix.trimEnd(), shape, sentenceStart, r) + " " + second
    }

    /** Whether [typed] is a valid word as typed in any active language (lexicon, LM, or an intended learned word). */
    private fun isValidLiteral(typed: String, shape: TypedShape): Boolean {
        if (isIntended(typed)) return true
        for (rt in langs) {
            val lex = rt.slot.lexicon
            if (lex.contains(typed) || rt.slot.lm.isKnown(typed)) return true
            if (shape == TypedShape.FIRST_UPPER) {
                val lower = rt.slot.rules.lowercase(typed.substring(0, 1)) + typed.substring(1)
                if (lex.contains(lower) || isIntended(lower)) return true
            }
        }
        return false
    }

    /**
     * Among [h] and its other-dialect spellings, the one the active language model prefers (its
     * dialect's form: "organisation" under en-GB, "organization" under en-US), or [h] itself.
     */
    private fun preferredVariant(h: WordHypothesis, lmCtx: List<String>): WordHypothesis {
        if (isUserSlot(h.lang) || h.prefix.isNotEmpty()) return h
        val lex = lexiconOf(h)
        val vars = lex.variantsOf(h.wordId)
        if (vars.isEmpty()) return h
        val lm = langs[h.lang].slot.lm
        val prior = langPrior(h.lang)
        var best = h
        for (v in vars) {
            val w = lex.word(v)
            val lp = lm.logProb(lmCtx, w) + prior
            if (lp > best.lmLogProb + 1e-3f) best = WordHypothesis(h.lang, v, w, h.channelCost, lp, h.channelCost - config.tap.lmWeight * lp, h.isCompletion, h.edits)
        }
        return best
    }

    /**
     * Whether [h] is another dialect's spelling of [typed], possibly also differing in apostrophes/case
     * ("neighbour's" for "neighbors"): same skeleton after dialect normalisation, different keys.
     */
    private fun isDialectVariantOf(h: WordHypothesis, typed: String): Boolean {
        if (h.prefix.isNotEmpty()) return false
        val r = rulesOf(h)
        val lex = lexiconOf(h)
        if (lex.skeletonOf(h.wordId) == r.skeleton(typed)) return false
        val lit = lex.idOf(typed).takeIf { it >= 0 } ?: lex.idOf(r.lowercase(typed))
        if (lit >= 0 && h.wordId in lex.variantsOf(lit)) return true
        return r.skeleton(r.variantKey(h.canonical)) == r.skeleton(r.variantKey(typed))
    }

    private fun assembleTapSuggestions(context: Context, lmCtx: List<String>, typed: String, literalCost: Float, input: TapInput, taps: List<TouchPoint>,
                                       all: List<WordHypothesis>, reopened: Boolean): Suggestions {
        val n = input.size
        val hyps = all.filter { !blocked(it, typed) }
        // "hel": "he" would drop a meant letter; it is neither an autocorrection nor ranked above "hello".
        val dropsMeant = hyps.filterTo(HashSet()) { dropsMeantLetters(it, typed, input, taps) }
        val shape = TypedShape.of(typed)
        val sentenceStart = context.sentenceStart
        // A word typed in lower case at a sentence start (no auto-capitals: a no-caps field, or shift turned off) keeps
        // its lower case in the candidates, as Gboard: "hel" -> "hello", not "Hello".
        val capStart = sentenceStart && shape != TypedShape.LOWER
        val cased = HashMap<WordHypothesis, String>(hyps.size * 2)
        for (h in hyps) cased[h] = display(h, shape, capStart)
        val typedSkeletons = (langs.map { it.slot.rules } + primaryRules).map { it.skeleton(typed) }
        val sameKeys = { h: WordHypothesis -> h.prefix.isEmpty() && lexiconOf(h).skeletonOf(h.wordId) == typedSkeletons[minOf(h.lang, langs.size)] }
        // Learned words count as "the literal is a word" only once they are intended (seen often enough).
        val countsAsWord = { h: WordHypothesis -> !isUserSlot(h.lang) || isIntended(h.canonical) }

        val literalForms = hyps.filter { !it.isCompletion && sameKeys(it) && countsAsWord(it) }
        val exact = literalForms.firstOrNull { cased[it] == typed || it.canonical == typed }
        // A literal the LM knows but the lexicon lacks still counts as a valid word.
        val validLiteral = exact != null || (literalForms.isEmpty() && isValidLiteral(typed, shape))
        val status = when {
            validLiteral -> AutocorrectPolicy.LiteralStatus.EXACT
            literalForms.isNotEmpty() -> AutocorrectPolicy.LiteralStatus.VARIANT
            else -> AutocorrectPolicy.LiteralStatus.UNKNOWN
        }
        val literalOverride = if (exact == null && validLiteral) {
            langs.withIndex().maxOf { (i, rt) -> rt.slot.lm.logProb(lmCtx, typed) + langPrior(i) }
                .coerceAtLeast(learned.logPrior(typed)).coerceAtLeast(if (typed in contactActive) config.learning.contactLogPrior else Float.NEGATIVE_INFINITY).let { literalCost - config.tap.lmWeight * it }
        } else null
        val rejectedForTyped = rejected[primaryRules.lowercase(typed)]
        // Learned words are offered from their first use but become autocorrect targets only once intended.
        val ranked = hyps.filter { !it.isCompletion && countsAsWord(it) && it !in dropsMeant &&
            (rejectedForTyped == null || primaryRules.lowercase(cased[it]!!) !in rejectedForTyped) }
        // Context support for a same-keys variant: its LM advantage over the literal exceeds the prior advantage.
        val contextSupports = { h: WordHypothesis ->
            val lit = exact
            if (lit == null || lit.lang != h.lang || isUserSlot(h.lang)) true
            else {
                val lex = lexiconOf(h)
                (h.lmLogProb - lit.lmLogProb) - (lex.prior(h.wordId) - lex.prior(lit.wordId)) > 0.5f
            }
        }
        // A lower-case input, or one capitalised only by the sentence start (auto-caps carries no signal).
        val lowerInput = shape == TypedShape.LOWER || (sentenceStart && shape == TypedShape.FIRST_UPPER)
        // Weak case variants of a short lower-case input: any capitalised form of two letters ("AR", "Mo"), an
        // all-capitals acronym of three ("JSX"); three-letter names ("Tom", "Jim") are real evidence.
        val weakCaseVariant = { h: WordHypothesis ->
            lowerInput && primaryRules.lowercase(h.canonical) == primaryRules.lowercase(typed) &&
                (n <= 2 || h.canonical.none { Character.isLowerCase(it) })
        }
        val decision = policy.decide(ranked, status, exact, literalCost, n, sameKeys, literalOverride, contextSupports, weakCaseVariant)
        // A "correction" to exactly what was typed (e.g. a learned word) is no correction.
        var target = decision.target?.takeIf { cased[it] != typed }
        // Completion on space (Gboard 18.6: "The first ti" -> "time", "goi" -> "going", bold while typed): a part of a
        // word that is no common word, whose completion the context makes overwhelmingly likely.
        if (target == null && !reopened) target = completionTarget(hyps, typed, typedSkeletons, n, rejectedForTyped,
            exact?.total ?: decision.literalTotal)?.takeIf { cased[it] != typed && countsAsWord(it) }
        // A short lower-case input is not turned into an acronym ("ar" -> "AR", "js" -> "JS").
        target?.let { t ->
            if (n <= 3 && t.canonical.length >= 2 && t.canonical.all { !Character.isLetter(it) || Character.isUpperCase(it) } &&
                lowerInput) target = null
        }
        // A lower-case word mid-sentence is corrected to the lower-case form when that is about as likely
        // ("centte" -> "centre", not "Centre"); words with no common lower-case form keep their capitals.
        target?.let { t ->
            if (shape == TypedShape.LOWER && !sentenceStart && !sameKeys(t) && t.prefix.isEmpty()) {
                val low = rulesOf(t).lowercase(t.canonical)
                if (low != t.canonical) hyps.firstOrNull { !it.isCompletion && it.prefix.isEmpty() && it.lang == t.lang && it.canonical == low }
                    ?.takeIf { it.total - t.total < config.caseVariantMargin }?.let { target = it }
            }
        }
        // CO-09 at a sentence start: a precisely typed, word-like capitalised unknown word is not turned into a
        // proper noun ("Mikey" -> "Mike", "Tomy" -> "Tony"); misspelt common words still are ("Teh" -> "The").
        val ac = config.autocorrect
        target?.let { t ->
            if (ac.protectNamesAtSentenceStart && status == AutocorrectPolicy.LiteralStatus.UNKNOWN && sentenceStart &&
                shape == TypedShape.FIRST_UPPER && n >= 3 && n <= ac.nameMaxLength && isProperNoun(t) && !sameKeys(t) &&
                literalCost <= ac.nameMaxCostPerKey * n && langs.any { it.letters.weakestTransition(typed) >= ac.nameMinLetterLogProb }
            ) target = null
        }
        // TP-02: a correction that only adds letters ("hel" -> "he'll") is a completion in disguise: it replaces the
        // typed word only when the language model clearly prefers it to every other word starting with those letters
        // ("wha" -> "what"), never while "help" is about as likely (Gboard: `hel | hello | help`, nothing bold).
        // Longer forms of the target itself ("wan" -> "want" vs "wanted", "tak" -> "take" vs "takes") do not compete.
        val skeletonOf = { h: WordHypothesis -> lexiconOf(h).skeletonOf(h.wordId) }
        val extendsTyped = { h: WordHypothesis ->
            h.prefix.isEmpty() && skeletonOf(h).let { s ->
                val t = typedSkeletons[minOf(h.lang, langs.size)]
                s.length > t.length && s.startsWith(t)
            }
        }
        target?.let { t ->
            if (ac.extendOnlyWhenLikeliest && extendsTyped(t)) {
                val word = primaryRules.lowercase(t.canonical)
                val ts = skeletonOf(t)
                if (hyps.any { it !== t && it.lmLogProb > t.lmLogProb - ac.extendMargin && extendsTyped(it) &&
                        primaryRules.lowercase(it.canonical) != word && !skeletonOf(it).startsWith(ts) }) target = null
            }
        }
        // As Gboard 18.6: a word typed in lower case gets a capital from autocorrection only when its lower-case form is
        // no word and the field asks for capitals ("brian" -> "Brian", "friday" -> "Friday"; "dome", "beau" stay, and
        // "london" in a no-caps field); the single letter "i" -> "I" and forms with inner capitals ("iPhone") always.
        target?.let { t ->
            if (!ac.capitaliseLowercase && shape == TypedShape.LOWER && typed.length > 1 && t.canonical.drop(1).none { Character.isUpperCase(it) } &&
                rulesOf(t).lowercase(t.canonical) == rulesOf(t).lowercase(typed) &&
                (validLiteral || !context.capitalsAllowed)) target = null
        }
        // A valid plural is not turned into a possessive ("his neighbors" stays): offered, never forced.
        target?.let { t -> if (validLiteral && sameKeys(t) && rulesOf(t).isPossessiveOf(t.canonical, typed)) target = null }
        // CO-09: a word the user capitalised mid-sentence that no language knows is most likely a name: never
        // replace it with a different word (capital/apostrophe fixes of a known word are still made).
        if (target != null && status == AutocorrectPolicy.LiteralStatus.UNKNOWN && config.autocorrect.protectCapitalisedUnknown &&
            shape == TypedShape.FIRST_UPPER && !sentenceStart) target = null
        // By default a word capitalised mid-sentence (unknown, or one only the language model knows, like "Tolu") may only
        // become another name or a case/apostrophe form of itself, not an ordinary word ("I met Tolu" -> "You").
        target?.let { t ->
            if (ac.capitalisedOnlyToNames && shape == TypedShape.FIRST_UPPER && !sentenceStart && !sameKeys(t) && !isProperNoun(t)) target = null
        }
        if (target != null) {
            // Dialects: never turn a valid spelling into its other-dialect form ("colour" <-> "color"), and
            // resolve corrections to the spelling of the active dialect ("organisaton" -> "organisation" in en-GB).
            if (validLiteral && isDialectVariantOf(target, typed)) target = null
            else {
                val pref = preferredVariant(target, lmCtx)
                if (pref !== target && !blocked(pref, typed)) {
                    cased[pref] = display(pref, shape, capStart)
                    target = if (validLiteral && isDialectVariantOf(pref, typed)) null else pref
                }
            }
        }

        val entries = ArrayList<Entry>()
        val seen = HashSet<String>()
        val folded = CaseFold()
        fun add(e: Entry) { if (seen.add(e.text)) { entries.add(e); folded.note(e) } }
        if (target != null) add(Entry(cased[target]!!, target.total, CandidateKind.CORRECTION))
        add(Entry(typed, exact?.total ?: decision.literalTotal, CandidateKind.TYPED))
        for (h in hyps) {
            if (entries.size >= config.tap.maxCandidates) break
            val text = cased[h]!!
            val kind = when {
                text == typed -> CandidateKind.TYPED
                h.isCompletion -> CandidateKind.COMPLETION
                else -> CandidateKind.CORRECTION
            }
            if (kind != CandidateKind.TYPED && folded.weakVariant(text, h.total)) continue
            add(Entry(text, h.total, kind, h in dropsMeant))
        }
        val rest = entries.subList(if (target != null) 2 else 1, entries.size)
        rest.sortBy { it.total }
        // Shorter words that drop meant letters go after the words that complete what was typed ("hel | hello | he").
        val lowerTyped = primaryRules.lowercase(typed)
        val lastCompletion = rest.indexOfLast { it.text.length > typed.length && primaryRules.lowercase(it.text).startsWith(lowerTyped) }
        if (lastCompletion >= 0) {
            val moved = rest.subList(0, lastCompletion).filter { it.demoted }
            if (moved.isNotEmpty()) {
                val reordered = rest.filter { !it.demoted || it !in moved }.toMutableList()
                reordered.addAll(reordered.indexOfLast { it.text.length > typed.length && primaryRules.lowercase(it.text).startsWith(lowerTyped) } + 1, moved)
                rest.clear(); rest.addAll(reordered)
            }
        }
        val autoCorrect = target != null && config.autocorrect.enabled && context.autoCorrectAllowed &&
            (!reopened || config.autocorrect.reopenedWords)
        return Suggestions(toCandidates(entries), if (autoCorrect) 0 else -1)
    }

    /**
     * Completion on space ([AutocorrectParams.completeOnSpace]): the best completion that extends the typed letters
     * exactly, when the typed letters are no common word and it beats every other reading (another word, a
     * correction, the literal) by [AutocorrectParams.completeMargin] nats.
     */
    private fun completionTarget(hyps: List<WordHypothesis>, typed: String, typedSkeletons: List<String>, n: Int,
                                 rejected: Set<String>?, literalTotal: Float): WordHypothesis? {
        val ac = config.autocorrect
        if (!ac.completeOnSpace || n < ac.completeMinLength) return null
        val lp = lexiconPrior(typed)
        if (!lp.isNaN() && lp >= ac.completeCommonPrior) return null
        val extends = { h: WordHypothesis ->
            h.prefix.isEmpty() && h.isCompletion && h.edits == 0 && !isUserSlot(h.lang) && lexiconOf(h).skeletonOf(h.wordId).let { s ->
                val t = typedSkeletons[minOf(h.lang, langs.size)]
                s.length > t.length && s.startsWith(t)
            }
        }
        var best: WordHypothesis? = null
        for (h in hyps) if (extends(h) && (best == null || h.total < best.total)) best = h
        val b = best ?: return null
        val word = primaryRules.lowercase(b.canonical)
        if (rejected != null && word in rejected) return null
        var other = literalTotal
        for (h in hyps) if (h.prefix.isEmpty() && primaryRules.lowercase(h.canonical) != word && h.total < other) other = h.total
        return b.takeIf { it.total + ac.completeMargin <= other }
    }

    /** A capitalised word that is mainly a name: no lower-case form in its lexicon, or a rarer one ("Mike", not "Abandon"). */
    private fun isProperNoun(h: WordHypothesis): Boolean {
        if (h.prefix.isNotEmpty() || h.canonical.isEmpty() || !Character.isUpperCase(h.canonical[0])) return false
        val lex = lexiconOf(h)
        val low = lex.idOf(rulesOf(h).lowercase(h.canonical))
        return low < 0 || lex.prior(low) < lex.prior(h.wordId)
    }

    /**
     * Neural re-ranking of the final candidates: the [EngineConfig.neuralRerankTopN] best single-word hypotheses
     * of slots with a rescorer (plus any hypothesis spelling [typed], so the autocorrect decision compares like
     * with like) get the rescorer's LM score in place of the n-gram's. Cheap after [prefetchNext]: one cached LSTM
     * output per context, a lookup per word.
     */
    private fun rerank(hyps: MutableList<WordHypothesis>, lmCtx: List<String>, lmWeight: Float, typed: String?) {
        val n = config.neuralRerankTopN
        if (n <= 0 || langs.none { it.slot.rescorer != null }) return
        val lowerTyped = typed?.let { primaryRules.lowercase(it) }
        var done = 0
        var changed = false
        for (i in hyps.indices) {
            val h = hyps[i]
            if (h.prefix.isNotEmpty() || isUserSlot(h.lang)) continue
            val literal = lowerTyped != null && primaryRules.lowercase(h.canonical) == lowerTyped
            if (done >= n && !literal) continue
            val rescorer = langs[h.lang].slot.rescorer ?: continue
            val lp = rerankLogProb(h.lang, rescorer, lmCtx, h.canonical) + langPrior(h.lang)
            hyps[i] = WordHypothesis(h.lang, h.wordId, h.canonical, h.channelCost, lp, h.total - lmWeight * (lp - h.lmLogProb), h.isCompletion, h.edits, h.prefix)
            if (!literal) done++
            changed = true
        }
        if (changed) hyps.sortBy { it.total }
    }

    /**
     * Rescorer scores for the current context, reused across the keystrokes of one word (like [ContextLm]); dropped
     * on every commit (the user model may have learned) and when the learned words change.
     */
    private var rerankContext: List<String>? = null
    private var rerankVersion = -1
    private var rerankMemo: Array<HashMap<String, Float>> = emptyArray()

    private fun rerankLogProb(lang: Int, rescorer: NeuralRescoringLm, lmCtx: List<String>, word: String): Float {
        val version = learned.version
        if (rerankContext != lmCtx || rerankVersion != version || rerankMemo.size != langs.size) {
            rerankContext = lmCtx
            rerankVersion = version
            rerankMemo = Array(langs.size) { HashMap() }
        }
        return rerankMemo[lang].getOrPut(word) { rescorer.logProb(lmCtx, word) }
    }

    /**
     * After a commit: the neural LM's step for the next word's context (one LSTM step from the cached state), so
     * the predictions that follow and the next word's keystrokes find it cached. See
     * [EngineConfig.neuralPrefetchInBackground].
     */
    private fun prefetchNext(context: Context, word: String) {
        if (config.neuralRerankTopN <= 0) return
        val next = lmContext(context) + word
        for (rt in langs) {
            val rescorer = rt.slot.rescorer ?: continue
            if (config.neuralPrefetchInBackground) rescorer.prefetch(next) else rescorer.output(next)
        }
    }

    private fun toCandidates(entries: List<Entry>): List<Candidate> {
        if (entries.isEmpty()) return emptyList()
        val best = entries.minOf { it.total }
        var z = 0.0
        for (e in entries) z += exp((best - e.total).toDouble())
        return entries.map { Candidate(it.text, (exp((best - it.total).toDouble()) / z).toFloat(), it.kind) }
    }

    // ---------------------------------------------------------------- swipe

    override fun suggestForSwipe(context: Context, path: List<TouchPoint>): Suggestions = swipe(context, path, final = true)

    override fun suggestForSwipePreview(context: Context, path: List<TouchPoint>): Suggestions = swipe(context, path, final = false)

    private fun swipe(context: Context, path: List<TouchPoint>, final: Boolean): Suggestions {
        val sm = spatial ?: return Suggestions.EMPTY
        val ks = keySpace ?: return Suggestions.EMPTY
        if (path.isEmpty()) return Suggestions.EMPTY
        val lmCtx = lmContext(context)
        val hyps = ArrayList<WordHypothesis>()
        var gesture = false
        var neural = false
        for ((i, rt) in active().withIndex()) {
            val cancel = cancelled
            val nd = phase("gesture-templates") { rt.neural() }
            if (nd != null) neural = true
            val r = if (nd != null) phase("gesture-decode") { nd.decode(path, rt.slot.lm, lmCtx, i, langPrior(i), cancelled = cancel) }
            else {
                val g = phase("gesture-templates") { rt.gesture() } ?: continue
                phase("gesture-decode") { g.decode(path, rt.slot.lm, lmCtx, i, langPrior(i), cancelled = cancel) }
            }
            if (cancel?.invoke() == true) return Suggestions.EMPTY
            if (r == null) continue
            gesture = true
            hyps.addAll(r)
        }
        lastTaps = emptyList()
        // A very short glide over 2-3 keys ("ty" from t to the edge of y): the keys it covers, or the word they spell.
        val short = shortGlide(path, sm, ks, lmCtx)
        if (short != null) {
            // Only where the decoders have no word for it (too short to be a gesture) or just a single letter: their
            // words stay first otherwise (human FUTO/HWS short words lost a few when the keys outranked them).
            val top = hyps.minByOrNull { it.total }
            val decoderAgrees = top != null && ((config.swipe.shortGlideKeepsWords && top.canonical.count { Character.isLetter(it) } >= 2) ||
                isSubsequenceWord(primaryRules.lowercase(top.canonical), short.keys))
            if (!decoderAgrees) {
                val shape = if (context.sentenceStart) TypedShape.FIRST_UPPER else TypedShape.LOWER
                val texts = LinkedHashSet<String>()
                short.word?.let { texts.add(Casing.apply(it, shape, false, primaryRules)) }
                texts.add(Casing.apply(short.keys, shape, false, primaryRules))
                for (h in hyps.sortedBy { it.total }) { if (texts.size >= config.swipe.maxResults) break; texts.add(Casing.apply(h.canonical, shape, false, rulesOf(h))) }
                lastGlidePath = path
                return Suggestions(texts.mapIndexed { i, t -> Candidate(t, if (i == 0) 0.9f else 0.1f / texts.size, CandidateKind.CORRECTION) }, 0)
            }
        }
        if (!gesture) {
            // Too short to be a gesture: decode as a single tap on the nearest key.
            val p = path[0]
            val arr = FloatArray(ks.size)
            val k = sm.costs(p.x, p.y, arr)
            if (k < 0) return Suggestions.EMPTY
            val label = ks.labels[k].toString()
            return suggestForTaps(context, listOf(p), if (context.sentenceStart) primaryRules.uppercase(label) else label)
        }
        hyps.removeAll { blocked(it, null) }
        hyps.sortBy { it.total }
        if (final) phase("rerank") { rerank(hyps, lmCtx, if (neural) config.swipe.neuralLmWeight else config.swipe.lmWeight, null) }
        // Dialects: the top gesture candidate is shown in the active dialect's spelling; the other
        // spelling stays available as the next suggestion.
        if (hyps.isNotEmpty()) {
            val pref = preferredVariant(hyps[0], lmCtx)
            if (pref !== hyps[0]) hyps.add(0, WordHypothesis(pref.lang, pref.wordId, pref.canonical, pref.channelCost, pref.lmLogProb, hyps[0].total - 1e-3f, false, 0))
        }
        val shape = if (context.sentenceStart) TypedShape.FIRST_UPPER else TypedShape.LOWER
        val entries = ArrayList<Entry>()
        val seen = HashSet<String>()
        val folded = CaseFold()
        for (h in hyps) {
            if (entries.size >= config.swipe.maxResults) break
            val text = Casing.apply(h.canonical, shape, false, rulesOf(h))
            if (folded.weakVariant(text, h.total)) continue
            if (seen.add(text)) { val e = Entry(text, h.total, CandidateKind.CORRECTION); entries.add(e); folded.note(e) }
        }
        // CO-06: the user deleted the previous glided word and is re-gliding much the same path, so the
        // deleted word was not what they meant: commit the alternative, keep the deleted word offered.
        // With gesture boundaries from the host, the deletion applies to every decode of the next gesture (its
        // live previews and the final decode); without them, to the next decode only.
        val deleted = if (gestureBoundaries) gestureDeleted else deletedGlide.also { deletedGlide = null }
        if (deleted != null && entries.size >= 2 && similarPath(deleted.first, path) &&
            primaryRules.lowercase(entries[0].text) == deleted.second) {
            val e = entries.removeAt(0); entries.add(1, e)
        }
        lastGlidePath = path
        return Suggestions(toCandidates(entries), if (entries.isEmpty()) -1 else 0)
    }

    /** The keys a short glide covers ([keys], lower case) and the likeliest word they spell, if any. */
    private class ShortGlide(val keys: String, val word: String?)

    /**
     * A glide of at most [dev.pboard.engine.swipe.SwipeParams.shortGlideMax] key widths over 2+ keys: the keys in
     * order (where it starts, the keys it runs at least a fifth of a key width through, where it ends), and the
     * likeliest word whose letters are those keys or a subsequence with the same first and last key ("in" for a
     * path over i, u, n). Null for longer paths or a path inside one key.
     */
    private fun shortGlide(path: List<TouchPoint>, sm: SpatialModel, ks: KeySpace, lmCtx: List<String>): ShortGlide? {
        if (path.size < 2) return null
        val kw = ks.keyWidth
        var len = 0f
        for (i in 1 until path.size) len += hypot(path[i].x - path[i - 1].x, path[i].y - path[i - 1].y)
        if (len > config.swipe.shortGlideMax * kw) return null
        val arr = FloatArray(ks.size)
        val keys = ArrayList<Int>(); val runs = ArrayList<Float>()
        for ((i, p) in path.withIndex()) {
            val k = sm.costs(p.x, p.y, arr)
            if (k < 0) continue
            val d = if (i == 0) 0f else hypot(p.x - path[i - 1].x, p.y - path[i - 1].y)
            if (keys.isEmpty() || keys.last() != k) { keys.add(k); runs.add(d) } else runs[runs.size - 1] += d
        }
        val sb = StringBuilder()
        for (i in keys.indices) {
            if (i != 0 && i != keys.lastIndex && runs[i] < SHORT_GLIDE_PASS * kw) continue
            val c = ks.labels[keys[i]]
            if (!Character.isLetter(c)) return null
            if (sb.isEmpty() || sb.last() != c) sb.append(c)
        }
        if (sb.length < 2 || sb.length > 5) return null
        val letters = sb.toString()
        // Subsequences that keep the first and last key: at most 2^3 for five keys.
        var best: String? = null
        var bestLp = Float.NEGATIVE_INFINITY
        val mid = letters.length - 2
        for (mask in 0 until (1 shl mid)) {
            val w = StringBuilder().append(letters[0])
            for (j in 0 until mid) if (mask and (1 shl j) != 0) w.append(letters[j + 1])
            w.append(letters.last())
            val cand = w.toString()
            for ((i, rt) in langs.withIndex()) {
                if (rt.slot.lexicon.idOf(cand) < 0 || (config.blockOffensive && rt.slot.lexicon.isOffensive(rt.slot.lexicon.idOf(cand)))) continue
                val lp = rt.slot.lm.logProb(lmCtx, cand) + langPrior(i)
                if (lp > bestLp) { bestLp = lp; best = cand }
            }
        }
        return ShortGlide(letters, best)
    }

    /** [word]'s letters are [keys] or a subsequence of them with the same first and last letter. */
    private fun isSubsequenceWord(word: String, keys: String): Boolean {
        val w = word.filter { Character.isLetter(it) }
        if (w.isEmpty() || w.first() != keys.first() || w.last() != keys.last()) return false
        var j = 0
        for (c in keys) if (j < w.length && w[j] == c) j++
        return j == w.length
    }

    /** Diagnostics (eval only): where [target] was lost when decoding [path] in the primary language. */
    fun traceSwipe(context: Context, path: List<TouchPoint>, target: String): dev.pboard.engine.swipe.SwipeTrace? {
        val rt = langs.first()
        val trace = dev.pboard.engine.swipe.SwipeTrace(rt.slot.rules.lowercase(target))
        rt.gesture()?.decode(path, rt.slot.lm, lmContext(context), 0, langPrior(0), trace) ?: return null
        return trace
    }

    /** The last gesture decoded, and (path, word) of a glided word the user deleted right after (CO-06). */
    private var lastGlidePath: List<TouchPoint>? = null
    private var deletedGlide: Pair<List<TouchPoint>, String>? = null

    /**
     * The IME deleted the word it had just committed from a glide (one backspace removes the whole glided
     * word, CO-06). If the next gesture traces much the same path, the deleted word is demoted below the
     * runner-up. Proposed API: KeyboardEngine.onGestureWordDeleted.
     */
    override fun onGestureWordDeleted(context: Context, word: String) {
        val p = lastGlidePath ?: return
        deletedGlide = p to primaryRules.lowercase(word)
    }

    /** Whether the host reports gesture starts ([onGestureStarted]); the deletion that applies to the current gesture. */
    private var gestureBoundaries = false
    private var gestureDeleted: Pair<List<TouchPoint>, String>? = null

    override fun onGestureStarted() {
        gestureBoundaries = true
        gestureDeleted = deletedGlide
        deletedGlide = null
    }

    private fun similarPath(a: List<TouchPoint>, b: List<TouchPoint>): Boolean {
        val kw = keySpace?.keyWidth ?: return false
        fun d(p: TouchPoint, q: TouchPoint) = kotlin.math.hypot(p.x - q.x, p.y - q.y) / kw
        if (a.isEmpty() || b.isEmpty()) return false
        return d(a.first(), b.first()) < 1.0f && d(a.last(), b.last()) < 1.5f
    }

    // ---------------------------------------------------------------- prediction

    override fun predictNext(context: Context): Suggestions = phase("predict") { predictNextImpl(context) }

    /**
     * A cold context (field opened, cursor moved, word re-opened) costs the neural LM one step per word of the
     * sentence before the cursor: up to ~12 on the first glide or tap there (phone: a swipe's final decode p99
     * 41.5 ms). Runs those steps, and builds the context's n-gram completion list, ahead of the first input.
     */
    override fun prepareContext(context: Context) = phase("prepare") {
        val lmCtx = lmContext(context)
        if (config.neuralRerankTopN > 0) for (rt in langs) rt.slot.rescorer?.output(lmCtx)
        contextLms(lmCtx, active()).forEach { it.predictions }
    }

    /** Forgets the neural LM's cached states (benchmarks of cold contexts). */
    internal fun forgetNeuralStates() { for (rt in langs) rt.slot.rescorer?.clearStates() }

    private fun predictNextImpl(context: Context): Suggestions {
        val lmCtx = lmContext(context)
        val shape = if (context.sentenceStart) TypedShape.FIRST_UPPER else TypedShape.LOWER
        val merged = ArrayList<Triple<String, Float, Int>>()
        // The same per-context list the next word's first keystroke uses for completions: computing it here,
        // right after a commit, takes that work off the keystroke.
        val ctxLms = contextLms(lmCtx, active())
        for ((i, rt) in langs.withIndex()) {
            val rescorer = rt.slot.rescorer?.takeIf { config.neuralRerankTopN > 0 }
            // With a rescorer the n-gram list is still the next word's completion source (built here, so the first
            // keystroke does not pay for it), and the rescorer re-ranks its head instead of predicting again.
            val ngramPreds = if (rescorer != null && config.tap.contextPredictions >= rescorer.rescoreN)
                phase("predict-ngram") { ctxLms[i].predictions } else null
            val preds = when {
                rescorer != null && ngramPreds != null ->
                    phase("predict-rescore") { rescorer.predictFrom(lmCtx, ngramPreds, config.predictions + 6) }
                rescorer != null -> phase("predict-rescore") { rescorer.predict(lmCtx, "", config.predictions + 6) }
                config.tap.contextPredictions >= config.predictions + 6 -> ctxLms[i].predictions
                else -> rt.slot.lm.predict(lmCtx, "", config.predictions + 6)
            }
            for ((w, lp) in preds) merged.add(Triple(w, lp + langPrior(i), i))
        }
        merged.sortByDescending { it.second }
        publishContextPrior(context, merged.map { it.first to it.second })
        val out = ArrayList<Candidate>()
        val seen = HashSet<String>()
        for ((w, lp, i) in merged) {
            if (out.size >= config.predictions) break
            if (w.isEmpty() || !Character.isLetter(w[0])) continue
            val lex = langs[i].slot.lexicon
            if (config.blockOffensive && lex.idOf(w).let { it >= 0 && lex.isOffensive(it) }) continue
            if (primaryRules.lowercase(w) in suppressed) continue
            val text = Casing.apply(w, shape, false, langs[i].slot.rules)
            if (seen.add(text)) out.add(Candidate(text, exp(lp.toDouble()).toFloat(), CandidateKind.PREDICTION))
        }
        return Suggestions(out)
    }

    // ---------------------------------------------------------------- learning

    override fun onWordCommitted(context: Context, word: String, wasTypedLiterally: Boolean) {
        phase("commit-lstm") { prefetchNext(context, word) }
        rerankContext = null
        noteCommit(context, word)
        if (!context.learningAllowed) { lastTaps = emptyList(); return }
        val lmCtx = lmContext(context)
        if (langs.size > 1) mixer.observe(FloatArray(langs.size) { langs[it].slot.lm.logProb(lmCtx, word) })
        val store = learned
        // The LM's user model learns every committed word in context (phrases, and new words for the decoders);
        // the standalone store only learns unknown words that were kept as typed.
        if (store is NgramUserWords) store.observe(lmCtx, storedForm(context, word))
        else learnWord(context, word, wasTypedLiterally)
        val taps = lastTaps
        lastTaps = emptyList()
        val sm = spatial ?: return
        val ks = keySpace ?: return
        if (!config.adaptTouchOffsets || taps.isEmpty()) return
        // Learn only from unambiguous alignments: one tap per key, each within a key width of its key.
        val skeleton = primaryRules.skeleton(word)
        if (skeleton.length != taps.size) return
        val keyIdx = IntArray(skeleton.length)
        for (i in skeleton.indices) {
            val k = ks.keyOf(skeleton[i])
            if (k < 0 || sm.distanceToKey(k, taps[i].x, taps[i].y) > 1.0f) return
            keyIdx[i] = k
        }
        for (i in skeleton.indices) {
            val (ox, oy) = sm.residual(keyIdx[i], taps[i].x, taps[i].y)
            touchOffsets.observe(ks.labels[keyIdx[i]], ox, oy)
        }
        for (rt in langs) rt.tap.invalidate()
        userRuntime?.tap?.invalidate()
        if (touchOffsets.version != resolverOffsetsVersion) rebuildTapResolver()
    }

    /**
     * The form a committed word is learned in (CO-07: as the user writes it mid-sentence). A capital that only marks
     * the sentence start is not the word's: "Hello" at the start of messages is learned as "hello" (docs/app-compat.md
     * #3: learning "Hello" made a mid-sentence "hello" come out as "Hello"). As AOSP LatinIME, the capital is kept only
     * for a word known with it and not in lower case ("Monday", a name), or one the user has typed capitalised
     * mid-sentence and never in lower case.
     */
    private fun storedForm(context: Context, word: String): String {
        if (!context.sentenceStart || TypedShape.of(word) != TypedShape.FIRST_UPPER) return word
        val lower = primaryRules.lowercase(word.substring(0, 1)) + word.substring(1)
        val learnedLower = learned.count(lower) > 0.0
        // Typed capitalised mid-sentence (learned with the capital: sentence starts no longer add it) and never in lower case.
        if (learned.count(word) > 0.0 && !learnedLower) return word
        val lowerKnown = learnedLower || langs.any { it.slot.lexicon.contains(lower) }
        return if (!lowerKnown && langs.any { it.slot.lexicon.contains(word) }) word else lower
    }

    private fun learnWord(context: Context, word: String, literal: Boolean, weight: Int = 1) {
        if (!literal || word.length < 2 || !word.all(::isWordChar)) return
        val stored = storedForm(context, word)
        val store = learned
        if (store is NgramUserWords) { store.observe(lmContext(context), stored, weight); return }
        if (langs.any { it.slot.lexicon.contains(stored) || it.slot.lm.isKnown(stored) }) return
        userVocabulary.observe(stored, weight)
    }

    /**
     * A word left in the field as typed (session ended or cursor moved away; SU-09). A word no language knows is
     * learnt as intended at once, as when undoing a correction: Gboard 18.6 keeps an abandoned "becuase" on space the
     * next time and offers it from "bec". A known word counts once, like a commit (the last word of every message
     * sent while composing must not outweigh the words committed with a space).
     */
    override fun onWordAbandoned(context: Context, word: String) {
        if (!context.learningAllowed) return
        val stored = storedForm(context, word)
        val known = langs.any { it.slot.lexicon.contains(stored) || it.slot.lm.isKnown(stored) }
        learnWord(context, word, true, if (known) 1 else config.learning.intendedMinCount)
    }

    override fun onAutoCorrectReverted(context: Context, original: String, corrected: String) {
        if (!context.learningAllowed) return
        rejected.getOrPut(primaryRules.lowercase(original)) { HashSet() }.add(primaryRules.lowercase(corrected))
        // Undoing a correction is an explicit "I meant this": learn it as intended straight away.
        learnWord(context, original, true, config.learning.intendedMinCount)
    }

    /** Picking the literal typed word from the strip is an explicit "keep this": learn it as intended. */
    override fun onSuggestionPicked(context: Context, candidate: Candidate, typed: String) {
        if (!context.learningAllowed) return
        if (candidate.kind == CandidateKind.TYPED || candidate.text == typed) learnWord(context, candidate.text, true, config.learning.intendedMinCount)
    }

    override fun removeSuggestion(word: String) {
        learned.remove(word)
        userVocabulary.remove(word)
        val l = primaryRules.lowercase(word)
        userVocabulary.snapshot().keys.filter { primaryRules.lowercase(it) == l }.forEach { userVocabulary.remove(it) }
        suppressed.add(l)
        memoContext = null
    }

    override fun exportUserData(): ByteArray = UserDataCodec.encode(
        UserDataCodec.Data(userVocabulary.snapshot(), touchOffsets.snapshot(), suppressed.toSet(), rejected.mapValues { it.value.toSet() },
            (learned as? NgramUserWords)?.save()),
    )

    override fun importUserData(data: ByteArray): Boolean {
        val d = UserDataCodec.decode(data) ?: return false
        // The LM user model first: if its (nested, separately versioned) state is unreadable, nothing is applied.
        val lm = learned as? NgramUserWords
        if (d.lmUser != null && lm != null && !lm.load(d.lmUser)) return false
        // Migration (format <= 2 -> 3), once: sentence-start capitals learned as the word's form ("Hello" from the start
        // of messages made a mid-sentence "hello" come out as "Hello") fold into the lower-case word when that is a
        // dictionary word and the capital was never seen mid-sentence (see storedForm).
        if (d.version < 3 && lm != null) lm.foldSentenceStartCapitals { lower -> langs.any { it.slot.lexicon.contains(lower) } }
        userVocabulary.restore(d.learnedWords)
        // Migration (format 1 -> 2): before the LM user model existed, learned words lived in [userVocabulary]. With an
        // LM user model they move into it once (as context-free counts) and leave userVocabulary, so the next export
        // carries them only in the LM state and an evicted word is not resurrected by a later import.
        if (lm != null) for ((w, c) in d.learnedWords) {
            if (lm.count(w) <= 0.0) lm.observe(emptyList(), w, c.coerceIn(1, 50))
            userVocabulary.remove(w)
        }
        touchOffsets.restore(d.touchOffsets)
        rebuildTapResolver()
        suppressed.clear(); suppressed.addAll(d.suppressed)
        rejected.clear(); for ((k, v) in d.rejected) rejected[k] = v.toHashSet()
        for (rt in langs) rt.tap.invalidate()
        memoContext = null
        return true
    }

    override fun clearUserData() {
        learned.clear()
        userVocabulary.clear()
        touchOffsets.reset()
        rebuildTapResolver()
        suppressed.clear()
        rejected.clear()
        for (rt in langs) rt.tap.invalidate()
        memoContext = null
    }

    companion object {
        /** Spatial cost above which a typed character is considered not produced by the touched key. */
        private const val EXPLICIT_COST = 6f
        /** Strip slots a shortcut's word must already be in for it not to be added again. */
        private const val SHORTCUT_SLOT = 2
        /** Keys at most this far apart (centre to centre, in key widths) are neighbours: same-row and diagonal ones. */
        private const val ADJACENT_KEYS = 1.75f
        /** A bottom-row touch this close (key widths) above the space bar's top edge may be a missed space. */
        private const val NEAR_BAR = 0.35f
        /** A key a short glide runs at least this far (key widths) through counts as covered. */
        private const val SHORT_GLIDE_PASS = 0.2f

        /**
         * A language slot for an API-level spec (builds the lexicon with its locale's [TextRules]). A neural LM is
         * split into its n-gram model, which drives the searches, and the rescorer (unless
         * [EngineConfig.neuralFullSearch]).
         */
        fun slotFor(spec: LanguageSpec, config: EngineConfig): LanguageSlot {
            val lexicon = Lexicon.build(spec.lexicon.entries().asIterable(), TextRules.forLanguageTag(spec.tag))
            val lm = spec.lm
            return if (lm is NeuralRescoringLm && !config.neuralFullSearch) LanguageSlot(spec.tag, lm.ngram, lexicon, spec.weight, lm)
            else LanguageSlot(spec.tag, lm, lexicon, spec.weight)
        }

        /** An engine for API-level language specs (builds each lexicon with its locale's [TextRules]). */
        fun create(languages: List<LanguageSpec>, config: EngineConfig = EngineConfig()): PboardEngine {
            require(languages.isNotEmpty()) { "at least one language" }
            return PboardEngine(languages.map { slotFor(it, config) }, config)
        }
    }
}
