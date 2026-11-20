package dev.pboard.engine.lm

import dev.pboard.engine.LanguageModel
import dev.pboard.engine.LanguageSpec
import dev.pboard.engine.LexiconEntry
import dev.pboard.engine.LexiconSource
import java.io.File
import java.nio.ByteBuffer

/**
 * One loaded language/dialect: its [LanguageModel] and its vocabulary as a [LexiconSource] for the decoders.
 * Destructures like a pair: `val (lm, lexicon) = EnglishPack.load("en-GB", base, overlay)`.
 *
 * Lexicon entries carry:
 * - `word`: the canonical spelling ("I'm", "iPhone", "London");
 * - `logPrior`: the word's log relative frequency, with the dialect's spelling and vocabulary preferences
 *   applied (in en-GB "colour" outranks "color"; in en-US the reverse);
 * - `variantKey`: shared by the spellings of one word across dialects. The representative is the American
 *   form, so "colour" and "color" both have key "color";
 * - `offensive`: never suggest it unless it was typed in full.
 */
class LanguagePack(
    val tag: String,
    /** The n-gram model (lexicon, learning, offensive flags). */
    val model: NgramLanguageModel,
    /** What the engine should query: [model], or a [NeuralRescoringLm] over it when a neural LM is loaded. */
    val lm: LanguageModel = model,
) : LexiconSource {
    val lexicon: LexiconSource get() = this

    operator fun component1(): LanguageModel = lm
    operator fun component2(): LexiconSource = this

    fun spec(weight: Float = 1f): LanguageSpec = LanguageSpec(tag, lm, this, weight)

    override fun entries(): Sequence<LexiconEntry> = sequence {
        val pack = model.pack
        for (id in 0 until pack.nLex) {
            val v = pack.variantOf(id)
            yield(LexiconEntry(pack.word(id), model.wordLogPrior(id), if (v >= 0) pack.word(v) else null, model.isOffensiveId(id)))
        }
    }

    /** Number of vocabulary entries. */
    val size: Int get() = model.pack.nLex

    /**
     * Optional, call right after a commit: starts the neural LM's step for [context] on a background thread (see
     * [NeuralRescoringLm.prefetch]), so the next query in that context only pays for rescoring. Does nothing
     * without a neural LM.
     */
    fun prefetch(context: List<String>) {
        (lm as? NeuralRescoringLm)?.prefetch(context)
    }
}

/**
 * The shipped English pack: one shared base (`assets/lm/en.lm`) plus a dialect overlay per variety
 * (`assets/lm/en_GB.lmo`, `assets/lm/en_US.lmo`).
 *
 * On Android, map the base straight out of the APK (it is stored uncompressed) and share it between
 * dialects:
 * ```
 * val fd = context.assets.openFd(EnglishPack.BASE_ASSET)
 * val base = LmPack.wrap(FileInputStream(fd.fileDescriptor).channel.map(READ_ONLY, fd.startOffset, fd.length))
 * val gb = EnglishPack.load("en-GB", base, context.assets.open(EnglishPack.overlayAsset("en-GB")).readBytes())
 * engine.setLanguages(listOf(gb.spec()))
 * ```
 */
object EnglishPack {
    const val BASE_ASSET = "lm/en.lm"

    /** Dialect tags with a shipped overlay. */
    val TAGS = listOf("en-GB", "en-US")

    /**
     * Overlay used for a BCP-47 English tag. British spelling serves the Commonwealth varieties; American
     * spelling serves en-US and anything unrecognised. Canada mixes both conventions and is mapped to en-US.
     */
    fun dialectOf(tag: String): String {
        val region = tag.replace('_', '-').substringAfter('-', "").uppercase()
        return if (region in BRITISH_SPELLING_REGIONS) "en_GB" else "en_US"
    }

    fun overlayAsset(tag: String): String = "lm/${dialectOf(tag)}.lmo"

    /** Optional neural rescorer (toggle by loading it or not): `NeuralLm.read(assets.open(NEURAL_ASSET).readBytes())`. */
    const val NEURAL_ASSET = "lm/en.nnlm"

    /**
     * Loads [tag] over an already opened base pack (share one [LmPack], and one [NeuralLm], between dialects).
     * With [neural], the pack's [LanguagePack.lm] is a [NeuralRescoringLm] over the n-gram model.
     */
    fun load(
        tag: String, base: LmPack, overlay: ByteArray?, user: UserNgramModel? = null, neural: NeuralLm? = null,
    ): LanguagePack {
        val ngram = NgramLanguageModel(base, overlay?.let { DialectOverlay.read(it) }, user)
        return LanguagePack(normalize(tag), ngram, if (neural != null) NeuralRescoringLm(ngram, neural) else ngram)
    }

    /** Loads [tag] from the base pack bytes (preferably a memory-mapped buffer) and the overlay bytes. */
    fun load(
        tag: String, base: ByteBuffer, overlay: ByteArray?, user: UserNgramModel? = null, neural: NeuralLm? = null,
    ): LanguagePack = load(tag, LmPack.wrap(base), overlay, user, neural)

    /**
     * JVM convenience: loads from a directory holding en.lm, the .lmo overlays and optionally en.nnlm
     * (e.g. app/src/main/assets/lm).
     */
    fun load(tag: String, dir: File, user: UserNgramModel? = null, useNeural: Boolean = false): LanguagePack {
        val overlay = File(dir, "${dialectOf(tag)}.lmo")
        val nn = File(dir, "en.nnlm").takeIf { useNeural && it.exists() }?.let { NeuralLm.read(it) }
        return load(tag, LmPack.open(File(dir, "en.lm")), if (overlay.exists()) overlay.readBytes() else null, user, nn)
    }

    private fun normalize(tag: String) = "en-" + dialectOf(tag).substringAfter('_')

    private val BRITISH_SPELLING_REGIONS = setOf(
        "GB", "UK", "IE", "AU", "NZ", "ZA", "IN", "SG", "MY", "HK", "PK", "NG", "KE", "GH", "MT", "JM",
    )
}
