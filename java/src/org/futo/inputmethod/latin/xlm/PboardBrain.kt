/*
 * Copyright (C) 2026 Studio Keys
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.futo.inputmethod.latin.xlm

import android.content.Context
import android.util.Log
import dev.pboard.engine.lm.EnglishPack
import dev.pboard.engine.lm.LanguagePack
import dev.pboard.engine.lm.LmPack
import org.futo.inputmethod.latin.SuggestedWords
import org.futo.inputmethod.latin.utils.SuggestionResults
import java.io.FileInputStream
import java.nio.channels.FileChannel
import java.util.Locale
import java.util.concurrent.Executors

/**
 * Studio Keys: the vocabulary and language model of the pboard engine
 * (https://github.com/Ahmad-A0/pboard, MIT licence, see dev/pboard/ENGINE_LICENSE.txt and
 * dev/pboard/ENGINE_NOTICE.txt; the language packs are CC BY-SA 4.0, see assets/lm/LICENSE.txt).
 *
 * The whole engine is vendored under `java/src/dev/pboard/engine` because it is plain Kotlin/JVM with
 * no Android dependencies. This class is the only bridge the rest of the keyboard talks to.
 *
 * What it does today: it maps pboard's English language pack (23 MB, memory-mapped straight out of the
 * APK - the `.lm` extension is in the noCompress list) and answers "which words start with what I am
 * typing, most likely first". Those words are added to the suggestion list in [Suggest], so the strip is
 * filled from a much larger vocabulary than the three candidates the transformer model returns. Nothing
 * about auto-correction changes here: the added words always rank below the keyboard's own best
 * suggestion, so space still commits exactly what it committed before.
 */
object PboardBrain {
    private const val TAG = "PboardBrain"

    /** How many completions to offer per keystroke. */
    private const val MAX_COMPLETIONS = 6

    /** pboard ships English language packs only. */
    private const val LOCALE_TAG = "en-GB"

    @Volatile private var pack: LanguagePack? = null
    @Volatile private var loadingStarted = false
    @Volatile private var disabled = false

    private val loader = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "PboardBrainLoader").apply { isDaemon = true }
    }

    /** The pack's own queries are read-only; the lock keeps unrelated callers off each other anyway. */
    private val queryLock = Any()

    /**
     * Starts loading the language pack in the background. Safe to call more than once; the first call
     * wins. Until the pack is ready every query returns nothing, so typing keeps working normally.
     */
    @JvmStatic
    fun init(context: Context) {
        if (loadingStarted || disabled) return
        loadingStarted = true

        val appContext = context.applicationContext
        loader.execute {
            try {
                val base = appContext.assets.openFd(EnglishPack.BASE_ASSET).use { fd ->
                    FileInputStream(fd.fileDescriptor).use { input ->
                        input.channel.map(
                            FileChannel.MapMode.READ_ONLY,
                            fd.startOffset,
                            fd.length
                        ).also { it.load() }
                    }
                }
                val overlay = appContext.assets.open(EnglishPack.overlayAsset(LOCALE_TAG))
                    .use { it.readBytes() }
                val loaded = EnglishPack.load(LOCALE_TAG, LmPack.wrap(base), overlay)
                pack = loaded
                Log.i(TAG, "pboard engine ready: ${loaded.size} words")
            } catch (t: Throwable) {
                disabled = true
                Log.e(TAG, "pboard engine unavailable, falling back to the built-in one", t)
            }
        }
    }

    @JvmStatic
    fun isReady(): Boolean = pack != null

    /** Words that continue [prefix], most likely first, with their log probabilities. */
    fun completions(prefix: String, limit: Int = MAX_COMPLETIONS): List<Pair<String, Float>> {
        val loaded = pack ?: return emptyList()
        if (prefix.length < 2) return emptyList()
        return try {
            synchronized(queryLock) {
                loaded.model.predict(emptyList(), prefix.trim(), limit)
            }
        } catch (t: Throwable) {
            Log.w(TAG, "completion query failed", t)
            emptyList()
        }
    }

    /**
     * Adds pboard's completions of [typedWord] to [results]. They are ranked directly below the
     * keyboard's own best suggestion so that auto-correction (which is decided from the first entry)
     * is untouched, while the words are high enough in the list to be shown in the strip.
     */
    @JvmStatic
    fun addCompletions(results: SuggestionResults?, typedWord: String?, locale: Locale?) {
        if (results == null || typedWord == null || locale == null) return
        if (!isReady() || results.isEmpty()) return
        if (locale.language != "en") return

        val typed = typedWord.trim()
        if (typed.length < 2) return
        if (typed.any { it.isWhitespace() }) return

        val present = HashSet<String>()
        for (info in results) {
            present.add(info.mWord.lowercase(Locale.ROOT))
        }
        present.add(typed.lowercase(Locale.ROOT))

        val top = results.first().mScore
        var score = if (top > Int.MIN_VALUE + 64) top - 1 else 1

        var added = 0
        for ((word, _) in completions(typed)) {
            if (added >= MAX_COMPLETIONS) break
            if (word.isEmpty() || word.any { it.isWhitespace() }) continue
            if (!present.add(word.lowercase(Locale.ROOT))) continue

            val info = SuggestedWords.SuggestedWordInfo(
                word,
                "",
                score,
                SuggestedWords.SuggestedWordInfo.KIND_PREDICTION,
                null,
                0,
                0
            )
            // Shown with the same badge the transformer suggestions use.
            info.mOriginatesFromTransformerLM = true
            results.add(info)

            score -= 1
            added += 1
        }
    }
}
