package dev.pboard.engine.lm

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.io.FileInputStream
import java.nio.channels.FileChannel

/**
 * Zero-copy reader of a base language pack (`<locale>.lm`, magic "PBLM"); see docs/lm-format.md.
 *
 * Nothing is parsed eagerly except the header, the section table, the quantisation tables and the small
 * locale-rule tables, so opening a pack is O(1) in its size (the OS pages the mapped file in lazily).
 * All accessors are thread-safe (absolute ByteBuffer reads only).
 */
class LmPack private constructor(private val buf: ByteBuffer) {
    val versionMajor: Int
    val flags: Int
    val locale: String
    val order: Int
    val nIds: Int
    val nLex: Int
    val oovPenalty: Float
    val bosLogProb: Float
    val buildId: Long
    private val counts = IntArray(9)
    private val recBits = IntArray(9)
    private val ptrBits = IntArray(9)
    private val sections = HashMap<Int, Pair<Int, Int>>()
    private val quantProb = arrayOfNulls<FloatArray>(9)
    private val quantBow = arrayOfNulls<FloatArray>(9)
    val rules: LocaleRules
    val clitics: List<String>
    val linkingMorphemes: List<String>

    private val offStrings: Int
    private val offStrOffsets: Int
    private val offLexNgram: Int
    private val offInClass: Int
    private val offFlags: Int
    private val offHash: Int
    private val hashMask: Int
    private val offClassStart: Int
    private val offClassMembers: Int
    private val offUnigram: Int
    private val offOrder = IntArray(9)
    private val offLogFreq: Int
    private val offVariants: Int
    private val nVariants: Int

    init {
        buf.order(ByteOrder.LITTLE_ENDIAN)
        require(buf.capacity() >= HEADER_SIZE) { "not a language pack" }
        val magic = ByteArray(4).also { for (i in 0 until 4) it[i] = buf.get(i) }
        require(String(magic, Charsets.US_ASCII) == "PBLM") { "bad magic" }
        versionMajor = buf.getShort(4).toInt() and 0xFFFF
        require(versionMajor == 1) { "unsupported pack version $versionMajor" }
        flags = buf.getInt(12)
        locale = ascii(16, 16)
        order = buf.getInt(32)
        nIds = buf.getInt(36)
        nLex = buf.getInt(40)
        for (k in 1..8) counts[k] = buf.getInt(44 + 4 * (k - 1))
        for (k in 2..8) recBits[k] = buf.get(76 + k).toInt() and 0xFF
        for (k in 2 until order) ptrBits[k] = 32 - Integer.numberOfLeadingZeros(maxOf(1, counts[k + 1]))
        oovPenalty = buf.getFloat(88)
        bosLogProb = buf.getFloat(92)
        buildId = buf.getLong(96)
        val nSec = buf.getInt(124)
        for (i in 0 until nSec) {
            val p = 128 + 24 * i
            sections[buf.getInt(p)] = buf.getLong(p + 8).toInt() to buf.getLong(p + 16).toInt()
        }
        offStrings = off(SEC_STRINGS)
        offStrOffsets = off(SEC_STR_OFFSETS)
        offLexNgram = off(SEC_LEX_NGRAM)
        offInClass = off(SEC_LEX_INCLASS)
        offFlags = off(SEC_LEX_FLAGS)
        offHash = off(SEC_HASH_LOWER)
        hashMask = len(SEC_HASH_LOWER) / 4 - 1
        offClassStart = off(SEC_CLASS_START)
        offClassMembers = off(SEC_CLASS_MEMBERS)
        offUnigram = off(SEC_UNIGRAM)
        offLogFreq = sections[SEC_LEX_LOGFREQ]?.first ?: -1
        offVariants = sections[SEC_VARIANTS]?.first ?: 0
        nVariants = len(SEC_VARIANTS) / 8
        for (k in 2..order) {
            offOrder[k] = off(SEC_ORDER_BASE + k)
            val q = off(SEC_QUANT_BASE + k)
            quantProb[k] = FloatArray(256) { buf.getFloat(q + 4 * it) }
            quantBow[k] = FloatArray(256) { buf.getFloat(q + 1024 + 4 * it) }
        }
        rules = LocaleRules.parse(text(SEC_CASE_TABLE), text(SEC_FOLD_TABLE))
        val ruleMap = text(SEC_LOCALE_RULES).split('\n').filter { '\t' in it }
            .associate { it.substringBefore('\t') to it.substringAfter('\t').split(' ').filter(String::isNotEmpty) }
        clitics = ruleMap["clitics"] ?: emptyList()
        linkingMorphemes = ruleMap["linking"] ?: emptyList()
    }

    private fun off(id: Int) = sections[id]?.first ?: error("pack lacks section $id")
    private fun len(id: Int) = sections[id]?.second ?: 0
    private fun ascii(at: Int, n: Int): String {
        val b = ByteArray(n) { buf.get(at + it) }
        return String(b, Charsets.US_ASCII).trimEnd('\u0000')
    }

    private fun text(id: Int): String {
        val (o, n) = sections[id] ?: return ""
        val b = ByteArray(n)
        for (i in 0 until n) b[i] = buf.get(o + i)
        return String(b, Charsets.UTF_8)
    }

    val isCompounding: Boolean get() = flags and FLAG_COMPOUNDING != 0
    fun count(k: Int): Int = counts[k]

    // ------------------------------------------------------------------------------------------ lexicon

    fun word(lexId: Int): String {
        val a = buf.getInt(offStrOffsets + 4 * lexId)
        val b = buf.getInt(offStrOffsets + 4 * lexId + 4)
        val bytes = ByteArray(b - a)
        for (i in bytes.indices) bytes[i] = buf.get(offStrings + a + i)
        return String(bytes, Charsets.UTF_8)
    }

    /** Appends the UTF-16 chars of word [lexId] to [out] without allocating. */
    fun wordInto(lexId: Int, out: StringBuilder) {
        var p = offStrings + buf.getInt(offStrOffsets + 4 * lexId)
        val end = offStrings + buf.getInt(offStrOffsets + 4 * lexId + 4)
        while (p < end) {
            val b0 = buf.get(p).toInt() and 0xFF
            val cp: Int
            when {
                b0 < 0x80 -> { cp = b0; p += 1 }
                b0 < 0xE0 -> { cp = (b0 and 0x1F shl 6) or (buf.get(p + 1).toInt() and 0x3F); p += 2 }
                b0 < 0xF0 -> {
                    cp = (b0 and 0x0F shl 12) or (buf.get(p + 1).toInt() and 0x3F shl 6) or (buf.get(p + 2).toInt() and 0x3F)
                    p += 3
                }
                else -> {
                    cp = (b0 and 0x07 shl 18) or (buf.get(p + 1).toInt() and 0x3F shl 12) or
                        (buf.get(p + 2).toInt() and 0x3F shl 6) or (buf.get(p + 3).toInt() and 0x3F)
                    p += 4
                }
            }
            out.appendCodePoint(cp)
        }
    }

    /** True if the pack stores per-word frequencies (format >= 1.1). */
    val hasLogFreq: Boolean get() = offLogFreq >= 0

    /** log relative frequency of word [lexId] in the training data (context-free prior; format >= 1.1). */
    fun logFreq(lexId: Int): Float =
        if (offLogFreq >= 0) buf.getShort(offLogFreq + 2 * lexId) / MILLI
        else unigramLogProb(ngramIdOf(lexId)) + inClassLogProb(lexId)

    /**
     * Representative lexicon id of [lexId]'s dialect spelling-variant group ("colour" -> "color"), or -1 if
     * the word has no dialect variant. Binary search over the sorted (lexId, representative) pairs.
     */
    fun variantOf(lexId: Int): Int {
        var a = 0
        var b = nVariants - 1
        while (a <= b) {
            val mid = (a + b) ushr 1
            val k = buf.getInt(offVariants + 8 * mid)
            when {
                k < lexId -> a = mid + 1
                k > lexId -> b = mid - 1
                else -> return buf.getInt(offVariants + 8 * mid + 4)
            }
        }
        return -1
    }

    fun ngramIdOf(lexId: Int): Int = buf.getShort(offLexNgram + 2 * lexId).toInt() and 0xFFFF
    fun inClassLogProb(lexId: Int): Float = buf.getShort(offInClass + 2 * lexId) / MILLI
    fun flags(lexId: Int): Int = buf.get(offFlags + lexId).toInt() and 0xFF

    /** Lexicon ids of the words of n-gram class [ngramId] (most frequent first). Empty for specials/<rare>. */
    fun classMemberCount(ngramId: Int): Int = buf.getInt(offClassStart + 4 * ngramId + 4) - buf.getInt(offClassStart + 4 * ngramId)
    fun classMember(ngramId: Int, i: Int): Int = buf.getInt(offClassMembers + 4 * (buf.getInt(offClassStart + 4 * ngramId) + i))

    /**
     * First lexicon id of the case-variant group whose lowercase form equals [lower] (already lowercased with
     * [rules]), or -1. Variants of the group are contiguous: iterate while [lowerEquals] holds.
     */
    fun findLowerGroup(lower: CharSequence, scratch: StringBuilder): Int {
        var h = FNV_OFFSET
        for (i in 0 until lower.length) {
            h = h xor lower[i].code
            h *= FNV_PRIME
        }
        var slot = h and hashMask
        while (true) {
            val v = buf.getInt(offHash + 4 * slot)
            if (v == 0) return -1
            val id = v - 1
            if (lowerEquals(id, lower, scratch)) return id
            slot = (slot + 1) and hashMask
        }
    }

    /** Code point at byte position [p] of the strings section; its UTF-8 length is [utf8Len]. */
    private fun cpAt(p: Int): Int {
        val b0 = buf.get(p).toInt() and 0xFF
        return when {
            b0 < 0x80 -> b0
            b0 < 0xE0 -> (b0 and 0x1F shl 6) or (buf.get(p + 1).toInt() and 0x3F)
            b0 < 0xF0 -> (b0 and 0x0F shl 12) or (buf.get(p + 1).toInt() and 0x3F shl 6) or (buf.get(p + 2).toInt() and 0x3F)
            else -> (b0 and 0x07 shl 18) or (buf.get(p + 1).toInt() and 0x3F shl 12) or
                (buf.get(p + 2).toInt() and 0x3F shl 6) or (buf.get(p + 3).toInt() and 0x3F)
        }
    }

    private fun utf8Len(p: Int): Int {
        val b0 = buf.get(p).toInt() and 0xFF
        return if (b0 < 0x80) 1 else if (b0 < 0xE0) 2 else if (b0 < 0xF0) 3 else 4
    }

    /** Matches the UTF-16 chars of code point [cp] against [s] at [i]; returns the new index or -1. */
    private fun matchCp(cp: Int, s: CharSequence, i: Int): Int {
        if (cp < 0x10000) return if (i < s.length && s[i].code == cp) i + 1 else -1
        return if (i + 1 < s.length && s[i] == Character.highSurrogate(cp) && s[i + 1] == Character.lowSurrogate(cp)) i + 2 else -1
    }

    /** True if word [lexId] lowercased with [rules] equals [lower] (no allocation). */
    fun lowerEquals(lexId: Int, lower: CharSequence, @Suppress("UNUSED_PARAMETER") scratch: StringBuilder? = null): Boolean {
        if (lexId < 0 || lexId >= nLex) return false
        var p = offStrings + buf.getInt(offStrOffsets + 4 * lexId)
        val end = offStrings + buf.getInt(offStrOffsets + 4 * lexId + 4)
        var i = 0
        while (p < end) {
            i = matchCp(rules.lowerCodePoint(cpAt(p)), lower, i)
            if (i < 0) return false
            p += utf8Len(p)
        }
        return i == lower.length
    }

    /** True if word [lexId] is exactly [word]. */
    fun wordEquals(lexId: Int, word: CharSequence): Boolean {
        var p = offStrings + buf.getInt(offStrOffsets + 4 * lexId)
        val end = offStrings + buf.getInt(offStrOffsets + 4 * lexId + 4)
        var i = 0
        while (p < end) {
            i = matchCp(cpAt(p), word, i)
            if (i < 0) return false
            p += utf8Len(p)
        }
        return i == word.length
    }

    /** Compares the fold key of word [lexId] with [prefix]: 0 if it starts with it, else the sign of the order. */
    fun compareFoldPrefix(lexId: Int, prefix: CharSequence, @Suppress("UNUSED_PARAMETER") a: StringBuilder? = null,
                          @Suppress("UNUSED_PARAMETER") b: StringBuilder? = null): Int {
        var p = offStrings + buf.getInt(offStrOffsets + 4 * lexId)
        val end = offStrings + buf.getInt(offStrOffsets + 4 * lexId + 4)
        var i = 0
        while (p < end) {
            if (i >= prefix.length) return 0
            val cp = cpAt(p)
            p += utf8Len(p)
            val f = rules.foldOf(cp)
            if (f == null) {
                val lc = rules.lowerCodePoint(cp)
                if (lc < 0x10000) {
                    val c = lc.toChar().compareTo(prefix[i])
                    if (c != 0) return c
                    i++
                } else {
                    for (ch in Character.toChars(lc)) {
                        if (i >= prefix.length) return 0
                        val c = ch.compareTo(prefix[i]); if (c != 0) return c; i++
                    }
                }
            } else {
                for (ch in f) {
                    if (i >= prefix.length) return 0
                    val c = ch.compareTo(prefix[i]); if (c != 0) return c; i++
                }
            }
        }
        return if (i >= prefix.length) 0 else -1
    }

    // ------------------------------------------------------------------------------------------ n-grams

    fun unigramLogProb(id: Int): Float = buf.getShort(offUnigram + 8 * id) / MILLI
    fun unigramBow(id: Int): Float = buf.getShort(offUnigram + 8 * id + 2) / MILLI
    fun unigramChildStart(id: Int): Int = buf.getInt(offUnigram + 8 * id + 4)

    /** Raw 64-bit window starting at record [idx] of order [k]: word in bits 0-15, prob 16-23, bow 24-31, ptr 32+. */
    fun record(k: Int, idx: Int): Long {
        val bitpos = idx.toLong() * recBits[k]
        return buf.getLong(offOrder[k] + (bitpos ushr 3).toInt()) ushr (bitpos and 7).toInt()
    }

    fun recWord(k: Int, idx: Int): Int = (record(k, idx) and 0xFFFF).toInt()
    fun recLogProb(k: Int, idx: Int): Float = quantProb[k]!![((record(k, idx) ushr 16) and 0xFF).toInt()]
    /** Log probability of an order-[k] record already read with [record] (saves a second read). */
    fun recLogProbOf(k: Int, rec: Long): Float = quantProb[k]!![((rec ushr 16) and 0xFF).toInt()]
    fun recBow(k: Int, idx: Int): Float = quantBow[k]!![((record(k, idx) ushr 24) and 0xFF).toInt()]
    fun recChildStart(k: Int, idx: Int): Int = ((record(k, idx) ushr 32) and ((1L shl ptrBits[k]) - 1)).toInt()

    /** Start of the children (order k+1) of node (k, idx); end is the start of (k, idx+1). */
    fun childStart(k: Int, idx: Int): Int = if (k == 1) unigramChildStart(idx) else recChildStart(k, idx)

    /** Binary search for word [w] among records [lo, hi) of order [k]; returns the index or -1. */
    fun findChild(k: Int, lo: Int, hi: Int, w: Int): Int {
        var a = lo
        var b = hi - 1
        while (a <= b) {
            val mid = (a + b) ushr 1
            val x = recWord(k, mid)
            when {
                x < w -> a = mid + 1
                x > w -> b = mid - 1
                else -> return mid
            }
        }
        return -1
    }

    /**
     * Index of the n-gram [ids][from, to) at order (to - from), or -1 if absent. Order 1 returns the id itself.
     */
    fun find(ids: IntArray, from: Int, to: Int): Int {
        var idx = ids[from]
        var k = 1
        var i = from + 1
        while (i < to) {
            if (k >= order) return -1
            val lo = childStart(k, idx)
            val hi = childStart(k, idx + 1)
            if (lo >= hi) return -1
            idx = findChild(k + 1, lo, hi, ids[i])
            if (idx < 0) return -1
            k++
            i++
        }
        return idx
    }

    fun logProbAt(k: Int, idx: Int): Float = if (k == 1) unigramLogProb(idx) else recLogProb(k, idx)
    fun bowAt(k: Int, idx: Int): Float = if (k == 1) unigramBow(idx) else if (k < order) recBow(k, idx) else 0f

    companion object {
        const val HEADER_SIZE = 1024
        const val MILLI = 1000f
        const val FLAG_COMPOUNDING = 1
        const val FLAG_CLITICS = 2
        const val F_OFFENSIVE = 1
        const val F_NO_SPACE_AFTER = 2

        const val UNK = 0
        const val BOS = 1
        const val EOS = 2
        const val COMMA = 3
        const val NUM = 4
        const val RARE = 5
        const val NSPECIAL = 6

        private const val SEC_STRINGS = 1
        private const val SEC_STR_OFFSETS = 2
        private const val SEC_LEX_NGRAM = 3
        private const val SEC_LEX_INCLASS = 4
        private const val SEC_LEX_FLAGS = 5
        private const val SEC_HASH_LOWER = 7
        private const val SEC_CLASS_START = 8
        private const val SEC_CLASS_MEMBERS = 9
        private const val SEC_CASE_TABLE = 11
        private const val SEC_FOLD_TABLE = 12
        private const val SEC_UNIGRAM = 13
        private const val SEC_LOCALE_RULES = 14
        private const val SEC_LEX_LOGFREQ = 15
        private const val SEC_VARIANTS = 16
        private const val SEC_ORDER_BASE = 20
        private const val SEC_QUANT_BASE = 40
        private const val FNV_OFFSET = 0x811C9DC5.toInt()
        private const val FNV_PRIME = 0x01000193

        /** Memory-maps a pack file (read-only). */
        // Studio Keys: mapped through a FileInputStream rather than java.nio.file (which needs API 26),
        // so the engine still works on the keyboard's minimum Android version.
        fun open(file: File): LmPack = FileInputStream(file).use { input ->
            LmPack(input.channel.map(FileChannel.MapMode.READ_ONLY, 0, input.channel.size()))
        }

        /** Wraps an existing buffer (e.g. an asset mapped through an AssetFileDescriptor). */
        fun wrap(buffer: ByteBuffer): LmPack = LmPack(buffer.duplicate())
    }
}
