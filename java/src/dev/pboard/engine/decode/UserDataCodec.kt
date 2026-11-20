package dev.pboard.engine.decode

import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import dev.pboard.engine.lm.UserNgramModel

/**
 * Binary format of the personal data the engine learns on device (see KeyboardEngine.exportUserData).
 * The IME stores the bytes in app-private, no-backup storage; nothing here ever leaves the device.
 *
 * Layout (big-endian, DataOutputStream): magic "PBUD", version, then sections each as a count followed by
 * records — learned words (word, count), touch offsets (key label, Σdx, Σdy, weight), removed suggestions
 * (word), undone corrections (original, [corrections]); version 2 appends the language model's user n-gram
 * state (length-prefixed bytes, length -1 when absent; its own versioned format, see UserNgramModel).
 * Decoding is strict: anything unexpected returns null so a corrupt file is ignored rather than half-applied.
 *
 * Version history. Every version ever written MUST stay readable (an update keeps the user's file); the frozen
 * samples in engine/src/test/resources/userdata/ are loaded by UserDataFormatTest to enforce that.
 *   1 = learned words, touch offsets, removed suggestions, undone corrections
 *   2 = + the LM user n-gram state
 *   3 = same layout as 2; the LM user state has had sentence-start capitals folded (PboardEngine.importUserData
 *       runs that repair once on data of version 2 or older: "Hello" learned at message starts -> "hello")
 * Data from a NEWER version (after a downgrade) is refused (null); the IME then keeps that file aside instead of
 * overwriting it (dev.pboard.ime.engine.Engines).
 */
object UserDataCodec {
    private const val MAGIC = 0x50425544 // "PBUD"
    const val FORMAT_VERSION = 3
    private const val MAX_ITEMS = 1_000_000

    data class Data(
        val learnedWords: Map<String, Int>,
        val touchOffsets: Map<Char, FloatArray>,
        val suppressed: Set<String>,
        val rejected: Map<String, Set<String>>,
        /** The LM user model's own serialised state (UserNgramModel), when the language model has one. */
        val lmUser: ByteArray? = null,
        /** The format version the data was decoded from (encode always writes [FORMAT_VERSION]). */
        val version: Int = FORMAT_VERSION,
    )

    /**
     * [version] is written in the header: [FORMAT_VERSION] for data the engine exports. Merged data ([merge]) that
     * still needs a migration the engine runs on import is written with its older version (2 and 3 share this layout).
     */
    fun encode(d: Data, version: Int = FORMAT_VERSION): ByteArray {
        require(version in 2..FORMAT_VERSION)
        val bytes = ByteArrayOutputStream()
        DataOutputStream(bytes).use { out ->
            out.writeInt(MAGIC); out.writeInt(version)
            out.writeInt(d.learnedWords.size)
            for ((w, c) in d.learnedWords) { out.writeUTF(w); out.writeInt(c) }
            out.writeInt(d.touchOffsets.size)
            for ((k, v) in d.touchOffsets) { out.writeChar(k.code); out.writeFloat(v[0]); out.writeFloat(v[1]); out.writeFloat(v[2]) }
            out.writeInt(d.suppressed.size)
            for (w in d.suppressed) out.writeUTF(w)
            out.writeInt(d.rejected.size)
            for ((o, cs) in d.rejected) { out.writeUTF(o); out.writeInt(cs.size); for (c in cs) out.writeUTF(c) }
            val lm = d.lmUser
            if (lm == null) out.writeInt(-1) else { out.writeInt(lm.size); out.write(lm) }
        }
        return bytes.toByteArray()
    }

    /** The format version in the header of [bytes], or null when they are not pboard user data at all. */
    fun versionOf(bytes: ByteArray): Int? {
        if (bytes.size < 8) return null
        val b = java.nio.ByteBuffer.wrap(bytes)
        return if (b.getInt(0) == MAGIC) b.getInt(4) else null
    }

    fun decode(bytes: ByteArray): Data? = try {
        DataInputStream(ByteArrayInputStream(bytes)).use { inp ->
            if (inp.readInt() != MAGIC) return null
            val version = inp.readInt()
            if (version !in 1..FORMAT_VERSION) return null
            fun count(): Int = inp.readInt().also { require(it in 0..MAX_ITEMS) }
            val words = LinkedHashMap<String, Int>()
            repeat(count()) { val w = inp.readUTF(); val c = inp.readInt(); if (c > 0 && w.isNotEmpty()) words[w] = c }
            val offsets = HashMap<Char, FloatArray>()
            repeat(count()) { val k = inp.readChar(); offsets[k] = floatArrayOf(inp.readFloat(), inp.readFloat(), inp.readFloat()) }
            val suppressed = HashSet<String>()
            repeat(count()) { suppressed.add(inp.readUTF()) }
            val rejected = HashMap<String, Set<String>>()
            repeat(count()) { val o = inp.readUTF(); val cs = HashSet<String>(); repeat(count()) { cs.add(inp.readUTF()) }; rejected[o] = cs }
            var lmUser: ByteArray? = null
            if (version >= 2) {
                val n = inp.readInt()
                if (n >= 0) { require(n <= 64 * 1024 * 1024); lmUser = ByteArray(n).also { inp.readFully(it) } }
            }
            if (inp.available() != 0) return null
            Data(words, offsets, suppressed, rejected, lmUser, version)
        }
    } catch (e: Exception) {
        null
    }

    // ------------------------------------------------------------------ restore files (untrusted input)

    /**
     * [decode] for bytes from outside the app (a restore file): additionally the nested LM user state must load
     * completely and every number must be sane. Null when anything is off; nothing is applied anywhere.
     */
    fun decodeStrict(bytes: ByteArray): Data? {
        val d = decode(bytes) ?: return null
        if (d.touchOffsets.values.any { v -> v.any { !it.isFinite() } || v[2] < 0f }) return null
        if (d.lmUser != null && loadUser(d.lmUser) == null) return null
        return d
    }

    private fun loadUser(bytes: ByteArray): UserNgramModel? = try {
        val input = ByteArrayInputStream(bytes)
        UserNgramModel().also { it.load(input) }.takeIf { input.available() == 0 }
    } catch (e: Exception) {
        null
    }

    /**
     * [incoming] (a restored backup) merged into [base] (what the device has now). Idempotent: merging the same backup
     * twice gives the same result. Learned words keep the larger count, a key's touch offsets the better-supported
     * estimate; removed suggestions and undone corrections are united; the LM user states are merged n-gram by n-gram
     * ([UserNgramModel.mergeFrom]). Null when an LM user state cannot be read: the caller must then change nothing.
     * The result's [Data.version] is the older of the two, so the engine still runs that version's migration on import.
     */
    fun merge(base: Data, incoming: Data): Data? {
        val words = LinkedHashMap(base.learnedWords)
        for ((w, c) in incoming.learnedWords) words.merge(w, c, ::maxOf)
        val offsets = HashMap(base.touchOffsets)
        for ((k, v) in incoming.touchOffsets) { val mine = offsets[k]; if (mine == null || v[2] > mine[2]) offsets[k] = v }
        val rejected = HashMap<String, Set<String>>(base.rejected)
        for ((o, cs) in incoming.rejected) rejected[o] = (rejected[o] ?: emptySet()) + cs
        val a = base.lmUser
        val b = incoming.lmUser
        val lm = when {
            a == null -> b
            b == null -> a
            else -> {
                val mine = loadUser(a) ?: return null
                mine.mergeFrom(loadUser(b) ?: return null)
                ByteArrayOutputStream().also { mine.save(it) }.toByteArray()
            }
        }
        return Data(words, offsets, base.suppressed + incoming.suppressed, rejected, lm, minOf(base.version, incoming.version))
    }

    /** [merge] on encoded data, for the IME: null when either side is unreadable. Empty [base] = nothing learned yet. */
    fun mergeEncoded(base: ByteArray, incoming: ByteArray): ByteArray? {
        val b = decodeStrict(incoming) ?: return null
        if (base.isEmpty()) return encode(b, b.version.coerceAtLeast(2))
        val m = merge(decode(base) ?: return null, b) ?: return null
        return encode(m, m.version.coerceAtLeast(2))
    }
}
