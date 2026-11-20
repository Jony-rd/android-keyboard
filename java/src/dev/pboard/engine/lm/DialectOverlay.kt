package dev.pboard.engine.lm

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * A regional-variety layer (`<dialect>.lmo`, magic "PBLO") over a base pack: per-word log-probability
 * adjustments (spelling choice inside VarCon spelling classes such as colour/color, lexical preferences such
 * as mum/mom) and extra word flags. Loaded fully into memory (a few hundred kB).
 */
class DialectOverlay private constructor(
    val locale: String,
    val baseBuildId: Long,
    val logZ: Float,
    private val adjust: ShortArray,
    private val flagsOr: ByteArray,
) {
    fun adjust(lexId: Int): Float = adjust[lexId] / LmPack.MILLI
    fun extraFlags(lexId: Int): Int = flagsOr[lexId].toInt() and 0xFF
    val size: Int get() = adjust.size

    companion object {
        fun read(bytes: ByteArray): DialectOverlay {
            val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            require(String(bytes, 0, 4, Charsets.US_ASCII) == "PBLO") { "not a dialect overlay" }
            require((buf.getShort(4).toInt() and 0xFFFF) == 1) { "unsupported overlay version" }
            val locale = String(bytes, 16, 16, Charsets.US_ASCII).trimEnd('\u0000')
            val n = buf.getInt(40)
            val logZ = buf.getFloat(88)
            val build = buf.getLong(96)
            val sec = HashMap<Int, Int>()
            for (i in 0 until buf.getInt(124)) sec[buf.getInt(128 + 24 * i)] = buf.getLong(128 + 24 * i + 8).toInt()
            val a = ShortArray(n)
            val aOff = sec.getValue(60)
            for (i in 0 until n) a[i] = buf.getShort(aOff + 2 * i)
            val fOff = sec.getValue(61)
            val f = ByteArray(n) { bytes[fOff + it] }
            return DialectOverlay(locale, build, logZ, a, f)
        }

        fun read(file: File): DialectOverlay = read(file.readBytes())
    }
}
