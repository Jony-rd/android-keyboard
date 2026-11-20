package dev.pboard.engine.lm

/**
 * Locale-specific case mapping and prefix folding, loaded from the language pack so that the device uses
 * exactly the rules the pack was built with (no ASCII or English assumptions).
 *
 * - [lowerCodePoint]: pack overrides first (e.g. Turkish `I -> ı`, `İ -> i`), then Unicode simple lowercase.
 * - [fold]: the completion key: lowercase + the pack's fold table (diacritics stripped unless the locale keeps
 *   them, apostrophes removed, `ß -> ss` in German...). Characters absent from the table fold to their
 *   lowercase.
 *
 * Code points below [TABLE] use flat lookup tables (no boxing); others fall back to maps.
 */
class LocaleRules(
    private val caseOverrides: Map<Int, Int>,
    private val foldTable: Map<Int, String>,
) {
    private val lowerTab = IntArray(TABLE) { cp -> caseOverrides[cp] ?: Character.toLowerCase(cp) }
    /** Folded form per code point: null = the lowercase code point itself. */
    private val foldTab = arrayOfNulls<String>(TABLE).also { t ->
        for (cp in 0 until TABLE) {
            val f = foldTable[cp] ?: foldTable[lowerTab[cp]]
            if (f != null) t[cp] = f
        }
    }

    fun lowerCodePoint(cp: Int): Int = if (cp < TABLE) lowerTab[cp] else caseOverrides[cp] ?: Character.toLowerCase(cp)

    /** Folded form of [cp]: either a string (possibly empty) or null meaning "the code point [lowerCodePoint]". */
    fun foldOf(cp: Int): String? {
        if (cp < TABLE) return foldTab[cp]
        return foldTable[cp] ?: foldTable[lowerCodePoint(cp)]
    }

    fun lower(s: CharSequence): String {
        val sb = StringBuilder(s.length)
        var i = 0
        while (i < s.length) {
            val cp = Character.codePointAt(s, i)
            sb.appendCodePoint(lowerCodePoint(cp))
            i += Character.charCount(cp)
        }
        return sb.toString()
    }

    /** Appends the fold key of [s] to [out]. */
    fun foldInto(s: CharSequence, out: StringBuilder) {
        var i = 0
        while (i < s.length) {
            val cp = Character.codePointAt(s, i)
            val f = foldOf(cp)
            if (f != null) out.append(f) else out.appendCodePoint(lowerCodePoint(cp))
            i += Character.charCount(cp)
        }
    }

    fun fold(s: CharSequence): String = StringBuilder(s.length).also { foldInto(s, it) }.toString()

    companion object {
        private const val TABLE = 0x3000

        /** Parses "from<TAB>to" lines (UTF-8 text sections of the pack). */
        fun parse(caseText: String, foldText: String): LocaleRules {
            val case = HashMap<Int, Int>()
            for (line in caseText.split('\n')) {
                val tab = line.indexOf('\t')
                if (tab <= 0) continue
                case[line.codePointAt(0)] = line.codePointAt(tab + 1)
            }
            val fold = HashMap<Int, String>()
            for (line in foldText.split('\n')) {
                val tab = line.indexOf('\t')
                if (tab <= 0) continue
                fold[line.codePointAt(0)] = line.substring(tab + 1)
            }
            return LocaleRules(case, fold)
        }
    }
}
