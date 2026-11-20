package dev.pboard.engine.textintel

/**
 * One token of typed text: a word (letters, digits and inner apostrophes: "it's", "o'clock", "3pm") or a single
 * punctuation/symbol character. [start]/[end] are UTF-16 offsets into the text.
 */
class TextToken(val text: String, val start: Int, val end: Int, val isWord: Boolean) {
    val lower: String = text.lowercase()
    override fun toString() = text
}

/** A sentence: tokens [from, to) of the token list, and whether it ended with . ! ? or a line break. */
class TextSentence(val from: Int, val to: Int, val terminated: Boolean)

object TextTokens {
    private fun isWordChar(c: Char) = c.isLetterOrDigit() || Character.getType(c) == Character.NON_SPACING_MARK.toInt()
    private fun isApostrophe(c: Char) = c == '\'' || c == '’'

    /** Tokens of [text]. Whitespace separates; apostrophes inside a word stay in it ("don't"); "’" reads as "'". */
    fun tokenize(text: String): List<TextToken> {
        val out = ArrayList<TextToken>()
        var i = 0
        val n = text.length
        while (i < n) {
            val c = text[i]
            when {
                c.isWhitespace() -> i++
                isWordChar(c) -> {
                    val s = i
                    while (i < n && (isWordChar(text[i]) ||
                            (isApostrophe(text[i]) && i + 1 < n && text[i + 1].isLetter() && i > s))) i++
                    out += TextToken(text.substring(s, i).replace('’', '\''), s, i, true)
                }
                Character.isHighSurrogate(c) && i + 1 < n -> {
                    out += TextToken(text.substring(i, i + 2), i, i + 2, false)
                    i += 2
                }
                else -> {
                    out += TextToken(c.toString(), i, i + 1, false)
                    i++
                }
            }
        }
        return out
    }

    /** Splits tokens into sentences at . ! ? … and line breaks (an abbreviation's period also splits: harmless here). */
    fun sentences(text: String, tokens: List<TextToken>): List<TextSentence> {
        val out = ArrayList<TextSentence>()
        var from = 0
        for (i in tokens.indices) {
            val t = tokens[i]
            val endPunct = !t.isWord && t.text[0] in ".!?…"
            val nextBreak = i + 1 < tokens.size && text.substring(t.end, tokens[i + 1].start).contains('\n')
            if (endPunct) {
                // "..." or "?!" end once, at their last character
                if (i + 1 < tokens.size && !tokens[i + 1].isWord && tokens[i + 1].text[0] in ".!?…" && tokens[i + 1].start == t.end) continue
                out += TextSentence(from, i + 1, true)
                from = i + 1
            } else if (nextBreak) {
                out += TextSentence(from, i + 1, true)
                from = i + 1
            }
        }
        if (from < tokens.size) out += TextSentence(from, tokens.size, text.substring(tokens.last().end).contains('\n'))
        return out
    }
}
