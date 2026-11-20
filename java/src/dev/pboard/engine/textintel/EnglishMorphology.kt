package dev.pboard.engine.textintel

import java.io.InputStream

/**
 * Verb and adjective inflections for the grammar checker (asset `textintel/en_morph.txt`, built by
 * tools/textintel/build_morph.py from AGID; lowercase). "be" is built in.
 */
class EnglishMorphology private constructor(
    private val verbs: Map<String, Verb>,
    private val byForm: Map<String, List<Verb>>,
    private val comparatives: Set<String>,
) {
    /** The five forms of a verb; [alsoNoun]: its base or -s form is also a common noun ("work", "works"). */
    class Verb(val base: String, val past: String, val participle: String, val ing: String, val s3: String, val alsoNoun: Boolean)

    fun verb(base: String): Verb? = verbs[base]

    /** Verbs having [form] as one of their forms ("went" -> go; "saw" -> see and saw). */
    fun verbsWithForm(form: String): List<Verb> = byForm[form].orEmpty()

    /** A comparative or superlative adjective form ("taller", "best"). */
    fun isComparative(word: String) = word in comparatives

    val verbCount get() = verbs.size

    companion object {
        fun parse(text: String): EnglishMorphology {
            val verbs = HashMap<String, Verb>()
            val comp = HashSet<String>()
            for (line in text.lineSequence()) {
                if (line.isEmpty() || line[0] == '#') continue
                val p = line.split(' ')
                when (p[0]) {
                    "V" -> if (p.size >= 6) verbs[p[1]] = Verb(p[1], p[2], p[3], p[4], p[5], p.getOrNull(6) == "n")
                    "A" -> if (p.size >= 4) { comp += p[2]; comp += p[3] }
                }
            }
            val byForm = HashMap<String, MutableList<Verb>>()
            for (v in verbs.values) for (f in setOf(v.base, v.past, v.participle, v.ing, v.s3)) byForm.getOrPut(f) { ArrayList(2) } += v
            return EnglishMorphology(verbs, byForm, comp)
        }

        fun read(input: InputStream): EnglishMorphology = parse(input.bufferedReader().readText())
    }
}
