/*
 * Copyright (C) 2024 Studio Keys
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

package org.futo.inputmethod.latin.utils;

import org.futo.inputmethod.latin.SuggestedWords.SuggestedWordInfo;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Keeps suggestions useful while a word is being typed.
 *
 * <p>When the user is halfway through a word, the strip should contain that word (or something
 * very close to it). In practice the transformer language model only returns three candidates and
 * they are weighted much more heavily than dictionary results, so ordinary dictionary completions
 * are regularly pushed out of the visible slots even when they are exactly what is being typed.
 *
 * <p>This class promotes the best completion of the typed prefix to the slot directly below the
 * current top suggestion when none of the visible slots contains the typed word, a near miss of it,
 * or a completion of it. The top suggestion itself is never modified, so auto-correction decisions
 * made from the first result stay untouched.
 */
public final class CompletionRanker {
    private CompletionRanker() {
    }

    private static boolean isCompletionOf(final SuggestedWordInfo info, final String loweredTyped) {
        final String word = info.mWord.toLowerCase(Locale.ROOT);
        return word.length() > loweredTyped.length()
                && word.startsWith(loweredTyped)
                && word.indexOf(' ') < 0
                && word.indexOf('\t') < 0
                && word.indexOf('\n') < 0;
    }

    private static boolean isCloseToTyped(final SuggestedWordInfo info, final String loweredTyped) {
        final String word = info.mWord.toLowerCase(Locale.ROOT);
        if (word.equals(loweredTyped)) {
            return true;
        }
        final int tolerance = loweredTyped.length() <= 4 ? 1 : 2;
        return levenshteinDistance(word, loweredTyped) <= tolerance;
    }

    private static int levenshteinDistance(final String lhs, final String rhs) {
        final int len1 = lhs.length();
        final int len2 = rhs.length();
        int[] previous = new int[len2 + 1];
        int[] current = new int[len2 + 1];

        for (int j = 0; j <= len2; j++) {
            previous[j] = j;
        }

        for (int i = 1; i <= len1; i++) {
            current[0] = i;
            for (int j = 1; j <= len2; j++) {
                final int cost = lhs.charAt(i - 1) == rhs.charAt(j - 1) ? 0 : 1;
                current[j] = Math.min(
                        Math.min(current[j - 1] + 1, previous[j] + 1),
                        previous[j - 1] + cost);
            }
            final int[] swap = previous;
            previous = current;
            current = swap;
        }

        return previous[len2];
    }

    /**
     * Makes sure a completion of what is being typed can actually be seen in the suggestion strip.
     *
     * @param results The suggestion results that will be handed to the strip. Modified in place.
     * @param typedWord What the user has typed so far.
     * @param visibleSuggestionSlots How many of the highest ranked results are visible next to the
     *                               typed word.
     */
    public static void promoteCompletionOfTypedWord(final SuggestionResults results,
            final String typedWord, final int visibleSuggestionSlots) {
        if (results == null || results.size() < 2) {
            return;
        }

        final String typed = typedWord == null ? "" : typedWord.trim();
        if (typed.length() < 2) {
            return;
        }
        for (int i = 0; i < typed.length(); i++) {
            if (Character.isWhitespace(typed.charAt(i))) {
                return;
            }
        }

        final String lowered = typed.toLowerCase(Locale.ROOT);
        final List<SuggestedWordInfo> ranked = new ArrayList<>(results);
        if (ranked.size() < 2) {
            return;
        }

        final int visible = Math.min(Math.max(visibleSuggestionSlots, 1), ranked.size());
        for (int i = 0; i < visible; i++) {
            final SuggestedWordInfo info = ranked.get(i);
            if (isCloseToTyped(info, lowered) || isCompletionOf(info, lowered)) {
                return;
            }
        }

        SuggestedWordInfo candidate = null;
        for (final SuggestedWordInfo info : ranked) {
            if (isCompletionOf(info, lowered)) {
                candidate = info;
                break;
            }
        }
        if (candidate == null) {
            return;
        }

        final SuggestedWordInfo top = ranked.get(0);
        final SuggestedWordInfo second = ranked.get(1);

        // The promotion has to fit strictly between the top suggestion and the runner up; skip when
        // their scores are adjacent, and never touch the top suggestion itself.
        if ((long) top.mScore - (long) second.mScore < 2L) {
            return;
        }
        if (second.mScore >= Integer.MAX_VALUE - 1) {
            return;
        }

        final int targetScore = second.mScore + 1;
        if (candidate.mScore >= targetScore) {
            return;
        }

        results.remove(candidate);
        final SuggestedWordInfo promoted = new SuggestedWordInfo(
                candidate.mWord,
                candidate.mPrevWordsContext,
                targetScore,
                candidate.mKindAndFlags & ~SuggestedWordInfo.KIND_FLAG_APPROPRIATE_FOR_AUTO_CORRECTION,
                candidate.mSourceDict,
                candidate.mIndexOfTouchPointOfSecondWord,
                candidate.mAutoCommitFirstWordConfidence);
        promoted.mOriginatesFromTransformerLM = candidate.mOriginatesFromTransformerLM;
        results.add(promoted);
    }
}
