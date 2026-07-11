import Foundation

/// Per-language sub-tokenizer layout for the Magpie multilingual model.
///
/// `MagpieMultiTokenizer.AggregatedTTSTokenizer` (in NeMo) concatenates each
/// sub-tokenizer's vocab end-to-end into a single 2360-entry table and adds
/// a per-tokenizer **offset** when encoding:
///
/// ```
/// agg_id = sub_tokenizer.encode(text) + tokenizer_offsets[tokenizer_name]
/// ```
///
/// We reproduce the offsets here. They were derived from `<pad>` boundary
/// scanning of `tokenizer/en.json` (which ships the full aggregated vocab),
/// matching the order in the Magpie `model_config.yaml`:
///
///   english_phoneme       offset    0, size  96  (BaseTokenizer IPA)
///   spanish_phoneme       offset   96, size 103  (BaseTokenizer IPA)
///   german_phoneme        offset  199, size 150  (BaseTokenizer IPA)
///   mandarin_phoneme      offset  349, size 109  (ChinesePhonemesTokenizer)
///   japanese_phoneme      offset  458, size 175  (JapanesePhonemeTokenizer)
///   french_chartokenizer  offset  633, size 384  (byT5-small)
///   hindi_chartokenizer   offset 1017, size 191  (BaseTokenizer char)
///   italian_phoneme       offset 1208, size 384  (byT5-small)
///   vietnamese_phoneme    offset 1592, size 384  (byT5-small)
///   text_ce_tokenizer     offset 1976, size 384  (byT5-small)
public enum MagpieSubVocab {

    public static func offset(for language: MagpieLanguage) -> Int {
        switch language {
        case .english:    return 0
        case .spanish:    return 96
        case .german:     return 199
        case .chinese:    return 349
        case .japanese:   return 458
        case .french:     return 633
        case .hindi:      return 1017
        case .italian:    return 1208
        case .vietnamese: return 1592
        }
    }

    public static func size(for language: MagpieLanguage) -> Int {
        switch language {
        case .english:    return 96
        case .spanish:    return 103
        case .german:     return 150
        case .chinese:    return 109
        case .japanese:   return 175
        case .french:     return 384
        case .hindi:      return 191
        case .italian:    return 384
        case .vietnamese: return 384
        }
    }

    public static func usesByT5(_ language: MagpieLanguage) -> Bool {
        switch language {
        case .french, .italian, .vietnamese: return true
        default: return false
        }
    }
}
