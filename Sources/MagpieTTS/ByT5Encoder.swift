import Foundation

/// byT5-style byte encoder for Magpie's French / Italian / Vietnamese
/// tokenisers (and the unused-here text_ce_tokenizer).
///
/// `model_config.yaml` defines those three languages as
/// `AutoTokenizer(pretrained_model="google/byt5-small")`. HuggingFace
/// `ByT5Tokenizer` encodes text as UTF-8 bytes, mapping each byte X to
/// native byT5 id `X + 3` (pad=0, eos=1, unk=2 occupy slots 0–2; bytes
/// 0x00–0xFF occupy slots 3–258; <extra_id_*> tokens occupy 259–383).
///
/// `MagpieMultiTokenizer.AggregatedTTSTokenizer` concatenates each
/// sub-tokenizer's vocab and offsets the encode result, so the final
/// aggregated id for byte X under language L is:
///
///   `agg_id = X + 3 + sub_vocab_offset(L)`
///
/// (i.e. `byte + 1211` for IT, `byte + 636` for FR, `byte + 1595` for VI).
///
/// We keep the formula simple: UTF-8 encode, look up each byte's offset.
public enum MagpieByT5Encoder {

    /// Encode a string to Magpie *aggregated* vocab IDs for a byT5-using
    /// language. The `language` selects the per-tokenizer offset.
    public static func encode(_ text: String, language: MagpieLanguage) -> [Int] {
        precondition(MagpieSubVocab.usesByT5(language),
                     "ByT5 encoder called for non-byT5 language \(language)")
        let offset = MagpieSubVocab.offset(for: language)
        // byT5 native ids: pad=0, eos=1, unk=2, byte 0x00 → 3, … byte 0xFF → 258.
        // Aggregated id = native id + offset = (byte + 3) + offset.
        let bias = offset + 3
        var ids: [Int] = []
        for byte in text.utf8 {
            ids.append(Int(byte) + bias)
        }
        return ids
    }
}
