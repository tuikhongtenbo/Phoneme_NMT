import unicodedata

def split_syllable_IPA(ipa: str) -> list[str]:
    ipa = unicodedata.normalize('NFKC', ipa)
    ipa = ipa.replace("ţ", "t").replace("ɑ˞", "ɑːr").replace("ɔ˞", "ɔːr")
    ipa = ipa.replace('̭', "").replace('̬', "").replace('˞', "")
    ipa = ipa.replace("ˌ", " ").replace("ˈ", " ").replace("·", " ")
    ipa = ipa.replace(",", "")
    ipa = ipa.replace(".", " ")
    ipa = ipa.replace("ɝː", "ɜːr").replace("ɝ", "ɜːr").replace("ʊɚ", "ʊər").replace("ɚ", "ər")
    syllables = ipa.split()

    return syllables

def split_syllable_to_phoneme(syllable: str) -> tuple[list[str], list[str], list[str]]:
    consonants = [
        "tʃ", "dʒ", "b", "h", "hw", "j", "k", "kj", 
        "l", "lj", "m", "n", "nj", "ŋ", "t", "tj", 
        "θ", "θj", "v", "w", "z", "zj", "ʒ", "d", 
        "dj", "ð", "p", "r", "ʃ", "f", "ɡ", "s", "sj"
    ]

    vowels = [
        "aɪər", "aʊər", "ɔɪər", "ɛər", "ɪər", 
        "ʊər", "eɪ", "aɪ", "aʊ", "oʊ", "ɔɪ", 
        "ɑːr", "ær", "ɒr", "ɛr", "ɪr", "ɔːr", 
        "ʊr", "ɜːr", "ʌr" , "iə", "uə", "ər", 
        "oʊ", "əl", "ɑː", "ɒ", "æ", "ɛ", "ɜː", 
        "ɪ", "i", "iː", "ɔː", "ʊ", "uː", "u", 
        "ʌ",  "ə", "e"
    ]

    def get_phoneme(syllable: str, phonemes: list[str]) -> list[str]:
        if syllable == "":
            return [], syllable
        
        selected_phonemes = []
        times = 0
        while True:
            times += 1
            for phoneme in phonemes:
                if syllable.startswith(phoneme):
                    selected_phonemes.append(phoneme)
                    syllable = syllable[len(phoneme):]
                    times -= 1
                    break

            if times == 1:
                break

        return selected_phonemes, syllable

    initals, syllable = get_phoneme(syllable, consonants)
    nucluses, syllable = get_phoneme(syllable, vowels)
    finals, syllable = get_phoneme(syllable, consonants)
    
    return "".join(initals), "".join(nucluses), "".join(finals)

def convert_English_IPA_to_phoneme(IPA: str) -> list[str]:
    syllables = split_syllable_IPA(IPA)
    phonemes = []
    for syllable in syllables:
        phonemes.append(split_syllable_to_phoneme(syllable))
    
    return phonemes