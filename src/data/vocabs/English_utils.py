import unicodedata
import re

class EnglishIPA:
    Vowels = [
        # foreign diphthongs
        "ɪa",
        # English diphthongs
        "aɪər", "aʊər", "ɔɪər", "ɛər", "ɪər", 
        "ʊər", "eɪ", "aɪ", "aʊ", "oʊ", "ɔɪ",
        "ɑr", "ær", "ɒr", "ɛr", "er", "ɪr", 
        "ɔːr", "ʊr", "ɜːr", "ʌr" , "ər", "əl",
        "iː", "ɔː", "uː", "ɜː", "ɑː", 
        # foreign long vowel
        "oː", "æː",
        # foreign dipthongs
        "ɑæ", "əːr", "ɜr",
        # English monothongs
        "ɒ", "æ", "ɛ", "i", "ɪ", "ʊ", 
        "u", "ʌ",  "ə", "e",
        # foreign monothongs
        "a", "o", "ɑ", "ɔ",
    ]

    Consonants = [
        "tʃ", "dʒ", "tr", "dj", "lj",
        "nj", "sj", "tj", "θj", "zj",
        "b", "h", "j", "k",  "s",
        "l", "m", "n", "ŋ", "t", 
        "θ", "v", "w", "z", "ʒ", "d", 
        "ð", "p", "r", "ʃ", "f", "ɡ",
        # foreign consonants
        "β", "x", "c", "ɕ", "ɲ"
    ]

def preprocess_IPA(ipa: str) -> str:
    # remove all mark IPA
    ipa = unicodedata.normalize('NFD', ipa)
    ipa = "".join(
        char for char in ipa if unicodedata.category(char) != 'Mn'
    )

    ipa = ipa.replace("ɐ", "ɑ").replace("ţ", "t").replace("ɑ˞", "ɑːr").replace("ɔ˞", "ɔr").replace("ɚr", "eːr").replace("ɝr", "ɜːr").replace("ɹ", "r")
    # handling some false cases of LLM
    ipa = ipa.replace("ᵻ", "i").replace("ɾ", "t").replace("or", "ɔːr").replace("ʔ", "tə").replace("n̩", "n").replace("ɫ", "l")
    ipa = ipa.replace("-", "").replace("ʤ", "dʒ").replace(";", ".").replace("/", "").replace("ɟ", "j")

    ipa = ipa.replace('̭', "").replace('̬', "").replace('˞', "")
    ipa = ipa.replace("ɝ", "ɜːr").replace("ʊɚ", "ʊər").replace("ɚ", "əːr")
    ipa = ipa.replace(",", "")
    # the case of /deɪvɪtsː/
    ipa = ipa.replace("sː", "s")
    # the case of /ʃː/
    ipa = ipa.replace("ʃː", "ʃ")
    # the case of zː
    ipa = ipa.replace("zː", "z")

    return ipa

def split_syllable_to_phoneme(syllable: str) -> tuple[list[str], list[str], list[str]]:
    syllable = preprocess_IPA(syllable)

    vowels = EnglishIPA.Vowels

    if syllable == "":
        return [], syllable

    start_idx = 0
    end_idx = len(syllable)
    for vowel in vowels:
        if vowel in syllable:
            start_idx = syllable.index(vowel)
            end_idx = start_idx + len(vowel)
            break

    initial = syllable[:start_idx]
    vowel = syllable[start_idx:end_idx]
    final = syllable[end_idx:]

    return initial, vowel, final

def split_IPA_to_syllable(ipa: str) -> list[str]:
    ipa = preprocess_IPA(ipa)

    consonants = EnglishIPA.Consonants

    vowels = EnglishIPA.Vowels

    # ensure that all IPA characters are in consonants and vowels
    all_English_IPA_characters = set(list("".join(consonants) + "".join(vowels) + "ˌˈ·.'"))
    for character in ipa:
        if character not in all_English_IPA_characters:
            raise Exception(f"'{ipa}' contains character '{character}' not in the English IPA")

    # the special case where all phonemes are vowels (such as "aaaaaaa" /ɐææææ/)
    tmp_ipa = ipa
    idx = 0
    while idx < len(vowels):
        vowel = vowels[idx]
        if tmp_ipa.startswith(vowel):
            tmp_ipa = tmp_ipa[len(vowel):]
            idx = 0
        else:
            idx += 1

    if tmp_ipa == "":
        syllables = []
        while len(ipa) > 0:
            for vowel in vowels:
                if ipa.startswith(vowel):
                    syllables.append(vowel)
                    ipa = ipa[len(vowel):]
                    break
        
        return syllables

    if re.search(r"[ˌˈ·.']", ipa):
        ipa = ipa.replace("ˌ", " ").replace("ˈ", " ").replace("·", " ").replace("'", " ")
        ipa = ipa.replace(".", " ")
        syllables = ipa.split()

        return syllables

    syllable_dicts = []
    original_ipa = ipa
    ipa_len = len(ipa)
    while len(ipa) > 0:
        syllable_dict = {
            "initials": [],
            "nucleus": []
        }

        _break = False
        times_of_loop = 0
        while True:

            times_of_loop += 1
            if times_of_loop > 10:
                raise Exception(f"Infinite loop occurs. The given IPA is '{original_ipa}'. The remaining phonemes are '{ipa}'")
    
            for consonant in consonants:
                if ipa.startswith(consonant):
                    syllable_dict["initials"].append(consonant)
                    ipa = ipa[len(consonant):]

            # there is nothing to find
            if len(ipa) == 0:
                break
            
            # we can not find any initials
            if len(syllable_dict["initials"]) == 0:
                break
            
            # we meet the first vowel
            for vowel in vowels:
                if ipa.startswith(vowel):
                    _break = True
                    break

            if _break:
                break

        for vowel in vowels:
            if ipa.startswith(vowel):
                syllable_dict["nucleus"].append(vowel)
                ipa = ipa[len(vowel):]
                break

        syllable_dicts.append(syllable_dict)

        if ipa_len == len(ipa):
            raise Exception(f"Infinite loop occurs. The given IPA is '{original_ipa}'. The remaining phonemes are '{ipa}'")
        else:
            ipa_len = len(ipa)

    syllables = []
    for ith, syllable_dict in enumerate(syllable_dicts):
        syllable = []
        if ith == 0:
            syllable.append("".join(syllable_dict["initials"]))
            syllable.append("".join(syllable_dict["nucleus"]))
            syllables.append(syllable)
        else:
            initials = syllable_dict["initials"]
            if len(initials) > 1:
                # update the previous syllable
                previous_syllable: list = syllables[ith-1]
                previous_syllable.append(initials[0])
                initials = initials[1:]
                syllables[ith-1] = previous_syllable
                # the remaining initials become the intial of the current syllable
                syllable.append("".join(initials))
                syllable.append("".join(syllable_dict["nucleus"]))
                syllables.append(syllable)
            else:
                syllable.append("".join(initials))
                syllable.append("".join(syllable_dict["nucleus"]))
                syllables.append(syllable)

    if len(syllables) > 1:
        last_syllable: list[str] = syllables[-1]
        previous_last_syllable: list[str] = syllables[-2]
        if len(last_syllable[1]) == 0:  # the nucleus is empty
            if len(previous_last_syllable) == 3:
                final = previous_last_syllable[-1] + last_syllable[0]
                previous_last_syllable[-1] = final
            else:
                final = last_syllable[0]
                previous_last_syllable.append(final)
            
            # ignore the last syllable
            syllables = syllables[:-1]
            # update the last syllable
            syllables[-1] = previous_last_syllable

    # merge all phonemes together
    syllables = ["".join(syllable) for syllable in syllables]

    return syllables

def convert_English_IPA_to_phoneme(IPA: str) -> list[str]:
    syllables = split_IPA_to_syllable(IPA)
    phonemes = []
    for syllable in syllables:
        phonemes.append(split_syllable_to_phoneme(syllable))
    
    return phonemes
