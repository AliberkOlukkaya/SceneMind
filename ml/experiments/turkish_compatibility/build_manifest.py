"""Build the frozen, source-disjoint Turkish development manifest."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).with_name("calibration_v1.json")


def query(number, turkish, english, route, start, end, *traits):
    return {
        "query_id": number,
        "turkish_query": turkish,
        "english_equivalent": english,
        "route": route,
        "relevant_intervals": [[start, end]],
        "linguistic_traits": list(traits),
    }


def q(prefix, rows):
    return [query(f"{prefix}-{index:02d}", *row) for index, row in enumerate(rows, 1)]


SOURCES = [
    {
        "source_id": "wikipedia101-translation",
        "source_group": "commons-wikipedia101-turkish-series",
        "split": "calibration",
        "domain": "software_demo",
        "file": "Vikipedi 101 - Çeviri (video 10).webm",
        "page": "https://commons.wikimedia.org/wiki/File:Vikipedi_101_-_%C3%87eviri_(video_10).webm",
        "download_url": "https://upload.wikimedia.org/wikipedia/commons/d/d4/Vikipedi_101_-_%C3%87eviri_%28video_10%29.webm",
        "license": "CC BY-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "attribution": "Kurmanbek",
        "sha256": "5f8c4116c00026d657f9abb30cd6d18bebd8f8679a182397f30fb8a08f6b3840",
        "duration_seconds": 67.28,
        "language": "tr",
        "asr_characteristics": "Whisper-tiny detected Turkish; 7 segments. Core words are usable, with substantial suffix and proper-word errors.",
        "queries": q("trn", [
            ("Başka dildeki maddeyi çevirmeyi anlattığı yer nerede?", "Where does he explain translating an article from another language?", "SPEECH", 8, 17, "suffix", "question"),
            ("Çeviriler butonundan ne zaman bahsediyor?", "When does he mention the translations button?", "SPEECH", 19, 24, "technical_term", "question"),
            ("Çeviri ekranını açmayı açıkladığı kısmı bul.", "Find where he explains opening the translation screen.", "SPEECH", 23, 36, "omitted_subject", "suffix"),
            ("Wikipedia'de dünyayı değiştirmekten bahsettiği bölüm.", "The part where he talks about changing the world on Wikipedia.", "SPEECH", 35, 53, "apostrophe", "colloquial"),
            ("Ekranda iki sütunlu çeviri sayfasının açık olduğu yer", "The place where a two-column translation page is open on screen", "VISUAL", 38, 46, "turkish_characters", "spatial"),
            ("Vikipedi ana sayfasını göster.", "Show the Wikipedia home page.", "VISUAL", 19, 24, "imperative"),
            ("Beyaz kapüşonlu sunucunun göründüğü an", "The moment the presenter in a white hoodie is visible", "VISUAL", 10, 18, "suffix"),
            ("Çeviri aracını anlatırken iki dilli ekranı gösterdiği yer", "Where he explains the translation tool while showing the bilingual screen", "HYBRID", 32, 47, "while", "mixed"),
            ("Madde çevirmeyi anlatıp Museum of Design Atlanta sayfasını gösterdiği kısım", "The part where he discusses translating an article and shows the Museum of Design Atlanta page", "HYBRID", 39, 45, "mixed_technical", "omitted_subject"),
            ("ayarları gösterdiği yer", "where he shows the settings", "VISUAL", 24, 35, "ambiguous", "colloquial"),
        ]),
    },
    {
        "source_id": "wikipedia101-upload",
        "source_group": "commons-wikipedia101-turkish-series",
        "split": "calibration",
        "domain": "tutorial",
        "file": "Vikipedi 101 - Wikimedia Commons'a medya yükleme (video 9).webm",
        "page": "https://commons.wikimedia.org/wiki/File:Vikipedi_101_-_Wikimedia_Commons%27a_medya_y%C3%BCkleme_(video_9).webm",
        "download_url": "https://upload.wikimedia.org/wikipedia/commons/d/dd/Vikipedi_101_-_Wikimedia_Commons%27a_medya_y%C3%BCkleme_%28video_9%29.webm",
        "license": "CC BY-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "attribution": "Kurmanbek",
        "sha256": "18b116e6823d42027a0383f4ef87c05cdd65e11f539a55bf6b0e63ae80af8ba6",
        "duration_seconds": 62.28,
        "language": "tr",
        "asr_characteristics": "Whisper-tiny detected Turkish; 12 segments. Upload steps remain searchable despite repeated Commons transcription errors.",
        "queries": q("upl", [
            ("Fotoğraf yüklemeyi anlattığı kısım nerede?", "Where does he explain uploading a photo?", "SPEECH", 8, 16, "suffix", "question"),
            ("Yükle butonundan ne zaman bahsediyor?", "When does he mention the upload button?", "SPEECH", 15, 21, "technical_term"),
            ("kendi çalışmam seçeneğini açıkladığı yer", "where he explains the own work option", "SPEECH", 26, 30, "omitted_subject"),
            ("Creative Commons lisansını anlattığı bölümü bul", "Find the section where he explains the Creative Commons license", "SPEECH", 29, 34, "mixed_technical"),
            ("Kırmızı tren fotoğrafının olduğu yükleme formunu göster.", "Show the upload form with the red train photo.", "VISUAL", 34, 50, "color", "imperative"),
            ("ekranda takvimin açık olduğu an", "the moment the calendar is open on screen", "VISUAL", 44, 50, "suffix"),
            ("Wikimedia Commons ana sayfasının göründüğü bölüm", "The section where the Wikimedia Commons home page is visible", "VISUAL", 19, 24, "mixed_technical"),
            ("Lisansı anlatırken yükleme sihirbazını gösterdiği yer", "Where he explains the license while showing the upload wizard", "HYBRID", 29, 38, "while", "mixed"),
            ("Kategori eklemekten bahsedip formu doldurduğu kısım", "The part where he mentions adding a category while filling the form", "HYBRID", 40, 50, "colloquial", "mixed"),
            ("dosya seçimini gösterdiği yer", "where he shows file selection", "HYBRID", 21, 29, "ambiguous", "omitted_subject"),
        ]),
    },
    {
        "source_id": "mitx-course-overview",
        "source_group": "commons-mitx-6002-course-overview",
        "split": "calibration",
        "domain": "presentation",
        "file": "course-overview-6.002x.webm",
        "page": "https://commons.wikimedia.org/wiki/File:Course_Overview_(6.002x).webm",
        "download_url": "https://upload.wikimedia.org/wikipedia/commons/1/12/Course_Overview_%286.002x%29.webm",
        "license": "CC BY 2.0",
        "license_url": "https://creativecommons.org/licenses/by/2.0/",
        "attribution": "mitxvtest / MITx",
        "sha256": "b80c402211d80949ebe7e81a539649e3cebcc9c3bb7b6c0a5cec8a0ac9344bf3",
        "duration_seconds": 662.398,
        "language": "en",
        "asr_characteristics": "Whisper-tiny detected English; 111 segments with usable technical terms and several course-number errors.",
        "queries": q("mit", [
            ("Maxwell denklemlerinden bahsettiği yer nerede?", "Where does he mention Maxwell's equations?", "SPEECH", 40, 59, "suffix", "technical_term"),
            ("Ohm kanununu açıkladığı kısmı bul.", "Find the part where he explains Ohm's law.", "SPEECH", 205, 264, "apostrophe", "omitted_subject"),
            ("Python'dan ne zaman bahsediyor?", "When does he mention Python?", "SPEECH", 497, 506, "apostrophe", "technical_term"),
            ("dijital soyutlamayı anlattığı bölüm", "the section where he explains digital abstraction", "SPEECH", 413, 444, "suffix"),
            ("Slaytta nature kutusunun solda olduğu yer", "The place where the nature box is on the left of the slide", "VISUAL", 108, 190, "mixed_technical", "spatial"),
            ("mavi kutularla çizilmiş ders haritasını göster", "show the course map drawn with blue boxes", "VISUAL", 300, 600, "color", "imperative"),
            ("kırmızı okların göründüğü karmaşık diyagram", "the complex diagram where red arrows are visible", "VISUAL", 350, 590, "color", "suffix"),
            ("ekrandaki purpose/use of science slaydı", "the purpose/use of science slide on screen", "VISUAL", 18, 58, "mixed_technical"),
            ("Python'ı anlatırken programlama dilleri kutusunu gösterdiği yer", "Where he mentions Python while showing the programming languages box", "HYBRID", 497, 506, "apostrophe", "while"),
            ("dijital abstraction'dan bahsedip şemada gate'leri gösterdiği bölüm", "The section where he discusses digital abstraction and shows gates in the diagram", "HYBRID", 413, 449, "mixed_technical", "apostrophe"),
            ("işlemcilerden konuşurken mavi şemayı gösterdiği kısım", "The part where he talks about processors while showing the blue diagram", "HYBRID", 457, 532, "while", "suffix"),
            ("model kısmı", "the model part", "HYBRID", 276, 365, "ambiguous", "colloquial"),
        ]),
    },
    {
        "source_id": "phone-throwing",
        "source_group": "commons-gameplay-send-me-to-heaven",
        "split": "calibration",
        "domain": "ordinary_visual",
        "file": "phone-throwing.webm",
        "page": "https://commons.wikimedia.org/wiki/File:Gameplay_of_Send_Me_To_Heaven.webm",
        "download_url": "https://upload.wikimedia.org/wikipedia/commons/1/19/Gameplay_of_Send_Me_To_Heaven.webm",
        "license": "CC BY 3.0",
        "license_url": "https://creativecommons.org/licenses/by/3.0/",
        "attribution": "petrsvar / Carrotpop; S.M.T.H.",
        "sha256": "8d682ff80b87acf16c8b2e7eb25cc30b812ade8469bc4f921a25cbd7724819d9",
        "duration_seconds": 21.267,
        "language": "none",
        "asr_characteristics": "No speech segments; visual-only source.",
        "queries": q("phn", [
            ("sarı başlık ekranını bul", "find the yellow title screen", "VISUAL", 0, 10, "color", "imperative"),
            ("buz pistindeki iki kadını göster", "show the two women at the ice rink", "VISUAL", 10, 21, "turkish_characters"),
            ("telefonu havaya atan kadın", "the woman throwing a phone into the air", "VISUAL", 10, 21, "action"),
            ("siyah beyaz çekimde yürüyen insanlar", "people walking in black-and-white footage", "VISUAL", 10, 21, "color", "action"),
            ("kırmızı bereli kadının göründüğü an", "the moment the woman in a red beret is visible", "VISUAL", 10, 16, "color", "suffix"),
            ("buz pateni pistinin görüldüğü yer", "where the ice skating rink is visible", "VISUAL", 10, 21, "suffix"),
            ("Send Me To Heaven yazısı ekrandayken", "while the Send Me To Heaven text is on screen", "VISUAL", 0, 10, "mixed_technical", "while"),
            ("telefon kısmı", "the phone part", "VISUAL", 10, 21, "ambiguous", "colloquial"),
            ("kameraya doğru yürüyen iki kişiyi bul", "find two people walking toward the camera", "VISUAL", 10, 21, "action", "imperative"),
            ("parlak sarı arka planın olduğu bölüm", "the section with a bright yellow background", "VISUAL", 0, 10, "color"),
        ]),
    },
    {
        "source_id": "wikitongues-ela",
        "source_group": "commons-wikitongues-ela-turkish",
        "split": "heldout",
        "domain": "speech_heavy",
        "file": "WIKITONGUES- Ela speaking Turkish.webm",
        "page": "https://commons.wikimedia.org/wiki/File:WIKITONGUES-_Ela_speaking_Turkish.webm",
        "download_url": "https://upload.wikimedia.org/wikipedia/commons/d/d9/WIKITONGUES-_Ela_speaking_Turkish.webm",
        "license": "CC BY 3.0",
        "license_url": "https://creativecommons.org/licenses/by/3.0/",
        "attribution": "Wikitongues",
        "sha256": "3c9142b5b2926462a4dcedfc35488bdb1e75b898c9924d29b8377609b9af1647",
        "duration_seconds": 79.467,
        "language": "tr",
        "asr_characteristics": "Whisper-tiny detected Turkish; 17 segments. Accent and proper nouns cause heavy errors, while journalism, television and language topics remain partly usable.",
        "queries": q("ela", [
            ("Ankara'da üniversite okuduğunu söylediği yer", "where she says she studied at university in Ankara", "SPEECH", 17, 25, "apostrophe", "suffix"),
            ("gazetecilikten bahsettiği bölümü bul", "find the section where she talks about journalism", "SPEECH", 24, 37, "omitted_subject"),
            ("televizyonda çalıştığını anlattığı kısım", "the part where she explains that she works in television", "SPEECH", 32, 43, "suffix"),
            ("Türkçenin Kosova'da konuşulmasından ne zaman bahsediyor?", "When does she mention Turkish being spoken in Kosovo?", "SPEECH", 67, 79, "apostrophe", "question"),
            ("pembe bluzlu kadının göründüğü an", "the moment the woman in a pink blouse is visible", "VISUAL", 5, 79, "color"),
            ("kadının önünde pencere olan görüntüyü bul", "find the shot with a window behind the woman", "VISUAL", 5, 79, "spatial"),
            ("kameraya konuşan koyu saçlı kadın", "dark-haired woman speaking to the camera", "VISUAL", 5, 79, "action"),
            ("gazeteciliği anlatırken kameraya baktığı yer", "where she looks at the camera while discussing journalism", "HYBRID", 24, 37, "while"),
            ("belgeselden bahsederken pembe bluzlu kadının göründüğü bölüm", "the section where the woman in a pink blouse is visible while she discusses the documentary", "HYBRID", 46, 59, "while", "color"),
            ("Türkçe konuşulmasını anlatırken kadının ekranda olduğu kısım", "the part where the woman is on screen while discussing speaking Turkish", "HYBRID", 58, 79, "mixed"),
        ]),
    },
    {
        "source_id": "brave-wikipedian",
        "source_group": "commons-sosyalkafa-brave-wikipedian",
        "split": "heldout",
        "domain": "education",
        "file": "Cesur ol, Wikipedist ol!.webm",
        "page": "https://commons.wikimedia.org/wiki/File:Cesur_ol,_Wikipedist_ol!.webm",
        "download_url": "https://upload.wikimedia.org/wikipedia/commons/1/14/Cesur_ol%2C_Wikipedist_ol%21.webm",
        "license": "CC BY 3.0",
        "license_url": "https://creativecommons.org/licenses/by/3.0/",
        "attribution": "sosyalkafa tv",
        "sha256": "f5ef216d1333912b9a448d971b076d32c1631419e7e46c7e4b94fe5f621f7354",
        "duration_seconds": 115.829,
        "language": "tr",
        "asr_characteristics": "Whisper-tiny detected Turkish; 36 segments. Meaning is recoverable but Wikipedia and several policy terms are repeatedly distorted.",
        "queries": q("brv", [
            ("Herkesin düzeltme yapabileceğini söylediği yer", "where she says everyone can make edits", "SPEECH", 26, 33, "suffix"),
            ("tarafsız bakış açısından bahsettiği bölüm", "the section where she talks about neutral point of view", "SPEECH", 40, 50, "turkish_characters"),
            ("lisans koşullarını açıkladığı kısmı bul", "find the part where she explains license conditions", "SPEECH", 53, 69, "omitted_subject"),
            ("değişikliklerin geri alınabildiğini ne zaman söylüyor?", "When does she say edits can be reverted?", "SPEECH", 98, 111, "question", "suffix"),
            ("Wikipedia logosunun ekranda olduğu an", "the moment the Wikipedia logo is on screen", "VISUAL", 4, 9, "mixed_technical"),
            ("dünyayı tutan eller çizimini göster", "show the illustration of hands holding the world", "VISUAL", 34, 40, "imperative"),
            ("kadının iki yanında çizgi karakterlerin olduğu yer", "where cartoon characters appear on both sides of the woman", "VISUAL", 44, 50, "spatial"),
            ("ekranda gerekenler sağduyu ve saygı yazıları", "the required, common sense and respect words on screen", "VISUAL", 88, 95, "omitted_subject"),
            ("kuralları anlatırken iki elini açtığı bölüm", "the section where she opens both hands while explaining rules", "HYBRID", 24, 38, "while"),
            ("telif hakkını anlatırken kaynak çiziminin göründüğü yer", "where the source illustration is visible while she explains copyright", "HYBRID", 53, 62, "while"),
            ("katkı yapmaktan bahsederken yazı yazan kişi çizimini gösterdiği kısım", "the part where she discusses contributing while showing an illustration of someone writing", "HYBRID", 93, 105, "mixed"),
            ("temel taşlardan konuşurken taş görselini tuttuğu an", "the moment she holds a stone image while talking about pillars", "HYBRID", 72, 87, "colloquial", "mixed"),
        ]),
    },
    {
        "source_id": "base-jump",
        "source_group": "commons-quest-films-base-jump",
        "split": "heldout",
        "domain": "ordinary_visual",
        "file": "base-jump.webm",
        "page": "https://commons.wikimedia.org/wiki/File:Base_jump.webm",
        "download_url": "https://upload.wikimedia.org/wikipedia/commons/1/1b/Base_jump.webm",
        "license": "CC BY 3.0",
        "license_url": "https://creativecommons.org/licenses/by/3.0/",
        "attribution": "Quest Films",
        "sha256": "4eff2d9c0d62cea6f2f806754bd1fea9ca7a5daefbd6208ee2201f67cd6509e5",
        "duration_seconds": 37.004,
        "language": "none",
        "asr_characteristics": "No speech segments; visual-only source.",
        "queries": q("jmp", [
            ("kırmızı kasklı adamı bul", "find the man wearing a red helmet", "VISUAL", 0, 8, "color", "imperative"),
            ("kayalık arazide duran sporcular", "athletes standing on rocky terrain", "VISUAL", 0, 31, "suffix"),
            ("karanlık mağaraya atlayan kişiyi göster", "show the person jumping into a dark cave", "VISUAL", 4, 10, "action", "imperative"),
            ("mağara girişinin yukarıdan göründüğü an", "the moment the cave entrance is seen from above", "VISUAL", 4, 37, "spatial"),
            ("siyah bereli adamın kenarda durduğu yer", "where the man in a black beanie stands at the edge", "VISUAL", 10, 25, "color"),
            ("arkada araçların olduğu kayalık alan", "the rocky area with vehicles in the background", "VISUAL", 0, 20, "spatial"),
            ("iki kişinin atlayışa hazırlandığı bölüm", "the section where two people prepare for the jump", "VISUAL", 10, 31, "action"),
            ("çukur kısmı", "the hole part", "VISUAL", 4, 37, "ambiguous", "colloquial"),
        ]),
    },
]


def main():
    source_groups = {source["source_group"] for source in SOURCES}
    acceptance = json.loads(
        (ROOT / "ml/evaluation/personal_acceptance_manifest_v1.json").read_text(encoding="utf-8")
    )
    acceptance_hashes = {video["sha256"] for video in acceptance["videos"]}
    if source_groups & {
        "personal-lecture-cup-of-tea",
        "personal-demo-blender-interface",
        "personal-street-traffic",
    }:
        raise ValueError("personal source group leaked into Turkish development data")
    if acceptance_hashes & {source["sha256"] for source in SOURCES}:
        raise ValueError("personal media checksum leaked into Turkish development data")
    payload = {
        "schema_version": "1.0.0",
        "experiment": "turkish-compatibility-v1",
        "annotation_status": "frozen",
        "annotation_frozen_at": "2026-09-15T03:20:00+03:00",
        "selection_policy": "Architecture and parameters use calibration sources only. Heldout sources and the personal acceptance set cannot influence selection.",
        "personal_acceptance_manifest_sha256": hashlib.sha256(
            (ROOT / "ml/evaluation/personal_acceptance_manifest_v1.json").read_bytes()
        ).hexdigest(),
        "sources": SOURCES,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
