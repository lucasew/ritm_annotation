import gettext
from pathlib import Path

locale_dir = Path(__file__).parent.parent / "i18n"
print(locale_dir)

translations = gettext.translation(
    'ritm_annotation',
    localedir=str(locale_dir),
    fallback=True,
    languages=['en'],
)
translations.install()
