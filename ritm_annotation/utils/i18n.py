import gettext
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

locale_dir = Path(__file__).parent.parent / "i18n"

translations = gettext.translation(
    'ritm_annotation',
    localedir=str(locale_dir),
    fallback=True,
    # languages=['en'],
)
translations.install()

logger.debug(_('Loading locale data from "{locale_folder}"').format(locale_folder=locale_dir))
