import gettext
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

locale_dir = Path(__file__).parent.parent / "i18n"

gettext.install(
    "ritm_annotation",
    localedir=str(locale_dir),
)

gettext.gettext = _

logger.debug(
    _('Loading locale data from "{locale_folder}"').format(
        locale_folder=locale_dir
    )
)
