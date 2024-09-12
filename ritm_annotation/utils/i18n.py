import gettext
import logging
from gettext import gettext as _
from pathlib import Path

logger = logging.getLogger(__name__)

locale_dir = Path(__file__).parent.parent / "i18n"


gettext.bindtextdomain(
    "ritm_annotation",
    localedir=str(locale_dir),
)

logger.debug(
    _('Loading locale data from "{locale_folder}"').format(
        locale_folder=locale_dir
    )
)
