"""What this is, who wrote it, and where it lives: the About box and the
line in the corner that opens it.

The start screen and the editor both carry the line, so it is built here
once and the two cannot drift apart.
"""

from PySide6.QtCore import QSize, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QLabel, QMessageBox

from . import __version__
from .sheets import centre

# A desktop app opening a link sends no Referer header -- there is no page it
# came from -- so the only way the site can tell is in the URL itself.
CAME_FROM = "utm_source=film-developer&utm_medium=app&utm_content=%s"

AUTHOR, SITE = "Jose A. Hurtado", "elcacharrista.com"


def identity():
    from .cli import identity
    who, site = identity()
    # A frozen build may carry no package metadata to read, and a corner that
    # says "www." is worse than one that says nothing.
    return who or AUTHOR, site or SITE


def about_box(parent=None):
    """The name and version, what it does, the author and the address,
    with the icon beside them.

    `sheets.say` drew it before, without the icon or the version, and with
    the address as plain text.
    """
    from .gui import app_icon
    who, site = identity()
    b = QMessageBox(parent)
    b.setWindowTitle("About film-developer")
    b.setTextFormat(Qt.RichText)
    b.setText(
        '<div style="font-size:15px"><b>film-developer</b> %s</div>'
        '<div style="color:#8a8a90; margin-top:6px">Digitise film '
        'negatives<br>photographed with a camera.</div>'
        '<div style="margin-top:12px">%s<br>'
        '<a style="color:#5ac8fa" href="https://www.%s">www.%s</a></div>'
        % (__version__, who, site, site))
    b.setIconPixmap(app_icon().pixmap(QSize(64, 64)))
    b.setStandardButtons(QMessageBox.Ok)
    b.adjustSize()
    centre(b)
    b.exec()


def signature(parent=None):
    """The name, the version and the address, as one clickable line.

    Two links in one line: the name opens About, the address opens the
    browser. Handled by hand rather than by `openExternalLinks`, which would
    try to open `#about` as a URL.

    Two links, so two colours: whichever one the pointer is on lights up and
    the other does not. A link's colour is in its `style`, which Qt does not
    restyle on hover, so the text is painted again from `linkHovered`.
    """
    site = identity()[1]
    lab = QLabel()
    lab.setTextFormat(Qt.RichText)
    lab.setCursor(Qt.PointingHandCursor)
    lab.setContentsMargins(0, 0, 0, 0)
    lab.setToolTip("what this is, and where it lives")
    lab.linkActivated.connect(
        lambda u: about_box(parent) if u == "#about"
        else QDesktopServices.openUrl(QUrl(u)))
    html = (
        '<a href="#about" style="color:%%s; text-decoration:none">'
        '<b>film-developer</b>'
        '<span style="color:#4a4c53"> %s</span></a>'
        '<span style="color:#4a4c53"> · </span>'
        '<a href="https://www.%s/?%s" '
        'style="color:%%s; text-decoration:none">www.%s</a>'
    ) % (__version__, site, CAME_FROM % __version__, site)
    DIM, LIT = ("#8a8a90", "#5a5a60"), ("#e8e8ea", "#9ee0ff")

    def paint(hovered=""):
        lab.setText(html % (LIT[0] if hovered == "#about" else DIM[0],
                            LIT[1] if hovered.startswith("http") else DIM[1]))
    paint()
    lab.linkHovered.connect(paint)
    return lab

