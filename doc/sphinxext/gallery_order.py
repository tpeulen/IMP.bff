"""Gallery subsection ordering for sphinx-gallery.

This lives in its own module rather than in ``conf.py`` because Sphinx
pickles the configuration when it builds in parallel (``-j auto``), and a
class defined inside ``conf.py`` has no importable qualified name:

    _pickle.PicklingError: Can't pickle <class 'SubSectionTitleOrder'>:
    attribute lookup SubSectionTitleOrder on builtins failed

``doc/sphinxext`` is on ``sys.path`` (see ``conf.py``), so from here the
class pickles and unpickles like any other.
"""

from __future__ import annotations

import os
import re


class SubSectionTitleOrder:
    """Sort example gallery by title of subsection.

    Assumes README.rst exists for all subsections and uses the subsection
    with dashes, '---', as the adornment.
    """

    def __init__(self, src_dir):
        self.src_dir = src_dir
        self.regex = re.compile(r"^([\w ]+)\n-", re.MULTILINE)

    def __repr__(self):
        return '<%s>' % (self.__class__.__name__,)

    def __call__(self, directory):
        src_path = os.path.normpath(os.path.join(self.src_dir, directory))

        # Forces Release Highlights to the top
        if os.path.basename(src_path) == "release_highlights":
            return "0"

        for name in ("README.rst", "README.txt"):
            readme = os.path.join(src_path, name)
            try:
                with open(readme, 'r') as f:
                    content = f.read()
            except FileNotFoundError:
                continue
            title_match = self.regex.search(content)
            if title_match is not None:
                return title_match.group(1)
            return directory
        return directory
