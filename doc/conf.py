# -*- coding: utf-8 -*-
#
import os
import subprocess
import sys
import warnings
import re
from datetime import datetime
from pathlib import Path

# If extensions (or modules to document with autodoc) are in another
# directory, add these directories to sys.path here. If the directory
# is relative to the documentation root, use os.path.abspath to make it
# absolute, like shown here.
sys.path.insert(0, os.path.abspath('sphinxext'))

import sphinx_gallery

# doc/sphinxext is on sys.path (see above); the class lives there so that
# Sphinx can pickle the config for a parallel build.
from gallery_order import SubSectionTitleOrder

HERE = Path(__file__).parent.resolve()
(HERE / "_static").mkdir(exist_ok=True)

# -- General configuration ---------------------------------------------------
root_doc = 'index'
master_doc = 'index'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'numpydoc',
    'nbsphinx',
    'sphinx_gallery.gen_gallery',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.imgconverter',
    'matplotlib.sphinxext.plot_directive',
    'sphinx.ext.autosectionlabel',
    # Markdown pages are documentation too: doc/labelizer.md and the
    # workshop's README are written in it, and without a parser they are
    # simply absent from the rendered docs.
    'myst_parser',
]

for _ext in ('sphinx_copybutton', 'sphinx_design', 'sphinxext.opengraph'):
    try:
        __import__(_ext)
        extensions.append(_ext)
    except ImportError:
        pass

nbsphinx_allow_errors = True

# Executing the notebooks and the example gallery needs the compiled module,
# several gigabytes of structure/photon data and, for a few pages, the
# network. That is a thing to do on a workstation, not on every CI run, so
# execution is opt-in:
#
#     IMP_BFF_DOCS_EXECUTE_EXAMPLES=1 make -C doc html
#
# With the switch off, notebooks are rendered from the outputs they carry in
# the repository and sphinx-gallery still produces a page, the highlighted
# source and the .py/.ipynb downloads for every example -- only freshly
# computed figures are missing.
_execute_examples = os.environ.get(
    'IMP_BFF_DOCS_EXECUTE_EXAMPLES', '0'
).lower() in ('1', 'true', 'yes', 'on')

# 'auto' would run any notebook that has no stored outputs.
nbsphinx_execute = 'auto' if _execute_examples else 'never'

# BibTEex
extensions += ['sphinxcontrib.bibtex']
bibtex_bibfiles = ['references.bib']

# this is needed for some reason...
# see https://github.com/numpy/numpydoc/issues/69
numpydoc_class_members_toctree = False

# For maths, use mathjax by default and svg if NO_MATHJAX env variable is set
# (useful for viewing the doc offline)
if os.environ.get('NO_MATHJAX'):
    extensions.append('sphinx.ext.imgmath')
    imgmath_image_format = 'svg'
    mathjax_path = ''
else:
    extensions.append('sphinx.ext.mathjax')
    mathjax_path = ('https://cdn.jsdelivr.net/npm/mathjax@3/es5/'
                    'tex-chtml.js')

autodoc_default_options = {
    'members': True,
    'inherited-members': True
}

# # Add any paths that contain templates here, relative to this directory.
templates_path = ['templates']

# generate autosummary even if no references
autosummary_generate = True

# The suffix of source filenames.
source_suffix = {'.rst': 'restructuredtext', '.md': 'markdown'}

# -- Project information -----------------------------------------------------
project = u'IMP.bff'
copyright = (
    f'2021 - {datetime.now().year}, IMP developers'
)
# GitHub Pages is the current public documentation site. The previous custom
# domain remains linked from the landing page as the legacy documentation.
html_baseurl = 'https://tpeulen.github.io/IMP.bff/dev/'
# The compiled module is optional for a documentation build. Nothing in
# doc/ uses `automodule`/`autoclass` -- the C++ API is Doxygen's job and the
# narrative pages are hand-written .rst/.ipynb -- so the only thing the
# import ever provided was the version string. Keeping it mandatory would
# force every docs build (including CI) to compile IMP.bff first, which buys
# nothing. Import it when it happens to be there, fall back otherwise.
try:
    import IMP.bff

    version = IMP.bff.__version__
except Exception:  # noqa: BLE001 - any import failure is a soft failure here
    version = os.environ.get("IMP_BFF_DOCS_VERSION", "dev")
    # So that a stray autodoc/autosummary directive degrades into a stub
    # instead of failing the build.
    autodoc_mock_imports = ["IMP", "IMP.bff"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    u'_build',
    'Thumbs.db',
    '.DS_Store',
    # Jupyter's autosave copies are not pages; without this nbsphinx picks
    # them up and every notebook is built twice under a second title.
    '**/.ipynb_checkpoints',
    # sphinx-gallery writes both a .rst and a .ipynb per example; reading the
    # notebook as a source file too is the "multiple files found" error.
    'auto_examples/**/*.ipynb',
    'auto_examples/*.ipynb',
]

# The reST default role (used for this markup: `text`) to use for all
# documents.
default_role = 'literal'

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

def _first_available_theme(candidates):
    for modname, theme_name in candidates:
        try:
            __import__(modname)
            return theme_name
        except ImportError:
            pass
    return "alabaster"


html_theme = _first_available_theme([
    ("pydata_sphinx_theme", "pydata_sphinx_theme"),
    ("furo", "furo"),
    ("sphinx_rtd_theme", "sphinx_rtd_theme"),
])

html_title = f"{project} v{version}"
html_short_title = "IMP.bff"
html_logo = "logos/imp_bff-logo.png" if (HERE / "logos" / "imp_bff-logo.png").exists() else None
html_favicon = "logos/favicon.ico" if (HERE / "logos" / "favicon.ico").exists() else None
html_static_path = ["_static"] if (HERE / "_static").exists() else []

html_theme_options = {
    "navigation_depth": 3,
}

if html_theme == "pydata_sphinx_theme":
    html_theme_options.update({
        "show_toc_level": 2,
        "github_url": "https://github.com/tpeulen/IMP.bff",
        "switcher": {
            "json_url": "https://tpeulen.github.io/IMP.bff/switcher.json",
            "version_match": os.environ.get("DOCS_VERSION", "dev"),
            "check_switcher": False,
        },
        "navbar_end": [
            "version-switcher", "theme-switcher", "navbar-icon-links",
        ],
        "show_version_warning_banner": True,
    })
    html_sidebars = {
        "**": ["search-field.html", "sidebar-nav-bs.html", "sourcelink.html"]
    }

# If false, no module index is generated.
html_domain_indices = True

# If false, no index is generated.
html_use_index = True

# Output file base name for HTML help builder.
htmlhelp_basename = 'IMP.bff.doc'

# If true, the reST sources are included in the HTML build as _sources/name.
html_copy_source = False

# Adds variables into templates
html_context = {}
# finds latest release highlights and places it into HTML context for
# index.html
release_highlights_dir = Path(__file__).parent / ".." / "examples" / "release_highlights"
# Finds the highlight with the latest version number
_highlights = sorted(release_highlights_dir.glob("plot_release_highlights_*.py"))
# templates/index.html dereferences both unconditionally.
html_context["release_highlights"] = "contents"
html_context["release_highlights_version"] = version
if _highlights:
    latest_highlights = _highlights[-1].with_suffix('').name
    html_context["release_highlights"] = \
        f"auto_examples/release_highlights/{latest_highlights}"

    # get version from higlight name assuming highlights have the form
    # plot_release_highlights_0_22_0
    highlight_version = ".".join(latest_highlights.split("_")[-3:-1])
    html_context["release_highlights_version"] = highlight_version

# -- Options for LaTeX output ------------------------------------------------
latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    'preamble': r"""
        \usepackage{amsmath}\usepackage{amsfonts}\usepackage{bm}
        \usepackage{morefloats}\usepackage{enumitem} \setlistdepth{10}
        \let\oldhref\href
        \renewcommand{\href}[2]{\oldhref{#1}{\hbox{#2}}}
        """
}

trim_doctests_flags = True

# intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/{.major}'.format(sys.version_info), None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'matplotlib': ('https://matplotlib.org/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'joblib': ('https://joblib.readthedocs.io/en/latest/', None),
    'seaborn': ('https://seaborn.pydata.org/', None),
    'tttrlib': ('https://docs.peulen.xyz/tttrlib', None)
}


sphinx_gallery_conf = {
    'doc_module': 'IMP.bff',
    'show_memory': False,
    'examples_dirs': ['../examples'],
    'gallery_dirs': ['auto_examples'],
    'subsection_order': SubSectionTitleOrder('../examples'),
    # avoid generating too many cross-links
    'inspect_global_variables': False,
    'remove_config_comments': True,
    'plot_gallery': _execute_examples,
    # The gallery is the `plot_*.py` files. Everything else under examples/
    # is a helper module imported by a notebook (bd.py, experiment.py,
    # cbm56.py, ...) or a script that is run by hand; sphinx-gallery would
    # demand a module docstring from each of them and abort the whole build
    # on the first one that has none.
    'ignore_pattern': r'(?:^|[\\/])(?:\.ipynb_checkpoints[\\/].*|(?!plot_)[^\\/]*)\.py$',
    # Never fail the whole build over one example; the traceback is rendered
    # into the example's own page instead.
    'abort_on_example_error': False,
}
if not _execute_examples:
    # Nothing was run, so there is no output to compare against.
    sphinx_gallery_conf['filename_pattern'] = r'(?!.*)'

# The following dictionary contains the information used to create the
# thumbnails for the front page of the scikit-learn home page.
# key: first image in set
# values: (number of plot in set, height of thumbnail)
carousel_thumbs = {
    'examples_structure_flexfit.gif': 600,
    'sphx_glr_plot_k2_thumb.png': 600,
    'sphx_glr_plot_path_maps_002': 600
}


def make_carousel_thumbs(app, exception):
    """produces the final resized carousel images"""
    if exception is not None:
        return
    print('Preparing carousel images')
    image_dir = os.path.join(app.builder.outdir, '_images')
    for glr_plot, max_width in carousel_thumbs.items():
        image = os.path.join(image_dir, glr_plot)
        if os.path.exists(image):
            c_thumb = os.path.join(image_dir, glr_plot[:-4] + '_carousel.png')
            sphinx_gallery.gen_rst.scale_image(image, c_thumb, max_width, 190)


def filter_search_index(app, exception):
    if exception is not None:
        return

    # searchindex only exist when generating html
    if app.builder.name != 'html':
        return

    print('Removing methods from search index')

    searchindex_path = os.path.join(app.builder.outdir, 'searchindex.js')
    with open(searchindex_path, 'r') as f:
        searchindex_text = f.read()

    searchindex_text = re.sub(r'{__init__.+?}', '{}', searchindex_text)
    searchindex_text = re.sub(r'{__call__.+?}', '{}', searchindex_text)

    with open(searchindex_path, 'w') as f:
        f.write(searchindex_text)


# Hack to get kwargs to appear in docstring #18434
# TODO: Remove when https://github.com/sphinx-doc/sphinx/pull/8234 gets
# merged
from sphinx.util import inspect  # noqa
from sphinx.ext.autodoc import ClassDocumenter  # noqa


class PatchedClassDocumenter(ClassDocumenter):

    def _get_signature(self):
        old_signature = inspect.signature

        def patch_signature(subject, bound_method=False, follow_wrapped=True):
            # changes the default of follow_wrapped to True
            return old_signature(subject, bound_method=bound_method,
                                 follow_wrapped=follow_wrapped)
        inspect.signature = patch_signature
        result = super()._get_signature()
        inspect.signature = old_signature
        return result


def setup(app):
    app.registry.documenters['class'] = PatchedClassDocumenter
    app.connect('build-finished', filter_search_index)


warnings.filterwarnings("ignore", category=UserWarning,
                        message='Matplotlib is currently using agg, which is a'
                                ' non-GUI backend, so cannot show the figure.')
