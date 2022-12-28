#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function

import unittest
import logging


def _check_extension_is_built():
    from pathlib import Path
    import sys

    if sys.version_info[0] >= 3:
        import builtins
    else:
        import __builtin__ as builtins
    builtins.__PKG_SETUP__ = True

    from gofft.distutils import get_extensions

    all_extensions_are_built = True
    exts = get_extensions('gofft')

    for ext in exts:
        fext = '.pyd' if sys.platform == 'win32' else '.so'
        fn = Path(*ext.name.split('.'))
        fn = Path(fn.parent).glob("%s*%s" % (fn.name, fext))
        if len(list(fn)) == 0:
            all_extensions_are_built = False
            break

    del builtins.__PKG_SETUP__
    return all_extensions_are_built


def _build_ext():
    from subprocess import Popen
    cmd = "python3 setup.py build_ext --inplace clean --all"
    proc = Popen(cmd.split(' '))
    out, _ = proc.communicate()


def run_test():
    if not _check_extension_is_built():
        logging.error('Some extensions are not built. Trying to build them...')
        _build_ext()

    loader = unittest.TestLoader()
    tests = loader.discover('.')
    unittest.TextTestRunner().run(tests)


if __name__ == '__main__':
    run_test()
