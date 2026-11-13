# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Evan Gress

"""Allow ``python -m dualect`` as an alias for the ``dualect`` command."""

import sys

from dualect.cli import main

sys.exit(main())
