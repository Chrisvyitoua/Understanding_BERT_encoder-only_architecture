# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT model module initialization with lazy loading.

LAZY LOADING PATTERN:
=====================
This module implements lazy loading to improve import performance. Instead of loading
all BERT components immediately, modules are loaded only when first accessed.

Benefits:
1. Faster startup: Initial import is nearly instant
2. Lower memory: Only loads what's actually used
3. Reduced dependencies: Doesn't require all deps unless used

How it works:
- TYPE_CHECKING block: Used by type checkers (mypy, pylance) for static analysis
- Runtime block: Replaces module with _LazyModule that loads components on-demand

Example usage:
    from transformers.models.bert import BertModel, BertTokenizer
    # At this point, nothing is actually loaded yet

    model = BertModel.from_pretrained("bert-base-uncased")
    # Now BertModel and its dependencies are loaded
"""

# TYPE_CHECKING is a special constant that's True during type checking, False at runtime
from typing import TYPE_CHECKING

# Lazy loading infrastructure from HuggingFace utilities
from ...utils import _LazyModule  # Enables deferred module loading
from ...utils.import_utils import define_import_structure  # Discovers available exports


# TYPE CHECKING BLOCK
# ===================
# This block executes ONLY during static type analysis (not at runtime)
# It imports all components normally so type checkers understand the module structure
if TYPE_CHECKING:
    # Import all configuration classes (BertConfig, etc.)
    from .configuration_bert import *
    # Import all model classes (BertModel, BertForSequenceClassification, etc.)
    from .modeling_bert import *
    # Import all tokenizer classes (BertTokenizer, BertTokenizerFast, etc.)
    from .tokenization_bert import *

# RUNTIME BLOCK
# =============
# This block executes during actual program execution
# It replaces this module with a lazy-loading proxy
else:
    import sys

    # Get the path to this __init__.py file
    _file = globals()["__file__"]

    # Replace the current module in sys.modules with a lazy-loading version
    # When someone accesses an attribute (like BertModel), _LazyModule:
    # 1. Checks if it's been loaded yet
    # 2. If not, imports the real module
    # 3. Returns the requested attribute
    # 4. Caches it for future use
    sys.modules[__name__] = _LazyModule(
        __name__,                              # Module name (transformers.models.bert)
        _file,                                 # Path to this file
        define_import_structure(_file),        # Discover what can be imported
        module_spec=__spec__                   # Module specification metadata
    )
