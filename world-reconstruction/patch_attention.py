"""Give upstream WorldMirror a PyTorch SDPA fallback when FlashAttention is absent."""

from __future__ import annotations

import sys
from pathlib import Path


root = Path(sys.argv[1])
attention = (
    root
    / "hyworld2/worldrecon/hyworldmirror/models/layers/attention.py"
)
source = attention.read_text(encoding="utf-8")

old_import = """except ImportError:
    from flash_attn.flash_attn_interface import flash_attn_func as flash_attn_func_v2
    _USE_FLASH_ATTN_V3 = False
"""
new_import = """except ImportError:
    try:
        from flash_attn.flash_attn_interface import flash_attn_func as flash_attn_func_v2
        _USE_FLASH_ATTN_V3 = False
        _HAS_FLASH_ATTN = True
    except ImportError:
        flash_attn_func_v2 = None
        flash_attn_func_v3 = None
        _USE_FLASH_ATTN_V3 = False
        _HAS_FLASH_ATTN = False
else:
    _HAS_FLASH_ATTN = True
"""
old_condition = "if q.dtype==torch.bfloat16 or q.dtype==torch.float16:"
new_condition = (
    "if _HAS_FLASH_ATTN and "
    "(q.dtype == torch.bfloat16 or q.dtype == torch.float16):"
)

if old_import not in source:
    raise RuntimeError("Upstream attention import changed; refusing an unsafe patch")
if old_condition not in source:
    raise RuntimeError("Upstream attention condition changed; refusing an unsafe patch")

source = source.replace(old_import, new_import, 1)
source = source.replace(old_condition, new_condition, 1)
attention.write_text(source, encoding="utf-8")
print(f"Patched optional FlashAttention fallback in {attention}")
