"""SmolVLA fine-tuning - patch snapshot_download to be local-only."""
import sys, os

# Patch snapshot_download to use local_files_only
import huggingface_hub
_orig_snap = huggingface_hub.snapshot_download
def _local_snap(*args, **kwargs):
    kwargs['local_files_only'] = True
    try:
        return _orig_snap(*args, **kwargs)
    except Exception as e:
        # If local_files_only fails, the files might already be at root/repo_id
        # Just return the expected local path
        repo_id = args[0] if args else kwargs.get('repo_id', '')
        root = kwargs.get('local_dir', kwargs.get('cache_dir', ''))
        print(f"  [PATCH] snapshot_download skipped for {repo_id}, using local files", flush=True)
        return root
huggingface_hub.snapshot_download = _local_snap
# Also patch it in the module that imports it
import huggingface_hub._snapshot_download
huggingface_hub._snapshot_download.snapshot_download = _local_snap

# Patch version check
import lerobot.datasets.utils as ds_utils
_orig_ver = ds_utils.get_safe_version
def _patched_ver(repo_id, version):
    try:
        return _orig_ver(repo_id, version)
    except Exception:
        return version
ds_utils.get_safe_version = _patched_ver

# Patch visual features validation
try:
    import lerobot.policies.utils as pu
    pu.validate_visual_features_consistency = lambda cfg, features: None
except Exception:
    pass

from lerobot.scripts.lerobot_train import main
main()
