import subprocess
from unittest.mock import patch

import numpy as np
import pytest

from scripts.preflight import assess, bounded_command, version_issues, write_new
from scripts.platform_guard import constraints_text, protected_versions, verify_guard
from scripts.compare_devices import compare_arrays, require_identity

BASE = {"torch": "2.1.0+cpu", "torchvision": "0.16.0+cpu", "torch-npu": None}


def test_cpu_preflight_pass():
    status, checks = assess("cpu", BASE, {"available": True, "device_count": 1})
    assert status == "pass"


@pytest.mark.parametrize("probe", [{"error": "missing plugin"}, {"available": False, "device_count": 0}])
def test_npu_preflight_missing_or_unavailable_fails(probe):
    assert assess("npu", BASE, probe)[0] == "fail"


def test_npu_discovery_does_not_certify_compatibility():
    versions = dict(BASE, **{"torch-npu": "2.1.0.post10"})
    assert assess("npu", versions, {"available": True, "device_count": 1})[0] == "warn"


@pytest.mark.parametrize("change", [
    {"torchvision": "0.17.0"}, {"torch-npu": "2.4.0"}, {"torch": "2.1.0+cu121"}])
def test_npu_version_inconsistency_fails(change):
    versions = dict(BASE, **{"torch-npu": "2.1.0.post10"})
    versions.update(change)
    assert any(status == "fail" for status, _ in version_issues(versions, "npu"))


def test_platform_build_tag_mismatch_rejected():
    versions = dict(BASE, torchvision="0.16.0+cu121")
    assert any(status == "fail" for status, _ in version_issues(versions, "cpu"))


def test_bounded_command_timeout():
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("probe", 1)):
        result = bounded_command(["probe"], 1)
    assert result["exit_code"] is None and "timeout" in result["error"]


def test_guard_keeps_local_build_and_detects_replacement():
    pins = protected_versions(BASE, "cpu")
    assert "torch==2.1.0+cpu\n" in constraints_text(pins)
    guard = {"packages": pins}
    assert verify_guard(guard, BASE)["status"] == "pass"
    assert verify_guard(guard, dict(BASE, torch="2.1.0+cu121"))["status"] == "fail"
    assert verify_guard(guard, dict(BASE, **{"torch-npu": "2.1.0"}))["status"] == "fail"
    with pytest.raises(ValueError):
        protected_versions(BASE, "npu")


def test_new_files_have_portable_line_endings(tmp_path):
    path = tmp_path / "constraints.txt"
    text = constraints_text(protected_versions(BASE, "cpu"))
    write_new(path, text)
    assert path.read_bytes() == text.encode("utf-8")
    with pytest.raises(FileExistsError):
        write_new(path, text)


def test_guard_invalid_requirement_rejected():
    with pytest.raises(ValueError):
        constraints_text({"torch": "2.1.0\nmalicious-package"})


def test_numerical_same_and_perturbed():
    reference = {"logits": np.arange(8, dtype=np.float32), "loss": np.array(0.5, dtype=np.float32)}
    assert compare_arrays(reference, reference, 0, 0)["success"]
    changed = {k: v.copy() for k,v in reference.items()}
    changed["logits"][1] += 0.1
    report = compare_arrays(reference, changed, 1e-5, 1e-6)
    assert not report["success"] and report["outputs"]["logits"]["max_abs_error"] > 0.09


@pytest.mark.parametrize("actual", [np.zeros(3, dtype=np.float32), np.zeros(2, dtype=np.float64), np.array([np.nan, 0], dtype=np.float32)])
def test_numerical_invalid_outputs_rejected(actual):
    with pytest.raises(ValueError):
        compare_arrays({"logits": np.zeros(2, dtype=np.float32)}, {"logits": actual}, 1e-3, 1e-4)


@pytest.mark.parametrize("key", ["checkpoint_sha256", "input_sha256", "image_ids", "preprocessing", "classes", "pipeline_sha256"])
def test_reference_identity_mismatch_rejected(key):
    base = {key: "a"}
    require_identity(base, base)
    with pytest.raises(ValueError, match="identity mismatch"):
        require_identity(base, {key: "b"})
