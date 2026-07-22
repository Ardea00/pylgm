import pylgm


def test_package_exports_version() -> None:
    assert pylgm.__version__ == "0.1.0"
