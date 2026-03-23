"""
GEO acquisition CLI for scMetaIntel-Hub.

Delegates to the vendored geodh module (originally from GEO-DataHub).
"""

from .geodh import main as geodh_main


def main():
    geodh_main()


if __name__ == "__main__":
    main()
